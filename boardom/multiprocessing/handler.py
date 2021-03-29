# This based on the PyTorch DataLoader. Adapted for generic multiprocessing
# TODO: ADD COPYRIGHT AND LINK
import time
import types
import threading
import multiprocessing as python_multiprocessing
import asyncio
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch.utils.data import _utils as torch_data_utils
from torch.utils.data._utils import signal_handling
from torch._six import queue, string_classes
from .pin_memory import _pin_memory_loop
from .worker import _worker_loop


class PersistentProcessPool(object):
    __initialized = False

    def __init__(
        self,
        num_workers=1,
        pin_memory=False,
        pinned_device=torch.device('cuda', 0),
        worker_init_fn=None,
        multiprocessing_context=None,
        async_sleep=0.01,
    ):
        if num_workers < 1:
            raise ValueError('num_workers option should be positive')
        if multiprocessing_context is None:
            multiprocessing_context = multiprocessing

        self.num_workers = num_workers
        self._pin_memory = pin_memory
        self._pinned_device = pinned_device
        self._async_sleep = async_sleep

        self._worker_pids_set = False
        self._shutdown = False

        self._workers_done_event = multiprocessing_context.Event()

        self._task_id = 0
        self._worker_is_active = []
        self._task_queues = []
        self._worker_result_queues = []
        self._workers = []
        self._available_worker_queue = queue.Queue()
        for i in range(self.num_workers):
            task_queue = multiprocessing_context.Queue()
            result_queue = multiprocessing_context.Queue()
            w = multiprocessing_context.Process(
                target=_worker_loop,
                args=(
                    task_queue,
                    result_queue,
                    self._workers_done_event,
                    worker_init_fn,
                    i,
                ),
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._task_queues.append(task_queue)
            self._worker_result_queues.append(result_queue)
            self._workers.append(w)
            self._worker_is_active.append(True)
            self._available_worker_queue.put(i)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._results_queues = [queue.Queue() for _ in range(self.num_workers)]
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self._worker_result_queues,
                    self._results_queues,
                    self._pinned_device,
                    self._pin_memory_thread_done_event,
                    self._async_sleep,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._results_queues = self._worker_result_queues

        signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True

        self.__multiprocessing_context = multiprocessing_context
        self.__initialized = True

    def _maybe_fd_error(self):
        import tempfile
        import errno

        try:
            # Raise an exception if we are this close to the FDs limit.
            # Apparently, trying to open only one file is not a sufficient
            # test.
            # See NOTE [ DataLoader on Linux and open files limit ]
            fds_limit_margin = 10
            fs = [  # noqa: F841
                tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)
            ]
        except OSError as e:
            if e.errno == errno.EMFILE:
                raise RuntimeError(
                    "Too many open files. Communication with the"
                    " workers is no longer possible. Please increase the"
                    " limit using `ulimit -n` in the shell or change the"
                    " sharing strategy by calling"
                    " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                    " at the beginning of your code"
                )

    def _maybe_failed_workers(self):
        failed_workers = []
        for worker_id, w in enumerate(self._workers):
            if self._worker_is_active[worker_id] and not w.is_alive():
                failed_workers.append(w)
                self._shutdown_worker(worker_id)
        if len(failed_workers) > 0:
            pids_str = ', '.join(str(w.pid) for w in failed_workers)
            raise RuntimeError(
                'DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)
            )

    def _get_helper(self, worker_id, timeout):
        # Tries to fetch result from `self._results_queues` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `torch_data_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        try:
            result = self._results_queues[worker_id].get(timeout=timeout)
            if isinstance(result, ExceptionWrapper):
                result.reraise()
            return (True, result)
        except Exception as e:
            self._maybe_failed_workers()
            if isinstance(e, queue.Empty):
                return (False, None)
            self._maybe_fd_error()
            raise

    def _get(self, worker_id, timeout=0):
        # Fetches result from `self._results_queues`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._get_helper(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if timeout > 0:
            success, result = self._get_helper(worker_id, timeout=timeout)
            if success:
                self._available_worker_queue.put(worker_id)
                return result
            else:
                raise RuntimeError('Timed out after {timeout} seconds')
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, result = self._get_helper(
                    worker_id, timeout=torch_data_utils.MP_STATUS_CHECK_INTERVAL
                )
                if success:
                    self._available_worker_queue.put(worker_id)
                    return result
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._results_queues` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, result = self._get_helper(
                    worker_id, timeout=torch_data_utils.MP_STATUS_CHECK_INTERVAL
                )
                if success:
                    self._available_worker_queue.put(worker_id)
                    return result

    async def _get_async(self, worker_id):
        while (not self._pin_memory) or self._pin_memory_thread.is_alive():
            try:
                result = self._results_queues[worker_id].get_nowait()
            except Exception as e:
                self._maybe_failed_workers()
                if isinstance(e, queue.Empty):
                    await asyncio.sleep(self._async_sleep)
                    continue
                self._maybe_fd_error()
                raise
            if isinstance(result, ExceptionWrapper):
                result.reraise()
            self._available_worker_queue.put(worker_id)
            return result
        else:
            raise RuntimeError('Pin memory thread exited unexpectedly')

    async def _run_func(self, func):
        worker_id = await self._put_async(func)
        return await self._get_async(worker_id)

    async def run_async(self, *funcs):
        if len(funcs) == 1 and isinstance(funcs[0], types.GeneratorType):
            funcs = funcs[0]
        return await asyncio.gather(
            *[asyncio.create_task(self._run_func(f)) for f in funcs]
        )

    def run(self, *funcs):
        return asyncio.run(self.run_async(*funcs))

    async def _put_async(self, func):
        while True:
            try:
                available_worker_idx = self._available_worker_queue.get_nowait()
                self._task_queues[available_worker_idx].put(func)
                return available_worker_idx
            except queue.Empty:
                await asyncio.sleep(self._async_sleep)

    def _put(self, func):
        return asyncio.run(self._put_async(func))

    def _shutdown_worker(self, worker_id):
        # Mark a worker as having finished its work and dead, e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._worker_is_active[worker_id]

        # Signal termination to that specific worker.
        q = self._task_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.
        self._worker_is_active[worker_id] = False

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = torch_data_utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    for worker_id in range(len(self._workers)):
                        self._worker_result_queues[worker_id].put((None, None))

                    self._pin_memory_thread.join()
                    for worker_id in range(len(self._workers)):
                        self._worker_result_queues[worker_id].cancel_join_thread()
                        self._worker_result_queues[worker_id].close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self.num_workers` in case we error before starting all
                    # workers.
                    if self._worker_is_active[worker_id]:
                        self._shutdown_worker(worker_id)
                for w in self._workers:
                    w.join()
                for q in self._task_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if isinstance(multiprocessing_context, string_classes):
                valid_start_methods = multiprocessing.get_all_start_methods()
                if multiprocessing_context not in valid_start_methods:
                    raise ValueError(
                        (
                            'multiprocessing_context option '
                            'should specify a valid start method in {}, but got '
                            'multiprocessing_context={}'
                        ).format(valid_start_methods, multiprocessing_context)
                    )
                multiprocessing_context = multiprocessing.get_context(
                    multiprocessing_context
                )

            if not isinstance(
                multiprocessing_context, python_multiprocessing.context.BaseContext
            ):
                raise TypeError(
                    (
                        'multiprocessing_context option should be a valid context '
                        'object or a string specifying the start method, but got '
                        'multiprocessing_context={}'
                    ).format(multiprocessing_context)
                )

        self.__multiprocessing_context = multiprocessing_context

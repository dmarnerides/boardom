# This is from PyTorch
# TODO: ADD COPYRIGHT AND LINK

import torch
from torch._six import queue
from torch._utils import ExceptionWrapper
from torch.utils.data._utils import (
    signal_handling,
    MP_STATUS_CHECK_INTERVAL,
)
from torch.utils.data._utils.worker import WorkerInfo, ManagerWatchdog


def _worker_loop(input_queue, output_queue, done_event, init_fn, worker_id):
    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        watchdog = ManagerWatchdog()

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id)

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

        except Exception:
            init_exception = ExceptionWrapper(where=f'in process {worker_id}')
            if watchdog.is_alive():
                output_queue.put((None, init_exception))

        while watchdog.is_alive():
            try:
                func = input_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if func is None:
                # Received the final signal
                break
            elif done_event.is_set():
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            try:
                result = func()
            except Exception:
                # It is important that we don't store exc_info in a variable.
                # `ExceptionWrapper` does the correct thing.
                # See NOTE [ Python Traceback Reference Cycle Problem ]
                result = ExceptionWrapper(where=f'in process {worker_id}')
            output_queue.put(result)
            del result, func  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    output_queue.cancel_join_thread()
    output_queue.close()

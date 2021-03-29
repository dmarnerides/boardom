# This is (slightly adapted) from PyTorch:
# TODO: ADD COPYRIGHT AND LINK

import asyncio
import torch
from torch._six import queue, container_abcs, string_classes
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper


async def _single_task(in_queue, out_queue, device_id, done_event, async_sleep):
    while not done_event.is_set():
        try:
            task = in_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(async_sleep)
            continue
        if not done_event.is_set() and not isinstance(task, ExceptionWrapper):
            try:
                task = pin_memory(task)
            except Exception:
                task = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id)
                )
        while not done_event.is_set():
            try:
                out_queue.put(task, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del task


def _pin_memory_loop(in_queues, out_queues, device_id, done_event, async_sleep):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    asyncio.run(_run_tasks(in_queues, out_queues, device_id, done_event, async_sleep))


async def _run_tasks(in_queues, out_queues, device_id, done_event, async_sleep):
    await asyncio.gather(
        *[
            asyncio.create_task(
                _single_task(in_queue, out_queue, device_id, done_event, async_sleep)
            )
            for in_queue, out_queue in zip(in_queues, out_queues)
        ]
    )


def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    elif isinstance(data, container_abcs.Sequence):
        return [pin_memory(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data

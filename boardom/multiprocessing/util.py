import boardom as bd

#  import torch.distributed as dist
from multiprocessing import current_process


def is_main_process():
    return current_process().name == 'MainProcess'


def only_main_process(func):
    if is_main_process():
        return func
    else:
        return bd.null_function

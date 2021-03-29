import time
import signal
import sys
import threading
from collections.abc import Mapping, Generator, Sequence, Set
from contextlib import contextmanager
import boardom as bd


def apply(f, recurse=False):
    """Returns a function that applies `f` to members of an input (or to the input directly).

    Useful to use in conjuction with :func:`boardom.util.compose`

    Args:
         f (function): Function to be applied.

    Returns:
         callable: A function that applies 'f' to collections


    Example:
         >>> pow2 = boardom.util.apply(lambda x: x**2)
         >>> pow2(42)
         1764
         >>> pow2([1, 2, 3])
         [1, 4, 9]
         >>> pow2({'a': 1, 'b': 2, 'c': 3})
         {'a': 1, 'b': 4, 'c': 9}

    """

    def apply(objs):
        g = apply if recurse else f
        if isinstance(objs, Mapping):
            return type(objs)({key: g(val) for key, val in objs.items()})
        # Not using Collection since this will also include Tensors
        # and will recurse on tensor elements as well
        elif isinstance(objs, (Sequence, Set)):
            return type(objs)(g(x) for x in objs)
        else:
            return f(objs)

    return apply


class Singleton(type):
    # Not using a WeakValueDict so that the object persists even if out of scope
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class _Null(metaclass=Singleton):
    def __repr__(self):
        return 'bd.Null'

    def __bool__(self):
        return False


Null = _Null()


@contextmanager
def interrupt_guard(reason=None):
    class _SIG:
        sig = False

        @staticmethod
        def guard(signum, frame):
            bd.write()
            bd.log(
                'Received Interrupt in guarded section. '
                'Program will terminate when section is done. '
                'To terminate immediately use SIGKILL.'
            )
            if reason is not None:
                bd.write(f'Reason: {reason}')
            _SIG.sig = True

    try:
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _SIG.guard)
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if _SIG.sig:
            raise KeyboardInterrupt


# Only for main thread
class SignalHandler:
    funcs = {}
    previous_handler = {}

    def __init__(self, func, signum):
        if threading.current_thread() == threading.main_thread():
            if signum in self.funcs:
                self.funcs[signum].append(func)
            else:
                self.funcs[signum] = [func]
            if signum not in self.previous_handler:
                self.previous_handler[signum] = signal.getsignal(signum)

    def __call__(self, signum, frame):
        self._handle(signum, frame)
        sys.exit(0)

    def _handle(self, signum, frame):
        if threading.current_thread() == threading.main_thread():
            if signum in self.funcs:
                for fn in self.funcs[signum]:
                    fn()
        if signum in self.previous_handler:
            self.previous_handler[signum](signum, frame)


def identity(*args):
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    else:
        return args


def null_function(*args, **kwargs):
    pass


def str_or_none(x):
    if (x is not None) and (x is not Null) and (not isinstance(x, str)):
        raise TypeError('str2bool function expected string or None/bd.Null input')
    elif (x is None) or (x is Null):
        return x
    elif x.strip().lower() == 'none':
        return None
    else:
        return x

    return not (x is None or x.lower() in ['not', 'no', 'false', 'f', '', '0'])


def str2bool(x):
    """Converts a string to boolean type.

    Returns False if the string is any of
    ['None', 'not', 'no', 'false', 'f', '0'] with any
    capitalization (e.g. 'fAlSe'), or the empty string ('')
    or the value None or bd.Null.  All other strings are True.
    If the input is not a string or None/bd.Null raises a TypeError.

    """
    if (x is not None) and (x is not Null) and (not isinstance(x, str)):
        raise TypeError('str2bool function expected string or None/bd.Null input')
    return not (x is None or x.lower() in ['not', 'no', 'false', 'f', '', '0'])


def str_is_int(s):
    """Checks if a string can be converted to int."""
    if not isinstance(s, str):
        raise TypeError('str_is_int function only accepts strings.')
    try:
        int(s)
        return True
    except ValueError as e:
        if 'invalid literal' in str(e):
            return False
        else:
            raise e


class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.previous = self.start
        self.other = []

    def __call__(self, tag=None):
        self.other.append((tag, (time.time() - self.start)))

    def lap(self, tag=None):
        self(tag)

    def __str__(self):
        timings = '\n'.join(f'\t{k}: {v}' for k, v in self.other)
        return f'Timings:\n{timings}\n'


def recurse_get_elements(x):
    if isinstance(x, Mapping):
        for y in x.values():
            for z in recurse_get_elements(y):
                yield z
    # Not using Collection since this will also include Tensors
    # and will recurse on tensor elements as well
    elif isinstance(x, (Sequence, Set, Generator)):
        for y in x:
            for z in recurse_get_elements(y):
                yield z
    else:
        yield x


def timestamp(mode='datetime'):
    if mode == 'datetime':
        fmt = "%Y-%m-%dT%H:%M:%S"
    elif mode == 'date':
        fmt = "%Y-%m-%d"
    elif mode == 'time':
        fmt = "%H:%M:%S"
    else:
        raise RuntimeError(f'Invilid timestamp mode: {mode}')
    return time.strftime(fmt)

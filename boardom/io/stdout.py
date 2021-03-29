import sys
import logging
import shutil
import pprint
from tqdm import tqdm
import boardom as bd
from contextlib import contextmanager

STREAMS = ('stdin', 'stdout', 'stderr')

# Interface is simple, only write, flush and readline is supported currently
class StdStreamReplicator:
    def __init__(self, filename, stream='stdout'):
        if stream not in STREAMS:
            s = ', '.join([f'"{s}"' for s in STREAMS])
            raise RuntimeError(f'Expected stream to be one of {s}. Got: {stream}')
        self._previous_stream = None
        self._activated = False
        self._stream = stream
        self.activate(filename)

    def activate(self, filename):
        if not self._activated:
            self._previous_stream = getattr(sys, self._stream)
            # Use a logging filehandler (for correct management of file)
            fh = logging.FileHandler(filename, mode='a', encoding=None, delay=False)
            fh.terminator = ''
            self.file = fh.stream
            self._line_location = self.file.tell()
            setattr(sys, self._stream, self)
            self._activated = True

    def deactivate(self):
        if self._activated and self._previous_stream is not None:
            setattr(sys, self._stream, self._previous_stream)
            self._previous_stream = None
            self._activated = False
            self.file = None
            self._location = 0

    def write(self, msg, *args, **kwargs):
        self._previous_stream.write(msg, *args, **kwargs)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self._previous_stream.flush()

    def readline(self, *args, **kwargs):
        line = self._previous_stream.readline(*args, **kwargs)
        self.file.write(line)
        self.file.flush()
        return line

    def __getattr__(self, key):
        return getattr(self._previous_stream, key)


def replicate_std_stream(filename, stream='stdout'):
    return StdStreamReplicator(filename, stream=stream)


@contextmanager
def silent():
    bd.verbose = False
    try:
        yield
    finally:
        bd.verbose = True


def write(*msg, sep=' ', end='\n'):
    if bd.verbose:
        tqdm.write(f'{sep.join(str(x) for x in msg)}', end=end)


def log(msg):
    if bd.verbose:
        write('[boardom]', msg)


def warn(msg):
    if bd.verbose:
        write('[boardom - Warning]', msg)


def error(msg):
    if bd.verbose:
        write('[boardom - Error]', msg)
        write('Terminating')
        exit()


def print_model_cfg(cfg):
    if bd.verbose:
        pprint.pprint(cfg)


def print_separator(sep='-'):
    if bd.verbose:
        w, _ = shutil.get_terminal_size()
        write(sep[0] * w)


def print_model_info(model, name=None, indent=4, guides=True):
    if name is not None:
        print_separator()
        log(name.capitalize())
    print_separator()
    with bd.pretty_print(indent, guides):
        write(model)
    print_separator()
    log(f'Total Parameters: {bd.count_parameters(model, trainable=False):,}')
    num_trainable = bd.count_parameters(model, trainable=True)
    log(f'Trainable Parameters: {num_trainable:,}')
    print_separator()

import time
import boardom as bd
import inspect
from functools import partial


def _every_second(num_seconds):
    def ret(engine):
        if (time.time() - ret.start) > num_seconds:
            ret.start = time.time()
            return True
        return False

    ret.start = time.time()

    return ret


def _every_x_counter(name, count):
    def ret(engine):
        return (engine[name] % count) == 0

    return ret


ALIASES = {'epoch': 'training.epoch', 'training_step': 'training.global_step'}
# Callback must be f(engine) -> bool
def every(callback=None, **kwargs):
    funcs = []
    for key, val in kwargs.items():
        key = key.lower()

        # Convert minutes and hours to seconds
        if key in ['m', 'min', 'mins', 'minute', 'minutes']:
            key, val = 'seconds', val * 60
        if key in ['h', 'hr', 'hrs', 'hour', 'hours']:
            key, val = 'seconds', val * 3600

        if key in ['s', 'sec', 'secs', 'second', 'seconds']:
            funcs.append(_every_second(val))
        else:
            key = ALIASES.get(key, key)
            funcs.append(_every_x_counter(key, val))
    if callback is not None:
        funcs.append(callback)

    def mwfunc(engine, callback, **kwargs):
        if any(f(engine) for f in funcs):
            return callback(**kwargs)

    return middleware(mwfunc)


def middleware(mw):
    def wrapper(wrapped):
        if inspect.isclass(wrapped):
            raise RuntimeError('bd.middleware() can not be used on Classes.')
        if not (callable(wrapped) or isinstance(wrapped, staticmethod)):
            raise RuntimeError(
                'Ivalid use of bd.middleware(). Expected callable or staticmethod.'
            )
        middleware = [mw]
        if isinstance(wrapped, partial):
            f = wrapped.func
            if hasattr(f, '__func__') and (f.__func__ == bd.Engine.event):
                name = wrapped.args[0]
                err = f'Can not apply middleware as "{name}" is an event name. '
                err += f'Perhaps try self.get_no_event(\'{name}\') to access method?'
            else:
                err = 'Can not apply middleware to object of type functools.partial.'
            raise RuntimeError(err)
        if hasattr(wrapped, '_bound_func_with_event'):
            wrapped = wrapped._bound_func_with_event
            #  name = wrapped._bound_name
            #  err = f'Can not apply middleware as "{name}" is an event name. '
            #  err += f'Perhaps try self.get_no_event(\'{name}\') to access method?'
            #  raise RuntimeError(err)

        # If bound or staticmethod register events to __func__
        if hasattr(wrapped, '__func__'):
            obj = wrapped.__func__
        else:
            obj = wrapped
        if hasattr(obj, '_bd_middleware'):
            # Add to start for correct ordering
            middleware = middleware + obj._bd_middleware
        obj._bd_middleware = middleware
        return wrapped

    return wrapper

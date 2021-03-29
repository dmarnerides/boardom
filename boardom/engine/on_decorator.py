import boardom as bd
from functools import partial
import inspect


# TODO: Every is for the frequency
def on(*event_name, every=None):
    if len(event_name) < 1:
        raise ValueError('bd.on expected an event name')
    if not all(isinstance(x, str) for x in event_name):
        raise RuntimeError('Attempted to register invalid (non-string type) event.')
    event_name = list(event_name)

    def wrapper(wrapped):
        if inspect.isclass(wrapped):
            raise RuntimeError('bd.on can not be used on Classes.')
        if not (callable(wrapped) or isinstance(wrapped, staticmethod)):
            raise RuntimeError(
                'Ivalid use of bd.on. Expected callable or staticmethod.'
            )
        events = []
        if isinstance(wrapped, partial):
            f = wrapped.func
            if hasattr(f, '__func__') and (f.__func__ == bd.Engine.event):
                name = wrapped.args[0]
                err = f'Can not apply middleware as "{name}" is an event name. '
                err += f'Perhaps try self.get_no_event(\'{name}\') to access method?'
            else:
                err = 'Can not apply middleware to object of type functools.partial.'
            raise RuntimeError(err)
        # If bound or staticmethod register events to __func__
        should_register = False
        if hasattr(wrapped, '__func__'):
            obj = wrapped.__func__
            should_register = hasattr(wrapped, '__self__') and isinstance(
                wrapped.__self__, bd.Engine
            )
        else:
            obj = wrapped
        if hasattr(obj, '_bd_engine_events'):
            events += obj._bd_engine_events
        events = list(set(events + event_name))
        obj._bd_engine_events = events
        if should_register:
            wrapped.__self__.register(wrapped)
        return wrapped

    return wrapper

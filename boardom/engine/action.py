from functools import partial
import inspect
import boardom as bd
from .event import _check_valid_eventname


# TODO:
# > Handle *args and **kwargs in definitions, as well as passed arguments


class Action:
    __slots__ = [
        'event_name',
        'function',
        'sig_params',
        'argnames',
        'id',
        'bound_engine',
        'bound_self_name',
    ]

    def __init__(self, event_name, function):
        _check_valid_eventname(event_name)
        self.event_name = event_name
        self.function = function
        self.sig_params = inspect.signature(self.function).parameters
        self.argnames = list(self.sig_params)
        self.id = hash((event_name, function))
        bound_engine = None
        bound_self_name = None

        if hasattr(self.function, '__self__') and (
            isinstance(self.function.__self__, bd.Engine)
        ):
            bound_engine = self.function.__self__
            bound_self_name = list(
                inspect.signature(self.function.__func__).parameters
            )[0]

        self.bound_engine = bound_engine
        self.bound_self_name = bound_self_name

    def __eq__(self, other):
        return (
            isinstance(other, Action)
            and (self.event_name == other.event_name)
            and (self.function == other.function)
        )

    def __str__(self):
        return f'Action(on={self.event_name}, func={self.function})'

    # TODO: Optimize
    def __call__(self, calling_engine, event):
        args = event.args
        kwargs = event.kwargs
        argnames = self.argnames
        sig_params = self.sig_params
        have_bound_engine = self.bound_engine is not None
        middleware = []
        if hasattr(self.function, '_bd_middleware'):
            middleware += self.function._bd_middleware

        engine = None

        # If engine is provided as self argument take it to be self and remove the engine
        if have_bound_engine and (len(args) > 0) and isinstance(args[0], bd.Engine):
            engine, args = args[0], args[1:]

        # If engine provided in kwargs
        # or position of arg matches signature then use that one
        if 'engine' in kwargs:  # kwarg provided engine takes precedence
            engine = kwargs['engine']
        # Otherwise if "engine" is in signature, only replace the engine
        # if position matches or if given as kwarg
        elif 'engine' in self.argnames:
            eng_param_val = sig_params['engine'].default
            idx = self.argnames.index('engine')
            # First check if it was provided as an arg at the position of 'engine'
            if (idx < len(args)) and isinstance(args[idx], bd.Engine):
                engine = args[idx]
                # Remove from args since it will be skipped later
                args = args[:idx] + args[idx + 1 :]
            # Alternatively, replace it if it was a default kwarg
            elif eng_param_val is not inspect._empty:
                if isinstance(eng_param_val, bd.Engine):
                    engine = eng_param_val
                else:
                    raise TypeError(
                        f'{self.function.__name__} got object of type {type(eng_param_val)} '
                        'for engine argument. Expected bd.Engine.'
                    )

        # Default to the calling engine
        if engine is None:
            engine = calling_engine

        # If function is bound, and its __self__ is not the current engine then
        # we unbind and inject this engine (or the one given in args and kwargs) as self
        if have_bound_engine and (engine is not self.bound_engine):
            func = self.function.__func__
            args = (engine,) + args
            argnames = [self.bound_self_name] + argnames
        else:
            func = self.function

        final_kwargs = {}
        for arg in argnames:
            if arg == 'engine':
                final_kwargs['engine'] = engine
            else:
                if arg in kwargs:
                    final_kwargs[arg] = kwargs[arg]
                elif args:
                    final_kwargs[arg] = args[0]
                    args = args[1:]
        if args:
            raise RuntimeError('Provided extra args that could not be accounted for.')

        call = func
        # Once engine resolution is done, apply middleware and function
        for m in middleware:
            # middleware are f(engine, callback, **kwargs)
            # kwargs is a dict - not **kwargs
            call = partial(m, engine=engine, callback=call)

        return call(**final_kwargs)

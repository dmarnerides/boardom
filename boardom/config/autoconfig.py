import wrapt
import inspect
import boardom as bd


def _get_collection(val, key):
    val = val[key]
    if isinstance(val, str):
        return (val,)
    elif isinstance(val, (list, tuple)):
        return val
    else:
        raise ValueError(f'Invalid {key} parameter')


def autoconfig(_wrapped=None, ignore=None, alias=None):
    ignore = ignore or []

    # Allow for string ignore
    if isinstance(ignore, str):
        ignore = [ignore]

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        def _execute(*_args, **_kwargs):

            name = wrapped.__name__
            if inspect.isclass(wrapped):
                sig = inspect.signature(wrapped.__init__)
                params = sig.parameters
                # Remove self
                params = {k: v for i, (k, v) in enumerate(params.items()) if i > 0}
                params = type(params)(params)
            else:
                sig = inspect.signature(wrapped)
                params = sig.parameters
            new_args, new_kwargs = [], {}
            # Validate _kwargs
            for i, param_name in enumerate(params):
                param = params.get(param_name)
                # Also skip if *args or **kwargs
                if param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]:
                    continue
                if i < len(_args):
                    new_args.append(_args[i])
                else:
                    if param_name in _kwargs:
                        new_kwargs[param_name] = _kwargs[param_name]
                    else:
                        # Skip fetching from configuration
                        # if we ignore this one
                        if param_name in ignore:
                            continue
                        try:
                            if alias and param_name in alias:
                                param_name_in_config = alias[param_name]
                            else:
                                param_name_in_config = param_name
                            new_kwargs[param_name] = bd.cfg[param_name_in_config]
                        # Attribute error means param_name is not in options
                        except AttributeError:
                            def_val = param.default
                            if def_val is sig.empty:
                                raise AttributeError(
                                    f'Could not find configuration value for '
                                    f'\'{param_name}\' for '
                                    f'\'{name}{str(sig)}\''
                                )
                            new_kwargs[param_name] = def_val

            return wrapped(*new_args, **new_kwargs)

        return _execute(*args, **kwargs)

    if _wrapped is None:
        return wrapper
    else:
        return wrapper(_wrapped)

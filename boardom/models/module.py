import inspect
from torch import nn
import boardom as bd
from contextlib import contextmanager


class Module(nn.Module):
    _registry = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._module_key = cls.__name__.lower()
        Module._registry[cls._module_key] = cls


# Make all nn.Modules subclasses of Module
def _subclass_modules():
    if _subclass_modules.called:
        return
    for cls in nn.Module.__subclasses__():
        if cls is not Module:
            base_list = list(cls.__bases__)
            base_list[base_list.index(nn.Module)] = Module
            cls.__bases__ = tuple(base_list)
    if Module._registry is None:
        Module._registry = {
            key.lower(): val
            for key, val in nn.__dict__.items()
            if inspect.isclass(val) and issubclass(val, nn.Module)
        }


_subclass_modules.called = False
_subclass_modules()
_subclass_modules.called = True


def _try_build_bd_module(val):
    if isinstance(val, (list, tuple, Module)):
        try:
            module = magic_module(val)
        except RuntimeError as e:
            if '[model_cfg]' in str(e):
                return None
            else:
                raise e from None
        return module
    else:
        return None


def magic__setattr__(self, name, attr):
    maybe_module = _try_build_bd_module(attr)
    if maybe_module is not None:
        nn.Module.__setattr__(self, name, maybe_module)
    else:
        try:
            nn.Module.__setattr__(self, name, attr)
        except Exception as e:
            raise type(e)(
                f'\nMaybe {attr} is an invalid module configuration.'
            ) from None


def magic_add_module(self, name, module):
    maybe_module = _try_build_bd_module(module)
    if maybe_module is not None:
        nn.Module.add_module(self, name, maybe_module)
    else:
        try:
            nn.Module.add_module(self, name, module)
        except Exception as e:
            raise type(e)(
                f'\nMaybe {module} is an invalid module configuration.'
            ) from None


_DEFAULT_SETATTR = Module.__setattr__
_DEFAULT_ADD_MODULE = Module.add_module


@contextmanager
def magic_off():
    old_setattr = Module.__setattr__
    old_add_module = Module.add_module
    try:
        Module.__setattr__ = _DEFAULT_SETATTR
        Module.add_module = _DEFAULT_ADD_MODULE
        yield
    finally:
        Module.__setattr__ = old_setattr
        Module.add_module = old_add_module


@contextmanager
def magic_builder():
    old_setattr = Module.__setattr__
    old_add_module = Module.add_module
    try:
        Module.__setattr__ = magic__setattr__
        Module.add_module = magic_add_module
        yield
    finally:
        Module.__setattr__ = old_setattr
        Module.add_module = old_add_module


_ARGDICT_KEYS = ['kwargs', 'apply_fn', 'add_members']


def magic_module(cfg):
    with magic_builder():
        return _build_magic_module(cfg)


def _build_magic_module(cfg):
    # cfg can be Module, tuple, list
    if isinstance(cfg, nn.Module):
        return cfg

    if not isinstance(cfg, (tuple, list)):
        raise RuntimeError(
            '[model_cfg] '
            'Model config must be composed of lists, tuples, and Modules. '
            f'Config: {cfg} is of {type(cfg)}'
        )

    # cfg is list or tuple now

    # If empty, return Identity
    if not cfg:
        return nn.Identity()
    # If first element is 'list' or 'tuple', return corresponding
    if cfg[0] == 'list':
        return list(cfg[1:])
    if cfg[0] == 'tuple':
        return tuple(cfg[1:])

    # If the first element is a dict then:
    # 1. It is a sequential
    # 2. It must be the only element OR can have another
    #     element that must be a dict with the kwargs
    #     of bd.Sequential
    if isinstance(cfg[0], dict):
        if len(cfg) == 1:
            return bd.Sequential(cfg[0])
        elif len(cfg) == 2:
            if isinstance(cfg[1], dict):
                seq_argnames = inspect.getfullargspec(bd.Sequential).kwonlyargs
                if not all(x in seq_argnames for x in cfg[1]):
                    raise RuntimeError(f'[model_cfg] Invalid kwargs in config {cfg}')
                else:
                    return bd.Sequential(cfg[0], **cfg[1])
            else:
                raise RuntimeError(f'[model_cfg] Invalid module {cfg}')
        else:
            raise RuntimeError(f'[model_cfg] Invalid module {cfg}')

    # cfg is a list or tuple now and first element is NOT dict

    # If the first element is not allowed raise error
    if not isinstance(cfg[0], (str, list, tuple, Module)):
        raise RuntimeError(
            '[model_cfg] '
            'First elements of config lists/tuples must be modules, strings or '
            'list/tuples/dicts in the case of sequential modules. '
            f'Got type {type(cfg[0])} for {cfg[0]}.'
        )

    if isinstance(cfg[0], str):
        module_name = cfg[0].lower()
        rest = cfg[1:]
    else:
        module_name = 'sequential'
        rest = cfg

    if module_name not in Module._registry:
        raise RuntimeError(
            f'[model_cfg] Unsupported module {cfg[0]} found in configuration.'
        )

    module_cls = Module._registry[module_name]
    module_argnames = inspect.getfullargspec(module_cls).args[1:]

    # Build modules
    if not rest:
        # No arguments
        return module_cls()
    elif isinstance(rest[-1], dict) and all(key in module_argnames for key in rest[-1]):
        # Final dict keys are ALL module kwargs
        return module_cls(*rest[:-1], **rest[-1])
    elif isinstance(rest[-1], dict) and all(key in _ARGDICT_KEYS for key in rest[-1]):
        # Final dict is of the form:
        # {'kwargs': dict, 'apply_fn': callable, 'add_members': dict}
        argdict = rest[-1]
        if 'kwargs' in argdict:
            module = module_cls(*rest[:-1], **argdict['kwargs'])
        else:
            module = module_cls(*rest[:-1])
        if 'apply_fn' in argdict:
            module.apply(argdict['apply_fn'])
        if 'add_members' in argdict:
            for key, val in argdict['add_members'].items():
                setattr(module, key, val)
        return module
    else:
        # Pass all rest items as *args
        return module_cls(*rest)


_ERROR_STR = (
    '[model_cfg] '
    'Parameters for modules must be in one of the following forms: '
    '["name", args] or ["name", kwargs] or ["name", args, kwargs]. '
    'args can be of type list or tuple and kwargs must be of type dict. '
)

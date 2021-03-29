import torch
from collections.abc import Callable, Mapping
from functools import partial
import boardom as bd

# TODO: ALL THE EVENT NAMES DEFINED IN BOARDOM SOURCE CODE SHOULD BE IDENTIFIED AS SUCH
#       E.G. BD_GET_CHECKPOINT_SETTINGS


def _prepare_cfg(cfg, keys):
    if cfg is None:
        cfg = bd.cfg
    # Check we have the necessary settings in the config
    have = {key: key in cfg for key in keys}
    if not all(have.values()):
        missing = ', '.join([f'"{k}"' for k, v in have.items()])
        raise RuntimeError(f'Could not find configuration for settings: {missing}')
    return cfg


def _create_state_dict_element(self, name):
    if name not in self:
        self[name] = {}
    if not isinstance(self[name], bd.State):
        clsname = self.__class__.__name__
        raise TypeError(f'Expected {clsname}.{name} to be of type bd.State (dict).')


def _check_member_exists(self, name):
    if name not in self:
        raise KeyError(f'Could not find {name} in State.')


def _set_default_value(self, name, value):
    if name not in self:
        self[name] = value


def _train(x):
    if hasattr(x, 'train') and isinstance(x.train, Callable):
        x.train()
    return x


def _eval(x):
    if hasattr(x, 'eval') and isinstance(x.eval, Callable):
        x.eval()
    return x


def _iterate_state(state, prefix='', recurse=False):
    for key, value in state.items():
        yield f'{prefix}{key}', value
        if recurse and isinstance(value, bd.State):
            for m in _iterate_state(value, f'{prefix}{key}.', recurse):
                yield m


def _skip_states(func, item):
    if isinstance(item, (list, tuple)):
        return type(item)(_skip_states(func, x) for x in item)
    elif isinstance(item, bd.State):
        return item
    elif isinstance(item, Mapping):
        return type(item)({key: _recurse_apply(func, val) for key, val in item.items()})
    else:
        return func(item)


def _recurse_apply(func, item):
    if isinstance(item, (list, tuple)):
        return type(item)(_recurse_apply(func, x) for x in item)
    elif isinstance(item, Mapping):
        return type(item)({key: _recurse_apply(func, val) for key, val in item.items()})
    else:
        return func(item)


class StateUtils(bd.Engine):
    def named_members(self, recurse=False, start_from=None):
        state = self
        prefix = ''
        if start_from is not None:
            if not isinstance(start_from, str):
                raise TypeError('"start_from" argument must be a string.')
            state = self[start_from]
            prefix = start_from + '.'

        if not isinstance(state, (bd.Engine, bd.State)):
            raise TypeError('Can only get members of Engine or State types')

        for name, module in _iterate_state(state, prefix, recurse):
            yield name, module

    def members(self, recurse=False, start_from=None):
        for _, module in self.named_members(recurse=recurse, start_from=start_from):
            yield module

    def apply(self, func, recurse=True, start_from=None):
        if recurse:
            func = partial(_recurse_apply, func)
        else:
            func = partial(_skip_states, func)
        for key, val in self.named_members(False, start_from):
            self[key] = func(val)

    def train(self, recurse=True, start_from=None):
        self.apply(_train, recurse=recurse, start_from=start_from)

    def eval(self, recurse=True, start_from=None):
        self.apply(_eval, recurse=recurse, start_from=start_from)

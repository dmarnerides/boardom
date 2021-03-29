import torch
from torch import nn
from .module import Module, _try_build_bd_module


# Adapted from nn.Sequential
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
class BaseCollection(nn.Sequential, Module):
    # Dict_out can be None (output is not a collection), True (give dict output)
    # or False (give list output)
    # enum_tag is prepended to the keys when enumerated
    def __init__(self, *modules, dict_out=None, enum_tag='', min_elem=0):
        nn.Module.__init__(self)
        min_elem = max(min_elem, 0)
        self._dict_out = dict_out
        elem_dict = self._get_elem_dict(*modules, enum_tag=enum_tag)
        # Check size requirements
        if len(elem_dict) < min_elem:
            raise RuntimeError(
                f'{self.__class__.__name__} expected at least {min_elem} '
                f'modules, got {len(elem_dict)}.'
            )

        for key, module in elem_dict.items():
            self.add_module(key, module)

    def _get_elem_dict(self, *modules, enum_tag):
        elem_dict = {}
        if len(modules) == 1:
            sub_module = modules[0]
            # If single element, user might have passed a dict
            if isinstance(sub_module, dict):
                #  elem_dict = sub_module
                elem_dict = {
                    f'{enum_tag}{key}': module for key, module in sub_module.items()
                }
            elif isinstance(sub_module, (tuple, list)):
                # If the single element is a tuple/list, it might be a config
                maybe_module = _try_build_bd_module(sub_module)
                if maybe_module is None:
                    elem_dict = {
                        f'{enum_tag}{idx}': module
                        for idx, module in enumerate(sub_module)
                    }
                else:
                    elem_dict = {f'{enum_tag}0': maybe_module}
            elif isinstance(sub_module, Module):
                elem_dict = {f'{enum_tag}0': sub_module}
            else:
                raise RuntimeError()
        else:
            elem_dict = {
                f'{enum_tag}{idx}': module for idx, module in enumerate(modules)
            }
        return elem_dict

    def extra_repr(self):
        if self._dict_out is not None:
            return f'dict_out={self._dict_out}'
        else:
            return ''

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return super().__getattr__(key)
            except AttributeError as e:
                raise KeyError(str(e)) from e
        return super().__getitem__(key)


def _flatten_sequential(modules, keys=[]):
    if isinstance(modules, Sequential):
        modules = modules._modules
    if isinstance(modules, dict):
        for key, val in modules.items():
            keys = keys + [key]
            for sub_keys, sub_val in _flatten_sequential(val, keys):
                yield sub_keys, sub_val
            keys = list(keys[:-1])
    elif isinstance(modules, Module):
        yield keys, modules


# This is from Raymond Hettinger:
# http://code.activestate.com/recipes/252177-find-the-common-beginning-in-a-list-of-strings/#c14
def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m:
        return ''
    a, b = min(m), max(m)
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi) // 2 + 1
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return a[:hi]


class Sequential(BaseCollection):
    def __init__(self, *modules, enum_tag=''):
        super().__init__(*modules, enum_tag=enum_tag)
        if not self._modules:
            return
        flat_modules = [(keys, val) for keys, val in _flatten_sequential(self._modules)]
        current_keys = list(self._modules.keys())
        for key in current_keys:
            delattr(self, key)
        valid_keys = ['/'.join(k[0]) for k in flat_modules]
        prefix_len = len(commonprefix(valid_keys))
        if prefix_len == max(len(v) for v in valid_keys):
            prefix_len = 0
        for i, (_, val) in enumerate(flat_modules):
            valid_key = valid_keys[i][prefix_len:]
            setattr(self, valid_key, val)


###
# 1 to many
###

# Splits have multiple arms that run parallel and are fused
# e.g. for 3 arms
#    --A-->
# -- --B-->
#    --C-->
# Splits should be a list of module lists
class Split(BaseCollection):
    def __init__(self, *splits, dict_out=False):
        if not ((dict_out is True) or (dict_out is False)):
            raise ValueError('dict_out must be True or False. Got {dict_out}.')
        super().__init__(*splits, dict_out=dict_out, enum_tag='arm_', min_elem=2)

    def forward(self, x):
        if self._dict_out:
            return {key: module(x) for key, module in self._modules.items()}
        else:
            return [module(x) for module in self]


# Branches allow to skip parts of sequential computations
# e.g. for three arms the following will happen:
#          | ------------>
#          |      | ----->
# --> --A--| --B--| --C-->
# branches should be a list of module lists
class Branch(BaseCollection):
    def __init__(self, *branches):
        super().__init__(*branches, enum_tag='arm_', min_elem=2)

    def forward(self, x):
        if self._dict_out:
            outputs = {}
            for key, module in self._modules.items():
                x = module(x)
                outputs[key] = x
            return outputs
        else:
            outputs = []
            for module in self:
                x = module(x)
                outputs.append(x)
            return outputs


###
# Many to 1 (reductions)
###


class Sum(Module):
    def forward(self, outputs):
        if isinstance(outputs, dict):
            outputs = outputs.values()
        return sum(outputs)


class Cat(Module):
    def forward(self, outputs):
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        return torch.cat(outputs, 1)


class Prod(Module):
    def forward(self, outputs):
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        result = outputs[0]
        for x in outputs[1:]:
            result = result * x
        return result


# Used in the UNet by Ronneberger et al.
class CropCat2d(Module):
    def center_crop(self, x, target):
        _, _, th, tw = target.shape
        _, _, h, w = x.shape
        hstart, wstart = (h - th) // 2, (w - tw) // 2
        return x[:, :, hstart : hstart + th, wstart : wstart + tw]

    def forward(self, outputs):
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        cropped = [self.center_crop(x, outputs[-1]) for x in outputs[:-1]]
        cropped = cropped + outputs[-1:]
        return torch.cat(cropped, 1)


class SingleSelect(Module):
    def __init__(self, idx_or_key=-1):
        super().__init__()
        self.idx_or_key = idx_or_key

    def forward(self, outputs):
        return outputs[self.idx_or_key]


###
# Many to Many
###


class Parallel(BaseCollection):
    def __init__(self, *modules, dict_out=False):
        super().__init__(*modules, dict_out=dict_out, enum_tag='arm_')

    def forward(self, x):
        if self._dict_out:
            if not isinstance(x, dict):
                raise RuntimeError(
                    f'Parallel dict module received input of type {torch.typename(x)}'
                )
            my_keys = set(self.keys())
            x_keys = set(x.keys())
            if my_keys != x_keys:
                raise RuntimeError(
                    f'Could not match keys in Parallel module.'
                    f' Have {my_keys} but got {x_keys}'
                )
            return {key: module(x[key]) for key, module in self._modules.items()}
        else:
            if not isinstance(x, (list, tuple)):
                raise RuntimeError(
                    f'Parallel list/tuple module received input of type {torch.typename(x)}'
                )
            my_len, x_len = len(self), len(x)
            if my_len != x_len:
                raise RuntimeError(
                    f'Parallel module of size {my_len} received input of size {x_len}'
                )
            return [module(x[i]) for i, module in enumerate(self)]


# Applies single module to all input arms
class Map(BaseCollection):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if isinstance(x, dict):
            return {key: self.module(val) for key, val in x.items()}
        elif isinstance(x, (list, tuple)):
            return [self.module(val) for val in x]
        else:
            raise RuntimeError(
                f'Map module received input of type {torch.typename(x)}. '
                'Expected dict or list/tuple.'
            )


class MultiSelect(Module):
    def __init__(self, indices_or_keys=[0]):
        super().__init__()
        self.indices_or_keys = indices_or_keys

    def forward(self, outputs):
        if isinstance(outputs, dict):
            if isinstance(self.indices_or_keys[0], int):
                out_keys = list(outputs.keys())
                keys = [out_keys[i] for i in self.indices_or_keys]
            else:
                keys = self.indices_or_keys

            return {key: outputs[key] for key in keys}
        else:
            return [outputs[i] for i in self.indices_or_keys]

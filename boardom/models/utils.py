from itertools import chain
import torch
from torch import nn
import boardom as bd

# About pretraining and freezing:
# 1. We set bool flags: "bd_is_pretrained", "bd_is_frozen"
# 2. These are attached to the module / parameter.
#    > If they don't exist it means that they are False
#    > WARNING!! Does not mean that the initializer/optimizer/algorithm automatically ignores them
# 3. If applied to module, then all the children modules / parameters (recursively)
#    will also inherit the flag
# 4. "bd_is_pretrained":
#    > Signifies that the parameters were pretrained.
# 5. "bd_is_frozen":
#    > Signifies that the parameters should be frozen during training.

BD_PRETRAINED_ATTR = 'bd_is_pretrained'
BD_FROZEN_ATTR = 'bd_is_frozen'


def _check(x, attr):
    if isinstance(x, (nn.Module, nn.Parameter)):
        if hasattr(x, attr):
            return getattr(x, attr)
        else:
            return False
    else:
        raise RuntimeError(
            f'Invalid type: {torch.typename(x)}. Expected nn.Module or nn.Parameter.'
        )


def is_pretrained(module_or_param):
    return _check(module_or_param, BD_PRETRAINED_ATTR)


def is_frozen(module_or_param):
    return _check(module_or_param, BD_FROZEN_ATTR)


def is_trainable(module_or_param):
    return not (is_frozen(module_or_param) or is_pretrained(module_or_param))


def _set(x, value, attr):
    setattr(x, attr, value)
    if isinstance(x, nn.Module):
        for name, m in x.named_modules():
            if name:
                bd.log(f'Setting {name} to pretrained={value}.')
            else:
                bd.log('Setting module to pretrained={value}.')
            setattr(m, attr, value)
        for name, p in x.named_parameters():
            bd.log(f'Setting {name} requires_grad={value}.')
            setattr(p, attr, value)
            p.requires_grad_(value)
    else:
        x.requires_grad_(value)


def set_pretrained(module_or_param, value=True):
    if not isinstance(value, bool):
        raise RuntimeError(
            f'set_pretrained expected value to be bool, got: {type(value)}'
        )
    _set(module_or_param, value, BD_PRETRAINED_ATTR)


def set_frozen(module_or_param, value=True):
    if not isinstance(value, bool):
        raise RuntimeError(f'set_frozen expected value to be bool, got: {type(value)}')
    _set(module_or_param, value, BD_FROZEN_ATTR)


def _check_is_module(x, fname):
    if not isinstance(x, nn.Module):
        raise RuntimeError(f'{fname} expected and nn.Module but got: {type(x)}')


def pretrained_parameters(module):
    _check_is_module(module, 'pretrained_parameters')
    for p in module.parameters():
        if is_pretrained(p):
            yield p


def named_pretrained_parameters(module):
    _check_is_module(module, 'named_pretrained_parameters')
    for name, p in module.named_parameters():
        if is_pretrained(p):
            yield name, p


def pretrained_modules(module):
    _check_is_module(module, 'pretrained_modules')
    for m in module.modules():
        if is_pretrained(m):
            yield m


def named_pretrained_modules(module):
    _check_is_module(module, 'named_pretrained_modules')
    for name, m in module.named_modules():
        if is_pretrained(m):
            yield name, m


def frozen_parameters(module):
    _check_is_module(module, 'frozen_parameters')
    for p in module.parameters():
        if is_frozen(p):
            yield p


def named_frozen_parameters(module):
    _check_is_module(module, 'named_frozen_parameters')
    for name, p in module.named_parameters():
        if is_frozen(p):
            yield name, p


def frozen_modules(module):
    _check_is_module(module, 'frozen_modules')
    for m in module.modules():
        if is_frozen(m):
            yield m


def named_frozen_modules(module):
    _check_is_module(module, 'named_frozen_modules')
    for name, m in module.named_modules():
        if is_frozen(m):
            yield name, m


# Returns non pretrained, non frozen parameters
def trainable_parameters(module):
    _check_is_module(module, 'trainable_parameters')
    for p in module.parameters():
        if (not is_frozen(p)) and (not is_pretrained(p)):
            yield p


# Returns non pretrained, non frozen parameters
def named_trainable_parameters(module):
    _check_is_module(module, 'named_trainable_parameters')
    for name, p in module.named_parameters():
        if (not is_frozen(p)) and (not is_pretrained(p)):
            yield name, p


# Trainable is the ones that are not pretrained or frozen
def count_parameters(net, trainable=False):
    """Counts the parameters of a given PyTorch model."""
    params = bd.trainable_parameters(net) if trainable else net.parameters()
    return sum(p.numel() for p in params)


#  def disable_biases_before_bn(net, sample_input):
#      module_list = []
#
#      def hook(module, input, output):
#          if input is not output:
#              module_list.append(module)
#
#      handle = net.register_forward_hook(hook)
#      with torch.no_grad():
#          net(sample_input)
#      handle.release()
#      for m1, m2 in zip(module_list[:-1], module_list[1:]):
#          if m2.

import torch
from torch import nn
import boardom as bd

# About freezing:
# 1. We set bool flag "bd_is_frozen"
# 2. It is attached to the module / parameter.
#    > If it doesn't exist it means that it is False
#    > WARNING!! Does not mean that the initializer/optimizer/algorithm automatically ignores it
#       It just signifies that the parameters should be frozen during training.
# 3. If applied to module, then all the children modules / parameters (recursively)
#    will also inherit the flag


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


def is_frozen(module_or_param):
    return _check(module_or_param, 'is_frozen')


def is_trainable(module_or_param):
    return not is_frozen(module_or_param)


BN_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def freeze(module_or_param, value=True):
    if not isinstance(value, bool):
        raise RuntimeError(f'bd.freeze expected value to be bool, got: {type(value)}')

    if isinstance(module_or_param, nn.Module):
        for name, m in module_or_param.named_modules():
            if name:
                bd.log(f'Setting {name} to frozen={value}.')
            else:
                bd.log(f'Setting module to frozen={value}.')
            setattr(m, 'is_frozen', value)
            for name, p in m.named_parameters(recurse=False):
                setattr(p, 'is_frozen', value)
                bd.log(f'Setting {name} requires_grad={not value}.')
                p.requires_grad_(not value)
    else:
        bd.log(f'Setting {name} requires_grad={not value}.')
        module_or_param.requires_grad_(not value)

    return module_or_param


def _check_is_module(x, fname):
    if not isinstance(x, nn.Module):
        raise RuntimeError(f'{fname} expected and nn.Module but got: {type(x)}')


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


# Returns non frozen parameters
def trainable_parameters(module):
    _check_is_module(module, 'trainable_parameters')
    for p in module.parameters():
        if not is_frozen(p):
            yield p


# Returns non frozen parameters
def named_trainable_parameters(module):
    _check_is_module(module, 'named_trainable_parameters')
    for name, p in module.named_parameters():
        if not is_frozen(p):
            yield name, p


# Trainable is the ones that are not frozen
def count_parameters(net, trainable=False):
    """Counts the parameters of a given PyTorch model."""
    params = trainable_parameters(net) if trainable else net.parameters()
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

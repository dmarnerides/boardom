import boardom as bd
from torch import nn

NN_CONV_LAYERS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Unfold,
    nn.Fold,
)

NN_NORM_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)

NN_LINEAR_LAYERS = (nn.Identity, nn.Linear, nn.Bilinear)


def module_is_conv(module, mname, parameter, pname):
    return isinstance(module, NN_CONV_LAYERS)


def module_is_norm(module, mname, parameter, pname):
    return isinstance(module, NN_NORM_LAYERS)


def module_is_linear(module, mname, parameter, pname):
    return isinstance(module, NN_LINEAR_LAYERS)


def param_is_weight(module, mname, parameter, pname):
    # This is to account for weight_orig when using spectral norm
    return 'weight' in pname


def param_is_bias(module, mname, parameter, pname):
    return 'bias' in pname


def param_is_initialized(module, mname, parameter, pname):
    return hasattr(parameter, 'initialized') and parameter.initialized


def param_is_not_frozen(module, mname, parameter, pname):
    return (parameter is None) or (not bd.is_frozen(parameter))


def param_is_not_none(module, mname, parameter, pname):
    return parameter is not None

import torch
from torch import nn
from torch.nn import utils
from .init_filters import NN_CONV_LAYERS, NN_LINEAR_LAYERS


def _check_module(model):
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            f'apply_spectral_norm expected an nn.Module, got {torch.typename(model)}'
        )


_DEFAULT_MODULES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.Bilinear,
]


def spectral_norm(model, n_power_iterations=1, modules=None):
    _check_module(model)
    if modules is None:
        modules = _DEFAULT_MODULES
    for name, module in model.named_children():
        sn_module = spectral_norm(module, modules=modules)
        setattr(model, name, sn_module)

    if isinstance(model, tuple(modules)):
        try:
            model = utils.spectral_norm(model, n_power_iterations=n_power_iterations)
        except RuntimeError as e:
            if 'Cannot register' not in str(e):
                raise

    return model


def remove_spectral_norm(model, modules=None):
    _check_module(model)
    for name, module in model.named_children():
        sn_module = remove_spectral_norm(module, modules=modules)
        setattr(model, name, sn_module)
    try:
        model = utils.spectral_norm(model)
    except ValueError as e:
        if 'spectral_norm of' not in str(e):
            raise
    return model

import types
from torch import nn
import torch.nn.functional as F
import boardom as bd


def _has_frozen_stats(m):
    return hasattr(m, 'frozen_stats') and m.frozen_stats


class BatchNormFreezer:
    @staticmethod
    def assign(module, method_name):
        old_method = getattr(module, method_name)
        new_method = types.MethodType(getattr(BatchNormFreezer, method_name), module)
        setattr(module, method_name, new_method)
        setattr(module, f'_old_{method_name}', old_method)

    @staticmethod
    def unassign(module, method_name):
        original_method = getattr(module, f'_old_{method_name}')
        setattr(module, method_name, original_method)
        delattr(module, f'_old_{method_name}')

    @staticmethod
    def freeze_stats(module, name=None):
        if not _has_frozen_stats(module):
            altname = '.' if name is None else f': {name}'
            bd.log(f'Freezing batchnorm running stats{altname}')
            BatchNormFreezer.assign(module, 'forward')
            BatchNormFreezer.assign(module, 'extra_repr')
            setattr(module, 'frozen_stats', True)

    @staticmethod
    def unfreeze_stats(module, name=None):
        if _has_frozen_stats(module):
            name = '.' if name is None else f': {name}'
            bd.log(f'Unfreezing batchnorm module{name}')
            BatchNormFreezer.unassign(module, 'forward')
            BatchNormFreezer.unassign(module, 'extra_repr')
            setattr(module, 'frozen_stats', False)

    def forward(self, input):
        self._check_input_dim(input)
        training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=training,
            momentum=0.0,
            eps=self.eps,
        )

    def extra_repr(self):
        return self._old_extra_repr() + ' --Frozen Running Stats-- '


def freeze_bn_running_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            BatchNormFreezer.freeze_stats(module, name)
    return model


def unfreeze_bn_running_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            BatchNormFreezer.unfreeze_stats(module, name)
    return model

import types
from torch import nn
import torch.nn.functional as F
import boardom as bd


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
    def freeze(module, name):
        if not bd.is_frozen(module):
            bd.log(f'Freezing batchnorm module: {name}')
            BatchNormFreezer.assign(module, 'forward')
            BatchNormFreezer.assign(module, 'extra_repr')
            bd.set_frozen(module)

    @staticmethod
    def unfreeze(module, name):
        if bd.is_frozen(module):
            bd.log(f'Unfreezing batchnorm module: {name}')
            BatchNormFreezer.unassign(module, 'forward')
            BatchNormFreezer.unassign(module, 'extra_repr')
            bd.set_frozen(module, False)

    def forward(self, input):
        self._check_input_dim(input)
        training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            # training False means
            training=training,
            momentum=0.0,
            eps=self.eps,
        )

    def extra_repr(self):
        return self._old_extra_repr() + ' --Frozen-- '


def freeze_batchnorm(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            BatchNormFreezer.freeze(module, name)
    return model


def unfreeze_batchnorm(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            BatchNormFreezer.unfreeze(module, name)
    return model

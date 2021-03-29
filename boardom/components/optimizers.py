from collections.abc import Mapping
import torch
from torch import optim
import boardom as bd
from .util import _create_state_dict_element, _prepare_cfg
from ..config.common import OPTIMIZER_KEYS

# ELEMENTS:
#     self.optimizers
# EVENTS ISSUED:
#     None
# EVENTS LISTENED:
#     "get_checkpoint_settings" -> setup_optimizer_checkpoints
# ATTACH FUNCTIONS:
#     attach_optimizers
class Optimizers(bd.Engine):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'optimizers')

    @bd.on('get_checkpoint_settings')
    def setup_optimizer_checkpoints(self):
        if not self.optimizers:
            bd.warn(
                'Attempted to setup optimizer checkpoints before attaching optimizers.'
            )
            return None
        else:
            return [dict(state_key=f'optimizers.{k}') for k in self.optimizers]

    def attach_optimizers(self, force=False):
        if force or (not self.optimizers):
            if not hasattr(self, 'setup_optimizers'):
                bd.warn(
                    'Could not find setup function for optimizers. Will not attach to engine.'
                )
                return
            optims = self.setup_optimizers()
            if isinstance(optims, Mapping):
                self.optimizers = optims
            else:
                self.optimizers.main = optims
                if not isinstance(optims, optim.Optimizer):
                    bd.warn('Optimizer is not of type torch.optim.Optimizer')

    def create_optimizer_from_cfg(self, parameters, cfg=None):
        cfg = _prepare_cfg(cfg, OPTIMIZER_KEYS)
        optimizer = cfg.optimizer.lower()
        if optimizer == 'adam':
            ret = optim.Adam(
                parameters,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'adamw':
            ret = torch.optim.AdamW(
                parameters,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'sgd':
            ret = torch.optim.SGD(
                parameters,
                lr=cfg.lr,
                momentum=cfg.momentum,
                dampening=cfg.dampening,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'adadelta':
            ret = torch.optim.Adadelta(
                parameters,
                lr=cfg.lr,
                rho=cfg.rho,
                eps=cfg.optim_eps,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'adagrad':
            ret = torch.optim.Adagrad(
                parameters,
                lr=cfg.lr,
                lr_decay=cfg.lr_decay,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'sparseadam':
            ret = torch.optim.SparseAdam(
                parameters, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.optim_eps
            )
        elif optimizer == 'adamax':
            ret = torch.optim.Adamax(
                parameters,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.optim_eps,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer == 'rmsprop':
            ret = torch.optim.RMSprop(
                parameters,
                lr=cfg.lr,
                alpha=cfg.alpha,
                eps=cfg.optim_eps,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
                centered=cfg.centered,
            )
        else:
            raise NotImplementedError(f'Optimizer {optimizer} not implemented.')
        return ret

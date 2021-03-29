from collections.abc import Mapping
import boardom as bd
from .util import _create_state_dict_element, _prepare_cfg
from ..config.common import CRITERIA_KEYS


# ELEMENTS:
#     self.criteria
# EVENTS ISSUED:
#     None
# EVENTS LISTENED:
#     "get_checkpoint_settings" -> setup_criterion_checkpoints
# ATTACH FUNCTIONS:
#     attach_criteria
class Criteria(bd.Engine):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'criteria')

    @bd.on('get_checkpoint_settings')
    def setup_criterion_checkpoints(self):
        if not self.criteria:
            bd.warn(
                'Attempted to setup criterion checkpoints before attaching optimizers.'
            )
            return None
        else:
            return [dict(state_key=f'criteria.{k}') for k in self.criteria]

    def attach_criteria(self, force=False):
        if force or (not self.criteria):
            if not hasattr(self, 'setup_criteria'):
                bd.warn(
                    'Could not find setup function for criteria. Will not attach to engine.'
                )
                return
            crits = self.setup_criteria()
            if isinstance(crits, Mapping):
                self.criteria = crits
            else:
                self.criteria.main = crits

    def create_criteria_from_cfg(self, cfg=None):
        cfg = _prepare_cfg(cfg, CRITERIA_KEYS)
        if not cfg.criteria:
            return
        bd.print_separator()
        bd.log('Building criteria from cfg')
        ret = bd.State({'weights': {}})
        w_strs, mod_strs = [], []
        for name in cfg.criteria:
            module = bd.magic_module([name])
            with cfg.group_fallback():
                weight = cfg.g[name].get('criterion_weight')
            mod_strs.append(f'\t{module}')
            w_strs.append(f'\t{name}={weight}')
            ret[name] = module
            ret.weights[name] = weight
        bd.write('Criteria:\n' + '\n'.join(mod_strs))
        bd.write('Weights:\n' + '\n'.join(w_strs))
        bd.print_separator()
        return ret

    def default_compute_losses(self, prediction, target):
        losses = bd.State()
        total = 0
        crits = self.criteria
        weights = crits.weights
        for key, loss_fn in crits.items():
            if key == 'weights':
                continue
            current = weights[key] * loss_fn(prediction, target)
            losses[key] = current
            total = total + current
        losses.total = total
        return losses

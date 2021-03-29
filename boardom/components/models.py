from collections.abc import Mapping
from functools import partial
import torch
import boardom as bd
from .util import (
    StateUtils,
    _create_state_dict_element,
    _check_member_exists,
)
from ..config.common import DEVICE_KEYS

# ELEMENTS:
#     self.models
# EVENTS ISSUED:
#     None
# EVENTS LISTENED:
#     "get_checkpoint_settings" -> setup_model_checkpoints
# ATTACH FUNCTIONS:
#     attach_models
class Models(bd.Engine):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'models')

    @bd.on('get_checkpoint_settings')
    def setup_model_checkpoints(self):
        if not self.models:
            bd.warn('Attempted to setup model checkpoints before attaching models.')
            return None
        else:
            return [dict(state_key=f'models.{k}') for k in self.models]

    def attach_models(self, force=False):
        if force or (not self.models):
            if not hasattr(self, 'setup_models'):
                bd.warn(
                    'Could not find setup function for models. Will not attach to engine.'
                )
                return
            models = self.setup_models()
            if isinstance(models, Mapping):
                self.models = models
            else:
                self.models.main = models

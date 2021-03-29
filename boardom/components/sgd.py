import torch
import boardom as bd
from .util import _set_default_value, _create_state_dict_element
from .data import Data
from .optimizers import Optimizers
from .models import Models
from .criteria import Criteria


# TODO: Restarting iterator from iteration.
# TODO: Disable gradients for validation
# TODO: torch.set_grad_enabled(False)
# TODO: Do .train() after validation ends

# ELEMENTS:
#     self.training
# EVENTS ISSUED:
#     fit() -> "training_start"
#     fit() -> "training_end"
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     None


class SGDTrainer(Data, Optimizers, Criteria, Models):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'training')
        # We are in epoch 1 even if training hasn't started yet
        # (convention that makes my life easier)
        _set_default_value(self.training, 'epoch', 1)
        _set_default_value(self.training, 'max_epochs', None)
        _set_default_value(self.training, 'global_step', 0)
        _set_default_value(self.training, 'epoch_step', 0)
        _set_default_value(self.training, 'should_stop_training', False)

    # Requires Data engine with iterate_data generator API
    def fit(self, max_epochs=None):
        self.event('training_start')
        if ('max_epochs' not in self.training) or (max_epochs is not None):
            self.training.max_epochs = max_epochs

        if self.training.max_epochs is None:
            bd.warn('max_epochs not set, training will continue indefinitely')

        while True:
            self.training.epoch_step = 0
            for _ in self.iterate_data('training'):
                self.training.global_step += 1
                self.training.epoch_step += 1
                self.train()
                self.training_iteration()
                if self.training.should_stop_training:
                    break
            # Get max epochs again in case it changed:
            me = self.training.max_epochs
            if self.training.should_stop_training or (
                (me is not None) and (self.training.epoch == me)
            ):
                break
            self.training.epoch += 1
        self.event('training_end')


# ELEMENTS:
#     self.training.n_accumulate_gradients
#     self.training.losses
#     self.training.predictions
#     self.training.metrics
# EVENTS ISSUED:
#     training_iteration() -> "forward"
#     training_iteration() -> "backward"
#     fit() -> "training_end"
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     None
class DefaultSGDTrainer(SGDTrainer):
    def __init__(self):
        super().__init__()
        _set_default_value(self.training, 'n_accumulate_gradients', 1)

    def accumulate_gradients(self, num_iter=1):
        if (not isinstance(num_iter, int)) or num_iter < 1:
            raise RuntimeError(
                f'Expected num_iter to be a positive integer. Got: {num_iter}'
            )

        self.training.n_accumulate_gradients = num_iter

    def training_iteration(self):
        self.do_forward()
        self.event('did_forward')
        training = self.training
        step = training.global_step
        if (step % training.n_accumulate_gradients) == 0:
            self.optimizers.main.zero_grad()
        self.do_backward()
        self.event('did_backward')
        if (step % training.n_accumulate_gradients) == 0:
            self.optimizers.main.step()
            self.event('optimizer_step')

    def do_forward(self):
        pred = self.models.main(self.datum.training[0])
        self.training.predictions = pred
        self.training.losses = self.criteria.main(pred, self.datum.training[1])
        self.training.metrics = None

    def do_backward(self):
        self.training.losses.backward()


# ELEMENTS:
#     self.validation
# EVENTS ISSUED:
#     validate() -> "validation_start"
#     validate() -> "validation_end"
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     None
class SGDValidator(Data):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'validation')
        _set_default_value(self.validation, 'global_step', 0)
        _set_default_value(self.validation, 'epoch_step', 0)

    def validate(self):
        self.validation.epoch_step = 0
        self.event('validation_start')
        for _ in self.iterate_data('validation'):
            self.validation.global_step += 1
            self.validation.epoch_step += 1
            self.eval()
            with torch.no_grad():
                self.validation_iteration()
        self.event('validation_end')

    def validation_iteration(self):
        raise NotImplementedError


# ELEMENTS:
#     self.validation.predictions
#     self.validation.losses
#     self.validation.metrics
# EVENTS ISSUED:
#     None
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     None
class DefaultSGDValidator(SGDValidator):
    def validation_iteration(self):
        pred = self.models.main(self.datum.validation[0])
        self.validation.prediction = pred
        self.validation.loss = self.criteria.main(pred, self.datum.validation[1])
        self.validation.metrics = None

import boardom as bd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from .device import Device
from .util import (
    StateUtils,
    _create_state_dict_element,
    _check_member_exists,
    _prepare_cfg,
)
from ..config.common import DATALOADER_KEYS

# TODO: Implement TORCHVISION CHECKPOINTS


# ELEMENTS:
#     self.data =  {mode: ...}
#     self.datum = {mode: ...}
# EVENTS ISSUED:
#     (<mode> in ['training', 'validation', 'testing'])
#     iterate_data(mode) -> "<mode>_epoch_setup"
#     iterate_data(mode) -> "<mode>_epoch_start"
#     iterate_data(mode) -> "<mode>_epoch_end"
#     iterate_data(mode) -> "<mode>_epoch_cleanup"
#     iterate_data(mode) -> "<mode>_iteration_start"
#     iterate_data(mode) -> "<mode>_iteration_end"
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     attach_trainining_data
#     attach_validation_data
#     attach_testing_data
class Data(Device, StateUtils):
    def __init__(self):
        super().__init__()
        # Manage self.data and self.datum dictionaries
        _create_state_dict_element(self, 'data')
        _create_state_dict_element(self, 'datum')

    def iterate_data_no_events(self, mode):
        _check_member_exists(self, f'data.{mode}')
        for self.datum[mode] in self.data[mode]:
            with bd.CleanupState(self):
                self.to(self.devices.default, 'datum')
                yield self.datum[mode]
            del self.datum[mode]

    def iterate_data(self, mode):
        _check_member_exists(self, f'data.{mode}')
        self.event(f'{mode}_epoch_start')
        for self.datum[mode] in self.data[mode]:
            with bd.CleanupState(self):
                self.to(self.devices.default, 'datum')
                self.event(f'{mode}_iteration_setup')
                self.event(f'{mode}_iteration_start')
                yield self.datum[mode]
                self.event(f'{mode}_iteration_end')
                self.event(f'{mode}_iteration_cleanup')
            del self.datum[mode]
        self.event(f'{mode}_epoch_end')

    def attach_data(self, mode, force=False):
        if force or (f'data.{mode}' not in self):
            bd.log(f'Attaching {mode} dataset.')
            fn_name = f'setup_{mode}_data'
            if not hasattr(self, fn_name):
                bd.warn(
                    f'Could not find setup function for {mode} dataset. Will not attach to engine.'
                )
                return
            data_fn = getattr(self, fn_name)
            self.data[mode] = data_fn()

    def attach_training_data(self, force=False):
        self.attach_data('training', force=False)

    def attach_validation_data(self, force=False):
        self.attach_data('validation', force=False)

    def attach_testing_data(self, force=False):
        self.attach_data('testing', force=False)

    def create_dataloader_from_cfg(
        self, dataset, cfg=None, worker_init_fn=None, collate_fn=None, postprocess=None
    ):
        cfg = _prepare_cfg(cfg, DATALOADER_KEYS)
        kwargs = {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers,
            'pin_memory': cfg.pin_memory,
            'shuffle': cfg.shuffle,
            'drop_last': cfg.drop_last,
            'worker_init_fn': worker_init_fn,
            'timeout': cfg.timeout,
        }
        collate_fn = collate_fn or default_collate
        if postprocess is not None:

            def collate_new(*args, **kwargs):
                return postprocess(collate_fn(*args, **kwargs))

        else:
            collate_new = collate_fn

        kwargs['collate_fn'] = collate_new
        return DataLoader(dataset, **kwargs)

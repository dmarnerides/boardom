from functools import partial
import torch
import boardom as bd
from .util import (
    StateUtils,
    _prepare_cfg,
    _create_state_dict_element,
)
from ..config.common import DEVICE_KEYS


def _to(x, device):
    if hasattr(x, 'to'):
        return x.to(device)
    elif isinstance(x, torch.optim.Optimizer):
        optim_sd = bd.apply(partial(_to, device=device))(x.state_dict())
        x.load_state_dict(optim_sd)
        return x
    else:
        return x


# ELEMENTS:
#     self.devices
# EVENTS ISSUED:
#     None
# EVENTS LISTENED:
#     None
# ATTACH FUNCTIONS:
#     N/A
class Device(StateUtils):
    def __init__(self):
        super().__init__()
        # Manage self.devices
        _create_state_dict_element(self, 'devices')
        if 'devices.default' not in self:
            self.devices.default = torch.device('cpu')

    def set_device(self, device, key='default'):
        try:
            torch.device(key)
            raise ValueError('Device key must not be a valid torch.device string')
        except RuntimeError:
            pass
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.devices[key] = device
        return self

    def setup_devices_from_cfg(self, cfg=None):
        cfg = _prepare_cfg(cfg, DEVICE_KEYS)
        self.set_device(cfg.device)
        device = self.devices.default
        if device.type == 'cuda':
            bd.log(f'Setting default cuda device: {device}')
            torch.cuda.set_device(device)
            bd.log(f'Setting cudnn_benchmark={cfg.cudnn_benchmark}')
            torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    def to(self, device=None, recurse=True, start_from=None):
        if device is None:
            try:
                device = self.devices.default
            except AttributeError:
                raise RuntimeError(
                    'Device not provided and default device not set'
                ) from None
        elif isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e:
                # Handle invalid device
                if 'Expected one of' in str(e):
                    try:
                        device = self.devices[device]
                    except (AttributeError, KeyError):
                        raise RuntimeError(f'Invalid device: {device}') from None
                else:
                    raise e

        to = partial(_to, device=device)
        return self.apply(to, recurse=recurse, start_from=start_from)

import torch.nn.functional as F
from .module import Module

# Keeping align corners
class Resize(Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        kwargs = {'scale_factor': scale_factor, 'mode': mode}
        if mode == 'bilinear':
            kwargs['align_corners'] = align_corners
        self.kwargs = kwargs

    def forward(self, t_in):
        return F.interpolate(t_in, **self.kwargs)

    def extra_repr(self):
        return f'Factor: {self.scale_factor}, Mode: {self.mode}'

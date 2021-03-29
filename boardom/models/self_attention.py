import torch
from torch import nn
import torch.nn.functional as F
from .module import Module


class SelfAttention(Module):
    def __init__(self, n_in, factor=8, gamma_init=0, scaling=1):
        super(SelfAttention, self).__init__()
        n_low_res = n_in // factor
        self.f = nn.Conv2d(n_in, n_low_res, 1)
        self.g = nn.Conv2d(n_in, n_low_res, 1)
        self.h = nn.Conv2d(n_in, n_in, 1)
        self.gamma = nn.Parameter(torch.Tensor([gamma_init]))
        self.scaling = scaling
        self.gamma_init = gamma_init
        self.factor = factor

    def forward(self, t_in):
        b, c, h_hr, w_hr = t_in.shape
        t_in_hr = t_in
        if self.scaling > 1:
            t_in = F.interpolate(
                t_in,
                scale_factor=1 / self.scaling,
                mode='bilinear',
                align_corners=False,
            )
            _, _, h, w = t_in.shape
        else:
            h, w = h_hr, w_hr

        f_val = self.f(t_in).view(b, -1, w * h)
        g_val = self.g(t_in).view(b, -1, w * h)
        h_val = self.h(t_in).view(b, -1, w * h)

        attention_map = torch.bmm(f_val.transpose(1, 2), g_val)
        attention_map = torch.softmax(attention_map, dim=-2)

        ret = torch.bmm(h_val, attention_map).view(b, c, h, w)
        if self.scaling > 1:
            ret = F.interpolate(
                ret, size=(h_hr, w_hr), mode='bilinear', align_corners=False
            )
        return self.gamma * ret + t_in_hr

    def extra_repr(self):
        return f'Factor: {self.factor}, Scale: {self.scaling}, gamma_init: {self.gamma_init}'

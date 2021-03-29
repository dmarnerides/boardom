import numpy as np
from torch import nn
import torch.nn.functional as F
from .module import Module
from .padding import padsame, pad2dsame, Pad2dSame


def conv(x, *args, **kwargs):
    if x.ndim == 3:
        return F.conv1d(x, *args, **kwargs)
    elif x.ndim == 4:
        return F.conv2d(x, *args, **kwargs)
    elif x.ndim == 5:
        return F.conv3d(x, *args, **kwargs)
    else:
        raise RuntimeError(f'Invalid input dimensions ({x.dim}) for input')


def convsame(
    x, weight, bias=None, stride=1, dilation=1, pad_mode='reflect', pad_value=0,
):
    padded = padsame(
        x,
        kernel_size=weight.shape[2:],
        dilation=dilation,
        mode=pad_mode,
        value=pad_value,
    )
    return conv(padded, weight, bias=bias, stride=stride, dilation=dilation)


def conv2dsame(
    x, weight, bias=None, stride=1, dilation=1, pad_mode='reflect', pad_value=0,
):
    padded = pad2dsame(
        x,
        kernel_size=weight.shape[-2:],
        dilation=dilation,
        mode=pad_mode,
        value=pad_value,
    )
    return F.conv2d(padded, weight, bias=bias, stride=stride, dilation=dilation)


# Keep dimensions same (or multiple of stride)
class Conv2dSame(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        pad_mode='reflect',
        pad_value=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        self.padder = Pad2dSame(kernel_size, dilation, pad_mode, pad_value)

    def forward(self, x):
        return super().forward(self.padder(x))


# Keep dimensions same (or multiple of stride)
class ConvTranspose2dSame(nn.ConvTranspose2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        out_padding = 0
        if ((kernel_size % 2 == 1) and (stride % 2 == 0)) or (
            (kernel_size % 2 == 0) and (stride % 2 == 1)
        ):
            out_padding = 1
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            output_padding=out_padding,
            bias=bias,
        )

        total_pad = kernel_size - stride + out_padding
        p = total_pad // 2
        q = total_pad - p
        padding = (p, q, p, q)
        padding = [None if x == 0 else x for x in padding]
        self.slc = np.s_[
            :,
            :,
            padding[0] : -padding[1] if padding[1] is not None else None,
            padding[2] : -padding[3] if padding[3] is not None else None,
        ]

    def forward(self, x):
        return super().forward(x)[self.slc]

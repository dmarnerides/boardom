# Adapted from: https://github.com/pytorch/pytorch/issues/21462
#
#  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
#  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
#  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
#  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
#  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
#  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
#  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
#  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
#  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
#  From Caffe2:
#
#  Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
#  All contributions by Facebook:
#  Copyright (c) 2016 Facebook Inc.
#
#  All contributions by Google:
#  Copyright (c) 2015 Google Inc.
#  All rights reserved.
#
#  All contributions by Yangqing Jia:
#  Copyright (c) 2015 Yangqing Jia
#  All rights reserved.
#
#  All contributions by Kakao Brain:
#  Copyright 2019-2020 Kakao Brain
#
#  All contributions from Caffe:
#  Copyright(c) 2013, 2014, 2015, the respective contributors
#  All rights reserved.
#
#  All other contributions:
#  Copyright(c) 2015, 2016 the respective contributors
#  All rights reserved.
#
#  Caffe2 uses a copyright model similar to Caffe: each contributor holds
#  copyright over their contributions to Caffe2. The project versioning records
#  all such contribution and copyright details. If a contributor wants to further
#  mark their specific copyright on a particular contribution, they should
#  indicate their copyright solely in the commit message of the change when it is
#  committed.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#     and IDIAP Research Institute nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

import torch
from functools import partial
from torch import nn
from torch.nn import functional as F
import numpy as np
import boardom as bd


def compl_mul(a, b):
    """
    Given a and b two tensors of dimension 4
    with the last dimension being the real and imaginary part,
    returns a multiplied by the conjugate of b, the multiplication
    being with respect to the second dimension.
    """
    op = partial(torch.einsum, "bchw,dchw->bdhw")
    return torch.stack(
        [
            op(a[..., 0], b[..., 0]) + op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) - op(a[..., 0], b[..., 1]),
        ],
        dim=-1,
    )


def fft_conv2d(x, weight, bias=None):
    b, c, h, w = x.shape
    n_out, n_in, k, k = weight.shape
    total_pad_w, total_pad_h = k - 1, k - 1
    pad_lw, pad_lh = total_pad_w // 2, total_pad_h // 2
    pad_rw, pad_rh = total_pad_w - pad_lw, total_pad_h - pad_lh
    x = F.pad(x, (pad_lw, pad_rw, pad_lh, pad_rh))
    b, c, h, w = x.shape  # Get it again in case we padded

    fft_pad_lw, fft_pad_lh = (w - 1) // 2, (h - 1) // 2
    fft_pad_rw, fft_pad_rh = w - 1 - fft_pad_lw, h - 1 - fft_pad_lh
    weight_pad_lw, weight_pad_lh = (w - k) // 2, (h - k) // 2
    weight_pad_rw, weight_pad_rh = w - k - weight_pad_lw, h - k - weight_pad_lh
    weight_padded = F.pad(
        weight,
        (
            weight_pad_lw + fft_pad_lw,
            weight_pad_rw + fft_pad_rw,
            weight_pad_lh + fft_pad_lh,
            weight_pad_rh + fft_pad_rh,
        ),
    )
    x = F.pad(x, (fft_pad_lw, fft_pad_rw, fft_pad_lh, fft_pad_rh))
    x_fft = torch.rfft(x, 2)
    weight_fft = torch.rfft(weight_padded, 2)
    result_fft = compl_mul(x_fft, weight_fft)
    result = torch.irfft(result_fft, 2, signal_sizes=(x.shape[-2], x.shape[-1]))
    b, c, h, w = result.shape  # Get it again in case we unpadded

    if bias is not None:
        result += bias
    res = result.clone()
    res[:, :, : int(np.floor(h / 2)), :] = result[:, :, int(np.ceil(h / 2)) :, :]
    res[:, :, int(np.floor(h / 2)) :, :] = result[:, :, : int(np.ceil(h / 2)), :]
    result = res.clone()
    res[:, :, :, : int(np.floor(w / 2))] = result[:, :, :, int(np.ceil(w / 2)) :]
    res[:, :, :, int(np.floor(w / 2)) :] = result[:, :, :, : int(np.ceil(w / 2))]
    res = res[
        :,
        :,
        pad_lh + fft_pad_lh : h - pad_rh - fft_pad_rh,
        pad_lw + fft_pad_lw : w - pad_rw - fft_pad_rw,
    ].contiguous()
    return res


class FFTConv2d(bd.Module):
    """
    Convoluton based on FFT, faster for large kernels and small strides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1))
        else:
            self.bias = None
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, signal):
        padded = F.pad(self.weight, (0, signal.size(-1) - self.weight.size(-1)))
        signal_fr = torch.rfft(signal, 1)
        weight_fr = torch.rfft(padded, 1)
        output_fr = compl_mul(signal_fr, weight_fr)
        output = torch.irfft(output_fr, 1, signal_sizes=(signal.size(-1),))
        output = output[..., :: self.stride]
        target_length = (signal.size(-1) - self.kernel_size) // self.stride + 1
        output = output[..., :target_length].contiguous()
        if self.bias is not None:
            output += self.bias
        return output

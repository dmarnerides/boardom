import torch.nn.functional as F
from .module import Module

# Convention: For same size convolutions with even kernel sizes,
#             padding right is padding left+1
# Convention: For images, and kernels, x is width, y is height,
# Convention: Tuples are (w,h), i.e. reversed in many imaging libraries
#             (e.g. in cv2.resize)
#             BUT: in torch, they are (h,w)! (ARE THEY? F.pad is (wleft, wright, hleft, hright))
#    Here we keep the torch convention of height, width in tuples (DO WE??)
# Convention: In torch, conv2d weight is out_channels x in_channels x k_h * k_w

# Modes: 'constant', 'reflect', 'replicate' or 'circular'
class PadSame(Module):
    def __init__(self, kernel_size, dilation=1, mode='constant', value=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mode = mode
        self.value = value
        self.kernel_size = kernel_size

    def forward(self, x):
        return padsame(x, self.kernel_size, self.dilation, self.mode, self.value)

    def extra_repr(self):
        ret = f'kernel_size={self.kernel_size}, dilation="{self.dilation}", mode="{self.mode}"'
        if self.mode == 'constant':
            ret += f', value={self.value}'
        return ret


# Modes: 'constant', 'reflect', 'replicate' or 'circular'
class Pad2dSame(Module):
    def __init__(self, kernel_size, dilation=1, mode='constant', value=0):
        super().__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.mode = mode
        self.value = value
        total_pad = [d * (k - 1) for d, k in zip(dilation, kernel_size)]
        self.left = [tp // 2 for tp in total_pad]
        self.right = [tp - l for tp, l in zip(total_pad, self.left)]
        self.padding = (self.left[0], self.right[0], self.left[1], self.right[1])

    def forward(self, x):
        return F.pad(x, self.padding, mode=self.mode, value=self.value)

    def extra_repr(self):
        ret = f'padding={self.padding}, mode="{self.mode}"'
        if self.mode == 'constant':
            ret += f', value={self.value}'
        return ret


def padsame(x, kernel_size, dilation=1, mode='constant', value=0):
    ndim = x.ndim - 2
    if ndim not in [1, 2, 3]:
        raise RuntimeError(
            f'Expected padding input to be 3,4 or 5 dimensional, but got {x.ndim}.'
        )
    if isinstance(dilation, int):
        dilation = (dilation,) * ndim
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndim
    total_pad = [d * (k - 1) for d, k in zip(dilation, kernel_size)]
    left = [tp // 2 for tp in total_pad]
    right = [tp - l for tp, l in zip(total_pad, left)]
    padding = [side[i] for i in range(ndim) for side in [left, right]]
    return F.pad(x, padding, mode=mode, value=value)


def pad2dsame(x, kernel_size, dilation=1, mode='constant', value=0):
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    total_pad = [d * (k - 1) for d, k in zip(dilation, kernel_size)]
    left = [tp // 2 for tp in total_pad]
    right = [tp - l for tp, l in zip(total_pad, left)]
    padding = (left[0], right[0], left[1], right[1])
    return F.pad(x, padding, mode=mode, value=value)


class Pad2dMultiple(Module):
    def __init__(self, multiple, mode='constant', value=0):
        super().__init__()
        self.mode = mode
        self.value = value
        self.multiple = multiple

    def get_padding(self, x):
        _, _, h, w = x.shape
        # Swapping w and h because of the peculiarity of F.pad
        total_pad = [(-w) % self.multiple, (-h) % self.multiple]
        left = [tp // 2 for tp in total_pad]
        right = [tp - l for tp, l in zip(total_pad, left)]
        return (left[0], right[0], left[1], right[1])

    def forward(self, x):
        return F.pad(x, self.get_padding(x), mode=self.mode, value=self.value)

    def extra_repr(self):
        ret = f'multiple={self.multiple}, mode="{self.mode}"'
        if self.mode == 'constant':
            ret += f', value={self.value}'
        return ret


def pad2dmultiple(x, multiple, mode='constant', value=0):
    *_, h, w = x.shape
    # Swapping w and h because of the peculiarity of F.pad
    total_pad = [(-w) % multiple, (-h) % multiple]
    left = [tp // 2 for tp in total_pad]
    right = [tp - l for tp, l in zip(total_pad, left)]
    padding = (left[0], right[0], left[1], right[1])
    return F.pad(x, padding, mode=mode, value=value)


class UnPad2dMultiple(Module):
    def __init__(self, orig_shape, contiguous=True):
        super().__init__()
        self.orig_shape = orig_shape
        self.contiguous = contiguous

    def get_unpadding(self, x):
        *_, h, w = x.shape
        h_orig, w_orig = self.orig_shape
        total_pad = [h - h_orig, w - w_orig]
        left = [tp // 2 for tp in total_pad]
        right = [dim - tp + l for dim, tp, l in zip((h, w), total_pad, left)]
        return [left[0], right[0], left[1], right[1]]

    def forward(self, x):
        lh, rh, lw, rw = self.get_unpadding(x)
        ret = x[..., lh:rh, lw:rw]
        if self.contiguous:
            ret = ret.contiguous()
        return ret

    def extra_repr(self):
        return f'multiple={self.multiple}, contiguous={self.contiguous}'


# orig_shape must be just (h,w)
def unpad2dmultiple(x, orig_shape, contiguous=True):
    *_, h, w = x.shape
    h_orig, w_orig = orig_shape
    total_pad = [h - h_orig, w - w_orig]
    left = [tp // 2 for tp in total_pad]
    right = [dim - tp + l for dim, tp, l in zip((h, w), total_pad, left)]
    ret = x[..., left[0] : right[0], left[1] : right[1]]
    if contiguous:
        ret = ret.contiguous()
    return ret

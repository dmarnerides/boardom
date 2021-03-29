from functools import partial
import torch
import torch.nn.functional as F
from .conv import convsame, conv2dsame
from .module import Module


def gaussian_kernel_nd(kernel_size, std, ndim):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != ndim:
            raise RuntimeError(
                'Kernel sizes must match the number of dimensions.'
                f'Got kernel_size={kernel_size} and ndim={ndim}'
            )
    else:
        kernel_size = (kernel_size,) * ndim
    if isinstance(std, (list, tuple)):
        if len(std) != ndim:
            raise RuntimeError(
                'std must match the number of dimensions.'
                f'Got std={std} and ndim={ndim}'
            )
    else:
        std = (std,) * ndim

    grid = torch.meshgrid(
        [torch.arange(-(k - 1) / 2, (k - 1) / 2 + 1).float() for k in kernel_size]
    )
    grid = torch.stack(grid, 0).double()
    std = torch.Tensor(std).view(ndim, *[1] * ndim)
    kernel = grid.pow(2).div(-2 * (std ** 2)).sum(0, keepdim=False).exp()
    return kernel / kernel.sum()


def gaussian_blur(x, kernel_size, std, pad_mode='none', pad_value=0):
    ndim = x.ndim - 2
    c = x.shape[1]
    kernel = gaussian_kernel_nd(kernel_size, std, ndim)
    kernel = kernel.to(x.dtype, device=x.device)
    kernel = kernel[None, None, ...].expand(c, c, **((-1,) * ndim))
    if pad_mode == 'none':
        return F.conv(x, kernel)
    else:
        return convsame(x, kernel, pad_mode=pad_mode, pad_value=pad_value)


# Batched x (b,c,h,w)
def gaussian_blur2d(x, kernel_size, std, pad_mode='none', pad_value=0):
    c = x.shape[-3]
    kernel = gaussian_kernel_nd(kernel_size, std, 2).to(x.dtype)
    kernel = kernel.to(x.device)[None, None, :, :].expand(c, c, -1, -1)
    if pad_mode == 'none':
        return F.conv2d(x, kernel)
    else:
        return conv2dsame(x, kernel, pad_mode=pad_mode, pad_value=pad_value)


class GaussianBlur2d(Module):
    def __init__(self, kernel_size, std, pad_mode='none', pad_value=0):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) != 2:
                raise RuntimeError(
                    'Kernel size must be of size 2 but got kernel_size={kernel_size}.'
                )
        else:
            kernel_size = (kernel_size, kernel_size)
        if isinstance(std, (list, tuple)):
            if len(std) != 2:
                raise RuntimeError('std must be of size 2 but got std={std}.')
        else:
            std = (std, std)
        self.kernel_size = kernel_size
        self.std = std

        self.pad_mode = pad_mode
        self.pad_value = pad_value
        kernel = gaussian_kernel_nd(kernel_size, std, 2)
        kernel = kernel[None, None, :, :]
        self.register_buffer('kernel', kernel)
        if pad_mode == 'none':
            self.conv = F.conv2d
        else:
            self.conv = partial(conv2dsame, pad_mode=pad_mode, pad_value=pad_value)

    def forward(self, x):
        c = x.shape[-3]
        kernel = self.kernel.to(x.dtype).expand(c, c, *self.kernel.shape[-2:])
        return self.conv(x, kernel)

    def extra_repr(self):
        ret = f'kernel_size={self.kernel_size}'
        ret += f', std={self.std}'
        ret += f', pad_mode={self.pad_mode}'
        if self.pad_mode == 'constant':
            ret += f', pad_value={self.pad_value}'
        return ret

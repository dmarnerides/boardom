import torch
import boardom as bd
from .module import Module

# This is an excercise to show the equivalence.
# Don't use this for interpolating as it is very memory inefficient
# Assumes x is sorted
class PiecewiseMLP(Module):
    def __init__(self, x, y):
        super().__init__()
        x = self._check_input(x, 'x')
        y = self._check_input(y, 'y')
        if x.numel() != y.numel():
            raise RuntimeError(
                f'Expected x and y to be of same size, got {x.shape} and {y.shape}.'
            )
        x, y = x.view(-1), y.view(-1)
        if x.dtype != y.dtype:
            raise RuntimeError(
                f'Expected x and y with same dtype, got {x.dtype} and {y.dtype}.'
            )
        dtype = x.dtype
        self.register_buffer('l1_bias', -x[:-1].view(1, -1))
        self.register_buffer('l2_bias', y[0])
        grads = torch.zeros((x.numel(),), dtype=dtype)
        grads[1:] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        grad_diffs = grads[1:] - grads[:-1]
        self.register_buffer('l2_weight', grad_diffs.view(-1, 1))

    def _check_input(self, z, name):
        if bd.is_array(z):
            z = torch.from_numpy(z)
        if not bd.is_tensor(z):
            raise RuntimeError(f'Expected PyTorch Tensor or Numpy Array for {name}.')
        return z.view(-1)

    def forward(self, x):
        initial_shape = x.shape
        x = x.view(-1, 1)
        # l1 is [M, N]
        l1 = torch.relu(x + self.l1_bias)
        ret = torch.matmul(l1, self.l2_weight) + self.l2_bias
        return ret.view(initial_shape)


class Interp1d(Module):
    def __init__(self, x, y):
        super().__init__()
        x = self._check_input(x, 'x')
        y = self._check_input(y, 'y')
        if x.numel() != y.numel():
            raise RuntimeError(
                f'Expected x and y to be of same size, got {x.shape} and {y.shape}.'
            )
        x, y = x.view(-1), y.view(-1)
        if x.dtype != y.dtype:
            raise RuntimeError(
                f'Expected x and y with same dtype, got {x.dtype} and {y.dtype}.'
            )

        self.register_buffer('x', x)

        gradient = torch.zeros_like(x)
        gradient[1:] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        self.register_buffer('gradient', gradient)

        intercept = (gradient * x) - y
        self.register_buffer('intercept', intercept)

    def _check_input(self, z, name):
        if bd.is_array(z):
            z = torch.from_numpy(z)
        if not bd.is_tensor(z):
            raise RuntimeError(f'Expected PyTorch Tensor or Numpy Array for {name}.')
        return z.view(-1)

    def forward(self, x):
        idx = torch.searchsorted(self.x, x)
        return self.gradient[idx] * x - self.intercept[idx]

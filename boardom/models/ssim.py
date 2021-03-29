from functools import partial
from .guided_filter import box_filter2d
from .module import Module
from .gaussian import GaussianBlur2d


class SSIM(Module):
    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=11,
        std=1.5,
        pad_mode='none',
        pad_value=0,
        L=1,
        window='gaussian',
        #  exponents=[1, 1, 1],
    ):
        super().__init__()
        self.c1 = (k1 * L) * (k1 * L)
        self.c2 = (k2 * L) * (k2 * L)
        #  self.c3 = self.c2 / 2
        #  self.exponents = exponents
        if window == 'gaussian':
            self.conv_window = GaussianBlur2d(
                kernel_size=kernel_size, std=std, pad_mode=pad_mode, pad_value=pad_value
            )
        elif window == 'uniform':
            self.conv_window = partial(box_filter2d, kernel_size=kernel_size)

    def _get_stats(self, x):
        x_mean = self.conv_window(x)
        x_mean_squared = x_mean * x_mean
        x_var = self.conv_window(x * x) - x_mean_squared
        return x_mean, x_mean_squared, x_var

    # x is prediction, y is target, both assumed in the [0,1] range
    def compute_map(self, x, y):
        x, y = x, y
        #  alpha, beta, gamma = self.exponents
        x_mean, x_mean_squared, x_var = self._get_stats(x)
        y_mean, y_mean_squared, y_var = self._get_stats(y)
        xy_mean = x_mean * y_mean
        #  x_std = x_var.sqrt()
        #  y_std = y_var.sqrt()
        covariance = self.conv_window(x * y) - xy_mean

        numerator = 2 * xy_mean + self.c1
        numerator = numerator * (2 * covariance + self.c2)

        denominator = x_mean_squared + y_mean_squared + self.c1
        denominator = denominator * (x_var + y_var + self.c2)

        return numerator / denominator

        #  luminance = 2 * x_mean * y_mean + self.c1
        #  luminance = luminance / (x_mean_squared + y_mean_squared + self.c1)
        #  luminance = luminance.pow(alpha)
        #
        #  contrast = 2 * x_std * y_std + self.c2
        #  contrast = contrast / (x_var + y_var + self.c2)
        #  contrast = contrast.pow(beta)
        #
        #  structure = covariance + self.c3
        #  structure = structure / (x_std * y_std + self.c3)
        #  structure = structure.pow(gamma)
        #
        #  return luminance * contrast * structure

    # x is prediction, y is target, both assumed in the [0,1] range
    def forward(self, x, y):
        ssim_map = self.compute_map(x, y)
        return ssim_map.mean()


class SSIMLoss(SSIM):
    def forward(self, x, y):
        ssim = super().forward(x, y)
        return (1 - ssim) / 2

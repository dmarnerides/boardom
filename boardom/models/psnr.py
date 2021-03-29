import torch.nn.functional as F
from .module import Module


def psnr(x, y):
    mse = F.mse_loss(x, y)
    return -10 * mse.log10()


class PSNR(Module):
    def forward(self, x, y):
        return psnr(x, y)

import torch
from torch import nn
from .module import Module


class GramMatrix(Module):
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        return torch.bmm(x, x.transpose(1, 2)) / (h * w)


class GramLoss(Module):
    def __init__(self):
        super().__init__()
        self.gram = GramMatrix()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        g_x = self.gram(x)
        g_y = self.gram(y)
        return self.mse_loss(g_x, g_y)

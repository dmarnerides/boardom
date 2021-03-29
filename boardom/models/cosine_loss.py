from torch import nn
from .module import Module


class CosineLoss(Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=1, eps=eps)
        self.eps = eps

    def forward(self, x, y):
        return (1 - self.similarity(x + self.eps, y + self.eps)).mean()

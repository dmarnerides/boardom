import torch
import boardom as bd
import os

_ASS_PATH = os.path.dirname(os.path.abspath(__file__))


class _Assets(metaclass=bd.Singleton):
    @bd.once
    @property
    def cat(self):
        return bd.imread(os.path.join(_ASS_PATH, 'cat.png'))

    @bd.once
    @property
    def cameraman(self):
        return bd.imread(os.path.join(_ASS_PATH, 'cameraman.tif'))

    @bd.once
    @property
    def peppers(self):
        return bd.imread(os.path.join(_ASS_PATH, 'peppers.png'))

    @bd.once
    @property
    def ballroom(self):
        return bd.imread(os.path.join(_ASS_PATH, 'ballroom_1k.hdr'))

    @bd.once
    @property
    def venice_sunset(self):
        return bd.imread(os.path.join(_ASS_PATH, 'venice_sunset_1k.hdr'))

    @bd.once
    @property
    def pu_space(self):
        return torch.load(os.path.join(_ASS_PATH, 'pu_space.pth'))

    @bd.once
    @property
    def pu_spline_coeffs(self):
        return torch.load(os.path.join(_ASS_PATH, 'pu_spline_coeffs.pth'))


assets = _Assets()

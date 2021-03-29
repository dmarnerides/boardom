from .module import Module

# By default, total variation does not average over pixels, just sums them (reduction='sum')


class IsotropicTotalVariation(Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                f'Unknown reduction for IsotropicTotalVariation: {reduction}'
            )
        self.reduction = reduction

    def forward(self, x):
        d_w = x[..., :-1] - x[..., 1:]
        d_h = x[..., :-1, :] - x[..., 1:, :]
        ret = (d_w ** 2 + d_h ** 2).sqrt()
        if self.reduction == 'none':
            return ret
        elif self.reduction == 'sum':
            return ret.sum()
        else:
            return ret.mean()


class AnisotropicTotalVariation(Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                f'Unknown reduction for AnisotropicTotalVariation: {reduction}'
            )
        self.reduction = reduction

    def forward(self, x):
        d_w = (x[..., :-1] - x[..., 1:]).abs()
        d_h = (x[..., :-1, :] - x[..., 1:, :]).abs()
        return d_w.abs().sum() + d_h.abs().sum()
        if self.reduction == 'none':
            return d_w + d_h
        elif self.reduction == 'sum':
            return d_w.sum() + d_h.sum()
        else:
            return d_w.mean() + d_h.mean()


class TotalVariation(Module):
    def __init__(self, version='anisotropic', reduction='sum'):
        super().__init__()
        if version == 'anisotropic':
            self.tv = AnisotropicTotalVariation(reduction=reduction)
        elif version == 'isotropic':
            self.tv = IsotropicTotalVariation(reduction=reduction)
        else:
            raise RuntimeError(f'Invalid TotalVariation version: {version}')

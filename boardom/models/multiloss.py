from torch import nn
import boardom as bd
from .module import Module


class MultiLoss(Module):
    def __init__(self):
        super().__init__()
        self._losses = []
        self._names = []
        self._weights = []
        self._idx = {}

    def register(self, loss, name, weight=1.0):
        if name.startswith('_'):
            raise ValueError('Loss name can not start with "_".')
        if name == 'total':
            raise ValueError('Loss name can not be "total".')
        setattr(self, name, loss)
        # This is to get the built loss if it's provided in a "magic" context
        loss = getattr(self, name)
        weight = float(weight)
        if name in self._idx:
            bd.warn(f'Replacing {name} in MultiLoss')
            idx = self._idx[name]
            self._losses[idx] = loss
            self._names[idx] = name
            self._weights[idx] = weight
        else:
            self._losses.append(loss)
            self._names.append(name)
            self._weights.append(weight)
            self._idx[name] = len(self._names) - 1
        return self

    @property
    def loss_names(self):
        return self._names + ["total"]

    def __iter__(self):
        for x in zip(self._names, self._losses, self._weights):
            yield x

    def forward(self, *args, **kwargs):
        # Unweighted
        ret = {n: f(*args, **kwargs) for n, f in zip(self._names, self._losses)}
        weighted = {k: self._weights[self._idx[k]] * v for k, v in ret.items()}
        total = sum(weighted.values())
        ret['total'] = total
        return total, {k: v.detach().item() for k, v in ret.items()}

    def extra_repr(self):
        ret = ', '.join([f'{n}={w}' for n, w in zip(self._names, self._weights)])
        return 'Losses and weights: ' + ret

    def __contains__(self, item):
        return item in self._names

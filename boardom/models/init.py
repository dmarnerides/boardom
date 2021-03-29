import math
import torch
from torch import nn
from torch.nn import init
import boardom as bd


class Initializer:
    _registry = {}
    arg_prepend = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._initializer_key = cls.__name__.lower()
        Initializer._registry[cls._initializer_key] = cls

    def __init__(self, filter_fn=None):
        self.filter_fn = filter_fn

    def _filter(self, *args, **kwargs):
        if self.filter_fn is not None:
            return self.filter_fn(*args, **kwargs)
        else:
            return True

    def __call__(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError(
                'Initializer expected nn.Module as model '
                f'but got {torch.typename(model)}'
            )
        bd.print_separator()
        bd.log(f'Initializing {model.__class__.__name__} with {self}')
        for mname, module in model.named_modules():
            for pname, parameter in module._parameters.items():
                mod_type = torch.typename(module).split('.')[-1]
                if self._filter(module, mname, parameter, pname):
                    bd.log(f'Initializing "{pname}" in type {mod_type} module: {mname}')
                    yield parameter, (module, mname, pname)
        bd.print_separator()

    def apply(self, model):
        for parameter, _ in self(model):
            self.init_parameter(parameter)

    def init_parameter(self, parameter):
        raise NotImplementedError('Initializers must implement init_parameter.')

    def extra_repr(self):
        return ''

    def __repr__(self):
        return f'{self.__class__.__name__}({self.extra_repr()})'


class InitUniform(Initializer):
    arg_prepend = 'uniform'

    def __init__(self, a=0, b=1, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.a = a
        self.b = b

    def init_parameter(self, parameter):
        init.uniform_(parameter, a=self.a, b=self.b)

    def extra_repr(self):
        return f'[{self.a}, {self.b}]'


class InitNormal(Initializer):
    arg_prepend = 'normal'

    def __init__(self, mean=0, std=1, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.mean = mean
        self.std = std

    def init_parameter(self, parameter):
        init.normal_(parameter, mean=self.mean, std=self.std)

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'


class InitConstant(Initializer):
    arg_prepend = 'constant'

    def __init__(self, val, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.val = val

    def init_parameter(self, parameter):
        init.constant_(parameter, val=self.val)

    def extra_repr(self):
        return f'val={self.val}'


class InitOnes(InitConstant):
    def __init__(self, filter_fn=None):
        super().__init__(val=1, filter_fn=filter_fn)


class InitZeros(InitConstant):
    def __init__(self, filter_fn=None):
        super().__init__(val=0, filter_fn=filter_fn)


class InitEye(Initializer):
    def __init__(self, filter_fn=None):
        super().__init__(filter_fn=filter_fn)

    def init_parameter(self, parameter):
        init.eye_(parameter)


class InitDirac(Initializer):
    arg_prepend = 'dirac'

    def __init__(self, groups=1, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.groups = groups

    def init_parameter(self, parameter):
        init.dirac_(parameter, groups=self.groups)

    def extra_repr(self):
        return f'groups={self.groups}'


class InitXavierUniform(Initializer):
    arg_prepend = 'xavier'

    def __init__(self, gain=1.0, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.gain = gain

    def init_parameter(self, parameter):
        init.xavier_uniform_(parameter, gain=self.gain)

    def extra_repr(self):
        return f'gain={self.gain}'


class InitXavierNormal(Initializer):
    arg_prepend = 'xavier'

    def __init__(self, gain=1.0, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.gain = gain

    def init_parameter(self, parameter):
        init.xavier_normal_(parameter, gain=self.gain)

    def extra_repr(self):
        return f'gain={self.gain}'


class InitKaimingUniform(Initializer):
    arg_prepend = 'kaiming'

    def __init__(
        self,
        a=0,
        mode='fan_in',
        nonlinearity='leaky_relu',
        filter_fn=None,
    ):
        super().__init__(filter_fn=filter_fn)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def init_parameter(self, parameter):
        init.kaiming_uniform_(
            parameter, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )

    def extra_repr(self):
        return f'a={self.a}, mode={self.mode}, nonlinearity={self.nonlinearity}'


class InitKaimingNormal(Initializer):
    arg_prepend = 'kaiming'

    def __init__(
        self,
        a=0,
        mode='fan_in',
        nonlinearity='leaky_relu',
        filter_fn=None,
    ):
        super().__init__(filter_fn=filter_fn)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def init_parameter(self, parameter):
        init.kaiming_normal_(
            parameter, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )

    def extra_repr(self):
        return f'a={self.a}, mode={self.mode}, nonlinearity={self.nonlinearity}'


class InitOrthogonal(Initializer):
    arg_prepend = 'orthogonal'

    def __init__(self, gain=1.0, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.gain = gain

    def init_parameter(self, parameter):
        init.orthogonal_(parameter, gain=self.gain)

    def extra_repr(self):
        return f'gain={self.gain}'


class InitSparse(Initializer):
    arg_prepend = 'sparse'

    def __init__(self, sparsity=1.0, std=0.01, filter_fn=None):
        super().__init__(filter_fn=filter_fn)
        self.sparsity = sparsity
        self.std = std

    def init_parameter(self, parameter):
        init.sparse_(parameter, sparsity=self.sparsity, std=self.std)

    def extra_repr(self):
        return f'sparsity={self.sparsity}, std={self.std}'


class InitSELU(Initializer):
    def __init__(self, filter_fn=None):
        super().__init__(filter_fn=filter_fn)

    def init_parameter(self, parameter):
        fan = init._calculate_correct_fan(parameter, 'fan_in')
        init.normal_(parameter, mean=0, std=1.0 / math.sqrt(fan))

import types
import wrapt
import inspect
from .core import Null
from .external.decorator import _decorator


class CachedBoundFunctionWrapper(wrapt.BoundFunctionWrapper):
    def __call__(self, *args, **kwargs):
        instance = self.__self__
        name = self.__name__
        cached_name = f'_{name}_cached'
        if not hasattr(instance, cached_name):
            val = super().__call__(*args, **kwargs)
            setattr(instance, cached_name, val)
        return getattr(instance, cached_name)


class CachedFunctionWrapper(wrapt.FunctionWrapper):

    __bound_function_wrapper__ = CachedBoundFunctionWrapper

    def __call__(self, *args, **kwargs):
        if isinstance(self, property):
            instance = args[0]
            caller = self.fget
            name = self.fget.__name__
        else:
            instance = self
            caller = super().__call__
            name = self.__name__
        cached_name = f'_{name}_cached'
        if not hasattr(instance, cached_name):
            val = caller(*args, **kwargs)
            setattr(instance, cached_name, val)
        return getattr(instance, cached_name)


def once(wrapped):
    isprop = False
    if isinstance(wrapped, property):
        isprop = True

    if not (inspect.isfunction(wrapped) or isinstance(wrapped, property)):
        raise RuntimeError(
            f'bd.once expected a function or property, got {type(wrapped)}.'
        )

    if inspect.isclass(wrapped):
        raise RuntimeError('bd.once can not be applied to classes')

    @_decorator(functionwrapper=CachedFunctionWrapper)
    def wrapper(_wrapped, instance, args, kwargs):
        return _wrapped(*args, **kwargs)

    if isprop:
        return property(wrapper(wrapped), wrapped.fset, wrapped.fdel, wrapped.__doc__)
    else:
        return wrapper(wrapped)


def once_property(wrapped):
    return once(property(wrapped))

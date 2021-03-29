import inspect
import boardom as bd


def initialize(model, filter_fn=None):
    initializer_name = bd.cfg.initializer
    initializer_cls = bd.Initializer._registry[initializer_name]
    argnames = inspect.getfullargspec(initializer_cls).args[1:]
    argnames = [x for x in argnames if x != 'filter_fn']
    prepend = initializer_cls.arg_prepend
    args = {arg: bd.cfg[f'{prepend}_{arg}'] for arg in argnames}
    initializer_cls(**args, filter_fn=filter_fn).apply(model)

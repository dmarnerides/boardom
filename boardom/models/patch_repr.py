from .module import Module

# Aesthetic changes to model printing
# Adapted from original PyTorch code
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
class pretty_print:
    def __init__(self, indent=4, guides=True):
        self._old_repr = Module.__repr__
        self.guides = '|:·' if guides else ' '
        indent = min(max(indent, 1), 6)
        extra_spaces = indent - 1
        loc = 1
        self.indent = loc * ' ' + '{0}' + (extra_spaces - loc) * ' '

    def __enter__(self):
        Module.__repr__ = get_patched_repr(self.indent, self.guides)
        return self

    def __exit__(self, type, value, traceback):
        Module.__repr__ = self._old_repr


def get_patched_repr(indent, guide_chars):
    glen = len(guide_chars)

    def get_guide(i):
        return guide_chars[i % glen]

    def _addindent(s_, ind):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = '\n'.join([f'{ind}{line}' for line in s])
        return f'{first}\n{s}'

    def _patched_repr(self):
        if not hasattr(self, '_bd_repr_level'):
            self._bd_repr_level = 0

        ind = f'{indent.format(get_guide(self._bd_repr_level))}'
        # We treat the extra repr like the sub-module, one item per line
        extra_repr = self.extra_repr()
        extra_repr = extra_repr.split('\n') if extra_repr else []

        child_lines = []
        for key, module in self._modules.items():
            module._bd_repr_level = self._bd_repr_level + 1
            child_lines.append(f'[{key}]: {_addindent(repr(module), ind)}')
        lines = extra_repr + child_lines

        main_str = self._get_name()
        if len(self._modules) > 0:
            main_str += ':'

        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_repr) == 1 and not child_lines:
                main_str += f'({extra_repr[0]})'
            else:
                main_str += f'\n{ind}' + f'\n{ind}'.join(lines)

        if hasattr(self, '_bd_repr_level'):
            delattr(self, '_bd_repr_level')

        return main_str

    return _patched_repr

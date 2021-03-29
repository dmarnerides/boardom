from collections.abc import Mapping, Sequence

# TODO:
# > object.__contains__(self, item)
# > __repr__ like nn.Module
# TODO: Add tests for KeepDict


class KeepDict(dict):
    pass


class State(dict):
    def __init__(self, data_dict=None):
        if data_dict is None:
            data_dict = {}
        if not all(x.isidentifier() for x in data_dict):
            invalid = [f'"{x}"' for x in data_dict if not x.isidentifier()]
            raise KeyError(
                'bd.State only supports keys that can be valid Python identifiers. '
                f'Got: {invalid}'
            )
        super().__init__(
            {
                key: State(val)
                if isinstance(val, Mapping) and not isinstance(val, KeepDict)
                else val
                for key, val in data_dict.items()
            }
        )

    def __setattr__(self, key, value):
        try:
            self.__setitem__(key, value)
        except KeyError as e:
            raise AttributeError(str(e)) from e

    def __setitem__(self, key, value):
        # Accommodate properties
        try:
            try:
                splitkey = key.split('.')
            except AttributeError:
                splitkey = key

            curr_val, rest = super().__getitem__(splitkey[0]), splitkey[1:]
            if rest:
                if isinstance(curr_val, State):
                    curr_val = curr_val[rest]
                else:
                    raise KeyError(
                        'Non state items can not be directly accessed with recursive indices.'
                    )
            curr_val.fset(None, value)  # self already bound with closure
            return
        except AttributeError:
            pass
        except TypeError:
            pass
        except KeyError as e:
            if '@property' in str(e):
                raise

        if isinstance(value, Mapping) and not isinstance(value, (State, KeepDict)):
            value = State(value)
        if (not isinstance(key, str)) and isinstance(key, Sequence):
            if len(key) == 0:
                raise KeyError('Key Sequence is empty.')
            elif len(key) == 1:
                self[key[0]] = value
            else:
                self[key[0]][key[1:]] = value
        elif not isinstance(key, str):
            raise KeyError(
                'Keys can only be strings or sequences of strings. '
                f'Got {type(key)} for key {key}.'
            )
        else:
            key = key.strip('. ').split('.')
            if len(key) == 1:
                key = key[0]
                if not key.isidentifier():
                    raise KeyError(
                        'bd.State only supports keys that can be valid Python identifiers. '
                        f'Got: "{key}"'
                    )
                super().__setitem__(key, value)
            else:
                self[key] = value

    def __repr__(self):
        return f'State({super().__repr__()})'

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError as e:
            # This is so that hasattr works on State
            raise AttributeError(str(e)) from e

    def __getitem__(self, key):
        try:
            key = key.split('.')
        except AttributeError:
            pass

        ret, rest = super().__getitem__(key[0]), key[1:]
        if rest:
            if isinstance(ret, State):
                ret = ret[rest]
            else:
                raise KeyError(
                    'Non state items can not be directly accessed with recursive indices.'
                )
        if isinstance(ret, property):
            fget = ret.fget
            if fget is None:
                raise KeyError(f'Can not read {key} attribute. fget not provided')
            ret = fget(None)  # self already bound with closure
        return ret

    def __call__(self, *args, **kwargs):
        raise RuntimeError('Could not find registered event.')

    def __delattr__(self, key):
        try:
            super().__delitem__(key)
        except KeyError as e:
            raise AttributeError(str(e)) from e

    def __delitem__(self, key):
        try:
            key = key.split('.')
        except AttributeError:
            pass
        if len(key) > 1:
            substate, rest = super().__getitem__(key[0]), key[1:]
            del substate[rest]
        else:
            super().__delitem__(key[0])

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def update(self, other):
        for k, val in other.items():
            if isinstance(val, Mapping):
                if (k not in self) or not isinstance(self[k], State):
                    self[k] = State()
                self[k].update(val)
            else:
                self[k] = val

    def __hash__(self):
        return id(self)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

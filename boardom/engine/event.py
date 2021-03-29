class EventReturnValue(list):
    pass


def _check_valid_eventname(name, should_raise=True):
    if not isinstance(name, str):
        if should_raise:
            raise TypeError(f'Event names must be strings (got {type(name)})')
        else:
            return False
    if not name.isidentifier():
        if should_raise:
            raise ValueError(f'Event names is not a valid identifier (got "{name}").')
        else:
            return False
    if name.startswith('_'):
        if should_raise:
            raise ValueError(
                f'Event names should not start with underscore (got "{name}").'
            )
        else:
            return False
    return True


class Event:
    __slots__ = ['name', 'args', 'kwargs']

    def __init__(self, name, args, kwargs):
        _check_valid_eventname(name)
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return self.name

    def __str__(self):
        return f'Event({self.name}, args={self.args}, kwargs={self.kwargs}'

    def __eq__(self, other):
        return self.name == repr(other)

    def __hash__(self):
        return hash(self.name)

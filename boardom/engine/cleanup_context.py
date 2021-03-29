import boardom as bd


def _recurse_get_keys(state):
    ret = set()
    for p, o in state.items():
        if isinstance(o, bd.State):
            ret |= set(f'{p}.{k}' for k in _recurse_get_keys(o))
    return ret | set(state.keys())


class CleanupState:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        self.start_keys = _recurse_get_keys(self.engine)

    def __exit__(self, exc_type, exc_value, traceback):
        end_keys = _recurse_get_keys(self.engine)
        new_keys = end_keys - self.start_keys
        for new_key in new_keys:
            if new_key in self.engine:
                del self.engine[new_key]

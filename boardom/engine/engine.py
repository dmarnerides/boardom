import inspect
from collections.abc import Mapping
from functools import partial
import boardom as bd
from .event import Event, _check_valid_eventname, EventReturnValue
from .state import State
from .action import Action

# TODO: __hasattr__
# TODO: __repr__
# TODO: saving and loading state
# TODO: Implement async / multithreading / multiprocessing executors
# TODO: Maybe make events be UPPERCASE only?

_OWN = ['state', '_actions', '_prior', 'get_no_event']
_API = ['register', 'event'] + [x for x in dir({}) if not x.startswith('_')]
_ALL = _OWN + _API


class _NonExisting(metaclass=bd.Singleton):
    pass


NonExisting = _NonExisting()


def _makeboundprop(obj, old_prop):
    def fget(self):
        return old_prop.fget(obj)

    if old_prop.fset is None:
        fset = None
    else:

        def fset(self, value):
            old_prop.fset(obj, value)

    if old_prop.fdel is None:
        fdel = None
    else:

        def fdel(self):
            old_prop.fdel(obj)

    return property(fget, fset, fdel, old_prop.__doc__)


# This is to avoid masking of inherited functions.
# e.g.
#  class Base(bd.Engine):
#      def func(self):
#          # This will run too
#          pass
#  class Derived(self):
#      @bd.on('func')
#      def other(self):
#          #This will run
#          pass
#  Derived().func()
#


def bind_func_and_event_of_same_name(orig_func, event_fn, name, action_dict, engine):
    def ret_fn(*args, **kwargs):
        orig_fn_action = Action(name, orig_func)
        action_list = list(action_dict[name].values())
        if orig_fn_action in action_list:
            action_list.remove(orig_fn_action)
        action_list = [orig_fn_action] + action_list
        event = Event(name, args, kwargs)
        return EventReturnValue(action(engine, event) for action in action_list)

    ret_fn._bound_func_with_event = orig_func

    return ret_fn


def _delegated_dict_api(self, method):
    def func(*args, **kwargs):
        return getattr(self.state, method)(*args, **kwargs)

    return func


class Engine:
    def __init__(self, *components):
        self._actions = {}
        self._prior = []
        self.state = State()
        # Iterate over members and add things to state if needed
        for key in dir(self):
            if key.startswith('__') or (key in _ALL):
                continue
            # Handle properties
            if isinstance(getattr(self.__class__, key), property):
                val = _makeboundprop(self, getattr(self.__class__, key))
            else:
                val = getattr(self, key)
            if inspect.ismethod(val) or inspect.isfunction(val):
                continue
            # Delete the attribute from instance and set it to state using setattr
            try:
                delattr(self, key)
            except AttributeError:
                # If the attribute is not an instance one but a class one
                # delattr raises an AttributeError
                pass
            setattr(self, key, val)

            self._prior.append(key)

        for key in dir({}):
            if key.startswith('_'):
                continue
            func = _delegated_dict_api(self, key)
            object.__setattr__(self, key, func)

        # Register events at the end
        self.register(self, *components)

    def register(self, *component):
        if len(component) < 1:
            raise RuntimeError('Engine.register did not receive and components')
        if len(component) == 1:
            component = component[0]

        events = []
        # Events from adding as a tuple or list in the form:
        #     eng.register('foo', 'bar', func)
        # Or
        #     eng.register(('foo', 'bar', func))
        if isinstance(component, (list, tuple)):
            events += list(component[:-1])
            component = component[-1]
        for x in events:
            _check_valid_eventname(x)

        # Add any events from decorator
        if hasattr(component, '_bd_engine_events'):
            events += component._bd_engine_events

        # Bound or staticmethods are registered on __func__
        if hasattr(component, '__func__') and hasattr(
            component.__func__, '_bd_engine_events'
        ):
            events += component.__func__._bd_engine_events

        events = list(set(events))

        # If object or class is given, recurse and register members
        for key, val in inspect.getmembers(component):
            # This is to avoid doubly processing bound methods
            if key == '__func__':
                continue
            if hasattr(val, '_bd_engine_events'):
                self.register(val)

        # For objects with attach method, run attach
        if hasattr(component, 'attach'):
            component.attach(self)

        # Register all events
        for event in events:
            if not callable(component):
                raise RuntimeError(
                    f'Attempted to register non callable {component} in Engine'
                )
            action_dict = self._actions.get(event, {})
            action = Action(event, component)
            already_exists = action.id in action_dict
            action_dict[action.id] = action
            self._actions[event] = action_dict
            if already_exists:
                bd.warn(f'Compoenent {component} already defined for event {event}.')
            #  else:
            #      bd.log(f'Registered {component} for event "{event}" on engine {self}.')
        return self

    def event(self, name, *args, **kwargs):
        action_dict = object.__getattribute__(self, '_actions')
        if name not in action_dict:
            return
        event = Event(name, args, kwargs)
        actions = action_dict[name].values()
        return EventReturnValue(action(self, event) for action in actions)

    def __call__(self, name, *args, **kwargs):
        return self.event(name, *args, **kwargs)

    def get_no_event(self, key):
        return object.__getattribute__(self, key)

    # Needs optimizing maybe
    def __getattribute__(self, key):
        raised = False
        try:
            default = object.__getattribute__(self, key)
        except AttributeError as final_exc:
            raised = final_exc

        try:
            # Delegate attribute logic to __getitem__
            action_dict = object.__getattribute__(self, '_actions')
            if key in action_dict:
                ret = partial(object.__getattribute__(self, 'event'), key)
                if not raised:
                    if not callable(default):
                        raise AttributeError(
                            'Attribute key is not a function and the name is also used as an event.'
                        )
                    return bind_func_and_event_of_same_name(
                        default, ret, key, action_dict, self
                    )
                return ret

            else:
                return object.__getattribute__(self, 'state')[key]
        except KeyError:
            pass
        if key in object.__getattribute__(self, '_prior'):
            # key was defined in class or instance and moved to state but then it was deleted
            raise AttributeError(f'{key} is not accessible.')
        if raised:
            raise AttributeError(str(raised))
        return default

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from e

    def __setattr__(self, key, value):
        if key in _API:
            raise AttributeError(f'Can not set {key} of Engine.')
        if key == 'state':
            if not isinstance(value, Mapping):
                raise TypeError(
                    'Expected object of type mapping for "state" attribute of engine.'
                )
            super().__setattr__(key, State())
            super().__getattribute__(key).update(value)
            return
        if key in _OWN:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if key in _API:
            raise KeyError(f'Can not set {key} of Engine')
        if key == 'state':
            if not isinstance(value, Mapping):
                raise TypeError(
                    'Expected object of type mapping for "state" item of engine.'
                )
            super().__setattr__('state', State())
            super().__getattribute__(key).update(value)
            return
        if key in _OWN:
            try:
                super().__setattr__(key, value)
            except AttributeError as e:
                raise KeyError(str(e)) from e
        else:
            if key in self._actions:
                raise KeyError(
                    f'Attempted to set "{key}" for engine, '
                    'but it is already defined as an event.'
                )
            self.state.__setitem__(key, value)

    def __delattr__(self, key):
        try:
            super().__delattr__(key)
            return
        except AttributeError:
            pass
        try:
            self.state.__delitem__(key)
        except KeyError as e:
            raise AttributeError(str(e)) from e

    def __delitem__(self, key):
        del self.state[key]

    def __contains__(self, key):
        return key in self.state

    # This is to inherit @on assignments
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own_members = {key: val for key, val in inspect.getmembers(cls)}
        for base in cls.__bases__:
            for key, base_method in inspect.getmembers(base):
                if (key in own_members) and hasattr(base_method, '_bd_engine_events'):
                    events = base_method._bd_engine_events
                    own_method = own_members[key]
                    if hasattr(own_method, '_bd_engine_events'):
                        events = own_method._bd_engine_events + events
                    own_method._bd_engine_events = events

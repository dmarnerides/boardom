import pytest
import boardom as bd


class TestEngine:
    def test_can_be_created(self):
        bd.Engine()

    def test_can_set_item_to_state(self):
        eng = bd.Engine()
        eng['a'] = 10
        assert eng['a'] == 10

    def test_can_set_attribute_to_state(self):
        eng = bd.Engine()
        eng.a = 10
        assert eng.a == 10
        assert eng['a'] == 10
        eng['b'] = 15
        assert eng.b == 15
        assert eng['b'] == 15

    def test_can_delete_item_and_attribute_from_state(self):
        eng = bd.Engine()
        eng['a'] = 10
        eng.b = 42
        assert eng['a'] == 10
        assert eng.a == 10
        assert eng['b'] == 42
        assert eng.b == 42
        del eng['a']
        with pytest.raises(KeyError):
            eng['a']
        with pytest.raises(AttributeError):
            eng.a
        assert eng.b == 42
        del eng.b
        with pytest.raises(KeyError):
            eng['b']
        with pytest.raises(AttributeError):
            eng.b

    def test_can_set_and_access_nested_attributes_and_items(self):
        eng = bd.Engine()
        eng.a = {'b': {'c': 10}}
        assert eng.a.b.c == 10
        assert eng.a['b'].c == 10
        assert eng.a.b['c'] == 10
        assert eng['a.b.c'] == 10
        eng['a.b.c'] = 15
        assert eng.a.b.c == 15
        assert eng.a['b'].c == 15
        assert eng.a.b['c'] == 15
        assert eng['a.b.c'] == 15

    def test_can_assign_the_state_attribute(self):
        eng = bd.Engine()
        eng.a = 2
        assert eng.a == 2
        eng.state = {'a': 3}
        assert eng.a == 3
        assert isinstance(eng.state, bd.State)

    def test_can_assign_the_state_item(self):
        eng = bd.Engine()
        eng.a = 2
        assert eng.a == 2
        eng['state'] = {'a': 3}
        assert eng.a == 3
        assert isinstance(eng.state, bd.State)

    def test_nested_dicts_of_assigned_state_attributes_become_state(self):
        eng = bd.Engine()
        eng.a = 2
        assert eng.a == 2
        eng.state = {'a': 3, 'b': {'c': 5}}
        assert eng.a == 3
        assert isinstance(eng.state, bd.State)
        assert isinstance(eng.b, bd.State)
        assert eng.b.c == 5

    def test_nested_dicts_of_assigned_state_items_become_state(self):
        eng = bd.Engine()
        eng.a = 2
        assert eng.a == 2
        eng['state'] = {'a': 3, 'b': {'c': 5}}
        assert eng.a == 3
        assert isinstance(eng.state, bd.State)
        assert isinstance(eng.b, bd.State)
        assert eng.b.c == 5

    def test_can_assign_the_state_attribute_only_with_mapping(self):
        eng = bd.Engine()
        with pytest.raises(TypeError) as e:
            eng.state = 5
        assert 'Expected object of type mapping' in str(e)
        with pytest.raises(TypeError) as e:
            eng['state'] = 5
        assert 'Expected object of type mapping' in str(e)

    def test_can_not_set_reserved_attributes_and_items(self):
        from boardom.engine.engine import _API

        eng = bd.Engine()
        for key in _API:
            with pytest.raises(AttributeError) as e:
                setattr(eng, key, 100)
            assert 'Can not set' in str(e)
            with pytest.raises(KeyError) as e:
                eng[key] = 100
            assert 'Can not set' in str(e)

    def test_dict_api_is_delegated_to_state(self):
        eng = bd.Engine()
        eng['a'] = 10
        assert eng['a'] == 10
        for key, val in eng.items():
            assert key == 'a'
            assert val == 10

        eng['b'] = 100
        assert eng.b == 100
        x = eng.pop('b')
        assert x == 100
        with pytest.raises(AttributeError) as e:
            eng.b
        assert 'has no attribute' in str(e)
        eng.b = 100

        eng_2 = bd.Engine()
        eng_2.a = 32
        eng_2.c = 50
        assert eng.a == 10
        assert eng.b == 100
        with pytest.raises(AttributeError):
            eng.c

        eng.update(eng_2)
        assert eng.a == 32
        assert eng.b == 100
        assert eng.c == 50
        assert eng_2.a == 32
        with pytest.raises(AttributeError):
            eng_2.b
        assert eng_2.c == 50

    def test_unregistered_events_run_with_any_args_and_kwargs_and_return_none(
        self,
    ):
        eng = bd.Engine()

        ret = eng.event('unreg')
        assert ret is None

    def test_can_register_function_single_event(self):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', f)
        ret = eng.event('do')
        assert ret == [100]

    def test_event_functions_return_eventreturnvalue_which_is_list(self):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', f)
        ret = eng.event('do')
        assert isinstance(ret, list)
        assert isinstance(ret, bd.EventReturnValue)

    def test_can_register_function_twice_but_runs_once(self):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', f)
        eng.register('do', f)
        ret = eng.event('do')
        assert ret == [100]

    def test_can_register_function_with_multiple_events(self):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', 'do2', f)
        ret = eng.event('do')
        assert ret == [100]
        ret = eng.event('do2')
        assert ret == [100]

    def test_can_register_function_with_multiple_events_in_different_calls_of_register(
        self,
    ):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', f)
        eng.register('do2', 'do2', f)
        ret = eng.event('do')
        assert ret == [100]
        ret = eng.event('do2')
        assert ret == [100]

    def test_can_register_function_on_different_engines(self):
        eng = bd.Engine()
        eng_2 = bd.Engine()

        def f(engine):
            return 100

        eng.register('do', f)
        ret = eng.event('do')
        assert ret == [100]

        eng_2.register('do', f)
        ret = eng_2.event('do')
        assert ret == [100]

    def test_can_use_event_name_to_call_func(self):
        eng = bd.Engine()

        def f(engine):
            return 100

        eng.register('name', f)
        assert eng.name() == [100]

    def test_can_pass_args_to_func(self):
        eng = bd.Engine()

        def f(engine, a, b):
            return a * b

        def g(a, engine, b):
            return a * b

        def h(a, b, engine):
            return a * b

        def w(a, b):
            return a * b

        eng.register('f', f)
        eng.register('g', g)
        eng.register('h', h)
        eng.register('w', w)

        assert eng.f(2, 3) == [6]
        assert eng.g(2, 3) == [6]
        assert eng.h(2, 3) == [6]
        assert eng.w(2, 3) == [6]

    def test_can_pass_kwargs_to_func(self):
        eng = bd.Engine()

        def f(engine, a, b):
            return a * b

        def g(a, engine, b):
            return a * b

        def h(a, b, engine):
            return a * b

        eng.register('f', f)
        eng.register('g', g)
        eng.register('h', h)

        # f no engine
        assert eng.f(a=2, b=3) == [6]
        assert eng.f(2, b=3) == [6]

        # f with engine
        assert eng.f(eng, 2, b=3) == [6]
        assert eng.f(eng, a=2, b=3) == [6]
        assert eng.f(engine=eng, a=2, b=3) == [6]

        # g no engine
        assert eng.g(a=2, b=3) == [6]
        assert eng.g(2, b=3) == [6]

        # g with engine
        assert eng.g(2, eng, b=3) == [6]
        assert eng.g(2, engine=eng, b=3) == [6]
        assert eng.g(a=2, engine=eng, b=3) == [6]

        # h no engine
        assert eng.h(a=2, b=3) == [6]
        assert eng.h(2, b=3) == [6]

        # h with engine
        assert eng.h(2, 3, eng) == [6]
        assert eng.h(2, 3, engine=eng) == [6]
        assert eng.h(2, b=3, engine=eng) == [6]
        assert eng.h(a=2, b=3, engine=eng) == [6]

    def test_works_with_default_kwargs(self):
        eng_1 = bd.Engine()
        eng_1.x = 7
        eng_2 = bd.Engine()
        eng_2.x = 11

        def f(engine, a, b=5):
            return a * b * engine.x

        eng_1.register('f', f)
        # f eng_1
        assert eng_1.f(2) == [70]
        assert eng_1.f(2, 3) == [42]
        assert eng_1.f(2, b=3) == [42]
        assert eng_1.f(a=2, b=3) == [42]
        # f eng_1 passed
        assert eng_1.f(2, engine=eng_1) == [70]
        assert eng_1.f(2, 3, engine=eng_1) == [42]
        assert eng_1.f(2, b=3, engine=eng_1) == [42]
        assert eng_1.f(a=2, engine=eng_1, b=3) == [42]
        # f eng_2 passed
        assert eng_1.f(2, engine=eng_2) == [110]
        assert eng_1.f(2, 3, engine=eng_2) == [66]
        assert eng_1.f(2, b=3, engine=eng_2) == [66]
        assert eng_1.f(a=2, engine=eng_2, b=3) == [66]

        def g(a, engine=eng_2, b=5):
            return a * b * engine.x

        eng_1.register('g', g)
        # g eng_2 is default
        assert eng_1.g(2) == [110]
        assert eng_1.g(2, 3) == [66]
        assert eng_1.g(2, b=3) == [66]
        assert eng_1.g(a=2, b=3) == [66]
        # g eng_1 passed kwarg
        assert eng_1.g(2, engine=eng_1) == [70]
        assert eng_1.g(2, 3, engine=eng_1) == [42]
        assert eng_1.g(2, b=3, engine=eng_1) == [42]
        assert eng_1.g(a=2, engine=eng_1, b=3) == [42]
        # g eng_1 passed arg
        assert eng_1.g(2, eng_1) == [70]
        assert eng_1.g(2, eng_1, 3) == [42]
        with pytest.raises(RuntimeError) as e:
            eng_1.g(2, 3, eng_1)
        assert 'extra args' in str(e)
        assert eng_1.g(2, eng_1, b=3) == [42]
        assert eng_1.g(engine=eng_1, a=2, b=3) == [42]
        # g eng_2 passed
        assert eng_1.g(2, engine=eng_2) == [110]
        assert eng_1.g(2, 3, engine=eng_2) == [66]
        assert eng_1.g(2, b=3, engine=eng_2) == [66]
        assert eng_1.g(a=2, engine=eng_2, b=3) == [66]

        def h(a=13, b=5, engine=eng_2):
            return a * b * engine.x

        eng_1.register('h', h)
        # h eng_2 is default
        assert eng_1.h() == [715]
        assert eng_1.h(2) == [110]
        assert eng_1.h(2, 3) == [66]
        assert eng_1.h(2, b=3) == [66]
        assert eng_1.h(a=2, b=3) == [66]
        # h eng_1 passed kwarg
        assert eng_1.h(2, engine=eng_1) == [70]
        assert eng_1.h(2, 3, engine=eng_1) == [42]
        assert eng_1.h(b=3, engine=eng_1) == [273]
        assert eng_1.h(2, b=3, engine=eng_1) == [42]
        assert eng_1.h(a=2, engine=eng_1, b=3) == [42]
        # h eng_1 passed arg
        with pytest.raises(RuntimeError) as e:
            eng_1.h(2, eng_1, b=3)
        assert 'extra args' in str(e)
        assert eng_1.h(2, 3, eng_1) == [42]
        # h eng_2 passed
        assert eng_1.h(2, engine=eng_2) == [110]
        assert eng_1.h(2, 3, engine=eng_2) == [66]
        assert eng_1.h(2, b=3, engine=eng_2) == [66]
        assert eng_1.h(a=2, engine=eng_2, b=3) == [66]

    def test_can_register_lambda(self):
        eng = bd.Engine()
        eng.register('f', lambda engine, a, b: a * b)
        assert eng.f(2, 3) == [6]

    def test_can_pass_different_engine_to_function(self):
        eng_1, eng_2 = bd.Engine(), bd.Engine()
        eng_1.x = 10
        eng_2.x = 20

        def f(engine):
            return engine.x

        eng_1.register('f', f)
        assert eng_1.f() == [10]
        assert eng_1.f(eng_1) == [10]
        assert eng_1.f(eng_2) == [20]

    def test_can_register_with_decorator(self):
        eng = bd.Engine()

        @bd.on('foo')
        def f(a, b):
            return a * b

        eng.register(f)

        assert eng.foo(2, 3) == [6]

    def test_can_register_staticmethod(self):
        class Foo:
            @staticmethod
            def bar(a, b):
                return a * b

        eng = bd.Engine()
        eng.register('x', Foo.bar)
        eng.register('y', Foo().bar)
        assert eng.x(2, 3) == [6]
        assert eng.y(2, 3) == [6]

        class Foo:
            @bd.on('z')
            @staticmethod
            def bar(a, b):
                return a * b

        eng.register(Foo.bar)
        assert eng.z(2, 3) == [6]

        class Baz:
            @staticmethod
            @bd.on('w')
            def baz(a, b):
                return a * b

        eng.register(Baz.baz)
        assert eng.w(2, 3) == [6]

    def test_can_register_class_members(self):
        class Foo:
            @bd.on('st')
            @staticmethod
            def static(a, b):
                return a * b

            @bd.on('meth')
            def method(self, a, b):
                return a * b

            @bd.on('meth_eng')
            def method_w_engine(self, a, b, engine):
                return a * b

            @staticmethod
            def static_no_on(self, a, b):
                return a * b

            def method_no_on(self, a, b):
                return a * b

            def to_attach(a, b):
                return a * b

            def attach(engine):
                engine.register('from_attach', Foo.to_attach)

        eng = bd.Engine().register(Foo)
        eng.register('st_no_on', Foo.static_no_on)
        eng.register('meth_no_on', Foo.method_no_on)
        assert eng.st(2, 3) == [6]
        assert eng.meth(None, 2, 3) == [6]

        with pytest.raises(TypeError) as e:
            eng.meth(2, 3) == [6]
        assert 'missing 1 required' in str(e)

        assert eng.meth_eng(None, 2, 3) == [6]
        with pytest.raises(TypeError) as e:
            eng.meth_eng(2, 3) == [6]
        assert 'missing 1 required' in str(e)
        assert eng.meth_eng(None, 2, 3, eng) == [6]

        assert eng.st_no_on(None, 2, 3) == [6]
        assert eng.meth_no_on(None, 2, 3) == [6]
        assert eng.from_attach(2, 3) == [6]

    def test_can_register_object_members(self):
        class Foo:
            def __init__(self, val):
                self.val = val

            @bd.on('st')
            @staticmethod
            def static(a, b):
                return a * b

            @bd.on('meth')
            def method(self, a, b):
                return a * b * self.val

            @bd.on('meth_eng')
            def method_w_engine(self, a, b, engine):
                return a * b * self.val

            @staticmethod
            def static_no_on(self, a, b):
                return a * b * self.val

            def method_no_on(self, a, b):
                return a * b * self.val

            def to_attach(self, a, b):
                return a * b * self.val

            def attach(self, engine):
                engine.register('from_attach', self.to_attach)

        class DummyForStaticSelfVal:
            val = 7

        obj = Foo(5)
        eng = bd.Engine().register(obj)
        eng.register('st_no_on', obj.static_no_on)
        eng.register('meth_no_on', obj.method_no_on)

        assert eng.st(2, 3) == [6]
        assert eng.meth(2, 3) == [30]

        assert eng.meth_eng(2, 3) == [30]
        assert eng.meth_eng(2, 3, eng) == [30]

        assert eng.st_no_on(DummyForStaticSelfVal, 2, 3) == [42]
        assert eng.meth_no_on(2, 3) == [30]
        assert eng.from_attach(2, 3) == [30]

    def test_can_inherit_from_engine(self):
        class Foo(bd.Engine):
            val_out = -5

            def __init__(self, val):
                super().__init__()
                self.val_in = val

            @bd.on('st')
            @staticmethod
            def static(a, b):
                return a * b * Foo.val_out

            @bd.on('meth')
            def method(self, a, b):
                return a * b * Foo.val_out * self.val_out * self.val_in

            @bd.on('meth_eng')
            def method_w_engine(self, a, b, engine):
                return a * b * Foo.val_out * self.val_out * self.val_in

            @staticmethod
            def static_no_on(self, a, b):
                return a * b * self.val

            def method_no_on(self, a, b):
                return a * b * Foo.val_out * self.val_out * self.val_in

            def to_attach(self, a, b):
                return a * b * Foo.val_out * self.val_out * self.val_in

            def attach(self, engine):
                engine.register('from_attach', self.to_attach)

        class DummyForStaticSelfVal:
            val = 11

        eng = Foo(7)
        assert eng.val_out == -5
        eng.val_out = 5
        assert eng.val_out == 5
        eng.register('st_no_on', eng.static_no_on)
        eng.register('meth_no_on', eng.method_no_on)

        assert eng.st(2, 3) == [-30]
        # Accessing the method works as well,
        # but now it's just a method, not an event
        assert eng.static(2, 3) == -30
        assert eng.meth(2, 3) == [-1050]

        assert eng.meth_eng(2, 3) == [-1050]
        assert eng.meth_eng(2, 3, eng) == [-1050]

        assert eng.st_no_on(DummyForStaticSelfVal, 2, 3) == [66]
        assert eng.meth_no_on(2, 3) == [-1050]
        assert eng.from_attach(2, 3) == [-1050]

    def test_child_engine_properties_work(self):
        class Foo(bd.Engine):
            def __init__(self, val):
                super().__init__()
                self._x = val
                self.got_value = False
                self.set_value = False

            @property
            def x(self):
                self.got_value = True
                return self._x

            @x.setter
            def x(self, val):
                self.set_value = True
                self._x = val

        f = Foo(5)
        assert not f.got_value
        assert f.x == 5
        assert f.got_value
        assert not f.set_value
        f.x = -5
        assert f.x == -5
        assert f.set_value

    def test_action_takes_precedence_over_member_with_same_name(self):
        class Foo(bd.Engine):
            val = 2

            @bd.on('foo')
            def foo(self):
                return self.val

        class Other(bd.Engine):
            val = 5

        f = Foo()
        assert f.foo() == [2]
        assert f.foo(Other()) == [5]

    def test_child_engine_self_injection_works(self):
        class Foo(bd.Engine):
            val = 2

            @bd.on('action')
            def foo(self):
                self.val = 3 * self.val
                return self.val * 11

        class Bar(bd.Engine):
            val = 5

            @bd.on('action')
            def bar(self):
                self.val = 7 * self.val
                return self.val * 13

        f = Foo()
        b = Bar()

        assert f.val == 2
        assert f.action() == [2 * 3 * 11]
        assert f.val == 2 * 3
        assert Foo.val == 2

        assert b.val == 5
        assert b.action() == [5 * 7 * 13]
        assert b.val == 5 * 7
        assert Bar.val == 5

        assert f.action(b) == [3 * 5 * 7 * 11]
        assert f.val == 2 * 3
        assert b.val == 3 * 5 * 7

        assert b.action(f) == [2 * 3 * 7 * 13]
        assert f.val == 2 * 3 * 7
        assert b.val == 3 * 5 * 7

    def test_engine_clashes_in_definition(self):
        class Default(bd.Engine):
            val = 11

        default_eng = Default()

        class Foo(bd.Engine):
            val = 2

            @bd.on('foo')
            def foo(self, engine):
                self.val = 3 * self.val
                return self.val * 5

            @bd.on('bar')
            def bar(self, a, engine):
                self.val = 3 * self.val
                return self.val * 5

            @bd.on('baz')
            def baz(self, a, engine=default_eng):
                self.val = 3 * self.val
                return self.val * 5

        class Other(bd.Engine):
            val = 7

        f = Foo()
        other = Other()

        #######
        # Foo
        #######
        # No arg
        assert f.val == 2
        assert f.foo() == [30]
        assert f.val == 6
        f.val = 2

        # 1 arg (goes to self, engine is ignored)
        assert f.foo(f) == [30]
        assert f.val == 6
        f.val = 2

        assert f.foo(other) == [3 * 7 * 5]
        assert other.val == 21
        assert f.val == 2
        other.val = 7

        # 2 args
        # engine keyword takes precedence
        assert f.foo(f, f) == [30]
        f.val = 2
        assert f.foo(f, other) == [3 * 7 * 5]
        other.val = 7
        assert f.foo(other, f) == [30]
        f.val = 2
        assert f.foo(other, other) == [3 * 7 * 5]
        other.val = 7

        # 3 args is wrong
        with pytest.raises(RuntimeError) as e:
            f.foo(f, f, f)
        assert 'extra args' in str(e)

        # 3 args + kwarg is wrong
        with pytest.raises(RuntimeError) as e:
            f.foo(f, f, f, engine=f)
        assert 'extra args' in str(e)

        # > 1 args + kwarg is wrong
        with pytest.raises(RuntimeError) as e:
            f.foo(f, f, engine=f)
        assert 'extra args' in str(e)
        with pytest.raises(RuntimeError) as e:
            f.foo(f, f, f, engine=f)
        assert 'extra args' in str(e)

        # kwarg takes precedence
        other.val = 7
        assert f.foo(f, engine=other) == [3 * 7 * 5]
        other.val = 7
        assert f.foo(engine=other) == [3 * 7 * 5]

    def test_works_when_function_called_from_other_engine(self):
        class A(bd.Engine):
            val = 2

            @bd.on('foo')
            def f(self):
                return self.val

        a = A()
        b = bd.Engine().register(a)
        b.val = 3

        assert a.val == 2
        assert a.foo() == [2]
        assert a.foo(b) == [3]
        assert b.foo() == [3]
        assert b.foo(a) == [2]

    def test_can_not_register_value_for_which_an_event_exists(self):
        class Foo(bd.Engine):
            @bd.on('foo')
            def f(self):
                pass

        f = Foo()
        with pytest.raises(KeyError) as e:
            f.foo = 100
        assert 'already defined as an event' in str(e)

    def test_inherited_object_variables_are_overwritten(self):
        class Base(bd.Engine):
            out_val = 2

        class Child(Base):
            out_val = 13

        eng = Child()
        assert eng.out_val == 13

    def test_initialized_variables_are_overwritten(self):
        class Base(bd.Engine):
            def __init__(self):
                super().__init__()
                self.in_val = 3

        class Child(Base):
            def __init__(self):
                super().__init__()
                self.in_val = 5

        eng = Child()
        assert eng.in_val == 5

    def test_child_member_functions_override_parents(self):
        class Base(bd.Engine):
            def no_event(self):
                return 5

        class Child(Base):
            def no_event(self):
                return 7

        eng = Child()
        assert eng.no_event() == 7

    # A.k.a events are inherited with the functions
    def test_child_member_functions_without_event_runs_on_parents_event(self):
        class Base(bd.Engine):
            @bd.on('p_event')
            def parent_has_event(self):
                return 5

        class Child(Base):
            def parent_has_event(self):
                return 7

        eng = Child()
        assert eng.parent_has_event() == 7
        assert eng.p_event() == [7]

    def test_child_member_functions_can_call_super(self):
        class Base(bd.Engine):
            @bd.on('event_a')
            def parent_has_event(self):
                return 1

            @bd.on('event_b')
            def both_have_same_event(self):
                return 3

            @bd.on('event_c')
            def both_have_different_event(self):
                return 5

        class Child(Base):
            def parent_has_event(self):
                return 7 * super().parent_has_event()

            @bd.on('event_b')
            def both_have_same_event(self):
                return 11 * super().both_have_same_event()

            @bd.on('event_d')
            def both_have_different_event(self):
                return 13 * super().both_have_different_event()

        eng = Child()
        assert eng.parent_has_event() == 7 * 1
        assert eng.both_have_same_event() == 11 * 3
        assert eng.both_have_different_event() == 13 * 5

        assert eng.event_a() == [7 * 1]
        assert eng.event_b() == [11 * 3]
        assert eng.event_c() == [13 * 5]
        assert eng.event_d() == [13 * 5]

    def test_child_member_functions_with_same_events_are_overridden(self):
        class Base(bd.Engine):
            @bd.on('event_a')
            def same_event_name(self):
                return 5

        class Child(Base):
            @bd.on('event_a')
            def same_event_name(self):
                return 7

        eng = Child()
        assert eng.same_event_name() == 7
        assert eng.event_a() == [7]

    def test_child_member_functions_with_same_name_as_events_work(self):
        class Base(bd.Engine):
            @bd.on('shared_event')
            def shared_event(self):
                return 2

            def child_event(self):
                return 3

            @bd.on('parent_event')
            def parent_event(self):
                return 5

        class Child(Base):
            @bd.on('shared_event')
            def shared_event(self):
                return 7

            @bd.on('child_event')
            def child_event(self):
                return 11

            def parent_event(self):
                return 13

        #  class SuperChild(Base):
        #      @bd.on('shared_event')
        #      def shared_event(self):
        #          return 7 * super().get_no_event('shared_event')()
        #
        #      @bd.on('child_event')
        #      def child_event(self):
        #          return 11 * super().get_no_event('child_event')()
        #
        #      def parent_event(self):
        #          return 13 * super().get_no_event('parent_event')()

        eng = Base()
        assert eng.shared_event() == [2]
        assert eng.child_event() == 3
        assert eng.parent_event() == [5]
        assert eng.event('shared_event') == [2]
        assert eng.event('child_event') is None
        assert eng.event('parent_event') == [5]
        assert eng.get_no_event('shared_event')() == 2
        assert eng.get_no_event('child_event')() == 3
        assert eng.get_no_event('parent_event')() == 5

        eng = Child()
        assert eng.shared_event() == [7]
        assert eng.child_event() == [11]
        assert eng.parent_event() == [13]
        assert eng.event('shared_event') == [7]
        assert eng.event('child_event') == [11]
        assert eng.event('parent_event') == [13]
        assert eng.get_no_event('shared_event')() == 7
        assert eng.get_no_event('child_event')() == 11
        assert eng.get_no_event('parent_event')() == 13

        #  eng = SuperChild()
        #  assert eng.shared_event() == [14]
        #  assert eng.child_event() == [33]
        #  assert eng.parent_event() == [65]
        #  assert eng.event('shared_event') == [14]
        #  assert eng.event('child_event') == [33]
        #  assert eng.event('parent_event') == [65]
        #  assert eng.get_no_event('shared_event')() == 14
        #  assert eng.get_no_event('child_event')() == 33
        #  assert eng.get_no_event('parent_event')() == 65

    def test_creating_an_event_named_the_same_as_a_method_will_also_run_the_method(
        self,
    ):
        class Foo(bd.Engine):
            def func_name(self):
                return 3

        class Bar(Foo):
            @bd.on('func_name')
            def other(self):
                return 5

        b = Bar()

        assert b.func_name() == [3, 5]

    def test_child_member_functions_with_different_events_run_on_both_events(self):
        class Base(bd.Engine):
            @bd.on('p_event')
            def both_have_event(self):
                return 5

        class Child(Base):
            @bd.on('c_event')
            def both_have_event(self):
                return 7

        eng = Child()
        assert eng.both_have_event() == 7
        assert eng.p_event() == [7]
        assert eng.c_event() == [7]

    def test_non_overriden_child_functions_are_inherited(self):
        class Base(bd.Engine):
            @bd.on('p_event')
            def only_parent_defines_this(self):
                return 5

        class Child(Base):
            pass

        eng = Child()
        assert eng.only_parent_defines_this() == 5
        assert eng.p_event() == [5]

    def test_parent_defined_events_are_executing_all_functions_even_from_different_child_functions(
        self,
    ):
        class Base(bd.Engine):
            @bd.on('event_a')
            def this_function_is_from_parent(self):
                return 5

        class Child(Base):
            @bd.on('event_a')
            def this_function_is_from_child(self):
                return 7

        eng = Child()
        assert eng.this_function_is_from_parent() == 5
        assert eng.this_function_is_from_child() == 7
        # NOTE: Order is not guaranteed to be "reasonable"
        assert eng.event_a() == [7, 5]

    def test_can_check_membership(self):
        eng = bd.Engine()
        eng.state = {'a': {'b': 10, 'd': ('f', 'g')}}
        assert 'a' in eng
        assert 'a.b' in eng
        assert 'c' not in eng
        assert 'a.c' not in eng
        assert 'a.d' in eng
        assert 'a.d.f' not in eng

        def f(engine):
            return 100

        eng.register('do', f)
        assert 'f' not in eng
        assert 'do' not in eng

    def test_can_access_method_that_has_an_event_name(self):
        class Foo(bd.Engine):
            @bd.on('foo')
            def foo(self):
                return 5

        f = Foo()

        assert f.foo() == [5]
        assert f.get_no_event('foo')() == 5

    def test_can_use_bd_on_as_function(self):
        class Foo(bd.Engine):
            def __init__(self):
                super().__init__()
                bd.on('foo')(self.foo)

            def foo(self):
                return 10

        f = Foo()
        assert f.event('foo') == [10]
        assert f.foo() == [10]

    def test_can_use_middleware(self):
        def doublermw(engine, callback, **kwargs):
            return 2 * callback(**kwargs)

        class Foo(bd.Engine):
            @bd.middleware(doublermw)
            @bd.on('foo')
            def foo(self):
                return 3

        f = Foo()
        assert f.foo() == [6]

    def test_can_use_middleware_function(self):
        def doublermw(engine, callback, **kwargs):
            return 2 * callback(**kwargs)

        class Foo(bd.Engine):
            def __init__(self):
                super().__init__()
                bd.middleware(doublermw)(self.foo_fn)

            @bd.on('foo')
            def foo_fn(self):
                return 3

        f = Foo()
        assert f.foo() == [6]

        class Bar(bd.Engine):
            @bd.on('bar')
            def bar_fn(self):
                return 5

        b = Bar()
        bd.middleware(doublermw)(b.bar_fn)
        assert b.bar() == [10]

    def test_using_middleware_on_overwritten_event_works(self):
        def doublermw(engine, callback, **kwargs):
            return 2 * callback(**kwargs)

        class Foo(bd.Engine):
            def __init__(self):
                super().__init__()
                bd.middleware(doublermw)(self.foo)

            @bd.on('foo')
            def foo(self):
                return 3

        f = Foo()
        assert f.foo() == [6]

    def test_applying_middleware_on_overwritten_method_works(self):
        def doublermw(engine, callback, **kwargs):
            return 2 * callback(**kwargs)

        class Foo(bd.Engine):
            @bd.on('foo')
            def foo(self):
                return 3

        f = Foo()
        bd.middleware(doublermw)(f.get_no_event('foo'))
        f.foo() == [6]

    def test_middleware_run_in_order(self):
        out = []

        def mw_1(engine, callback, **kwargs):
            out.append(1)
            return callback(**kwargs)

        def mw_2(engine, callback, **kwargs):
            out.append(2)
            return callback(**kwargs)

        class Foo(bd.Engine):
            @bd.middleware(mw_2)
            @bd.middleware(mw_1)
            @bd.on('foo')
            def foo(self):
                return 3

        f = Foo()
        assert f.foo() == [3]
        assert out == [1, 2]

    def test_bd_on_works_without_decorator(self):
        class Foo(bd.Engine):
            def foo(self):
                return 10

            def setup(self):
                bd.on('bonk')(self.foo)

        f = Foo()
        f.setup()
        assert f.bonk() == [10]


class TestCleanup:
    def test_cleanup_works(self):
        eng = bd.Engine()
        eng.state = {'a': 3, 'b': {'c': {'d': 15}, 'e': 1}, 'f': 3}
        with bd.CleanupState(eng):
            eng.state.d = 10
            assert (eng.state.d) == 10
            eng.state.g = {'h': {'j': 13}}
            assert eng.state.g.h.j == 13
            eng.state.b.c.w = {'z': 2}
            assert eng.state.b.c.w.z == 2
            assert 'b.c.w.z' in eng

        assert eng.state.a == 3
        assert eng.state.b.c.d == 15
        assert eng.state.b.e == 1
        assert eng.state.f == 3

        assert 'd' not in eng.state
        assert 'g.h.j' not in eng.state
        assert 'g.h' not in eng.state
        assert 'g' not in eng.state
        assert 'b.c.w' not in eng.state

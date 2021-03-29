import pytest
import boardom as bd


class TestOnce:
    def test_no_use_as_callable(self):
        with pytest.raises(TypeError) as e:

            @bd.once()
            def foo():
                pass

            assert 'missing 1 required positional argument' in str(e)

    def test_no_use_on_class(self):
        with pytest.raises(RuntimeError) as e:

            @bd.once
            class Foo:
                pass

            assert 'expected a method' in str(e)

    def test_method_runs_once(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @bd.once
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        x = f.bar()
        assert f.x == 20
        assert x == 40
        y = f.bar()
        assert f.x == 20
        assert y == 40

    def test_works_on_properties(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @bd.once
            @property
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        x = f.bar
        assert f.x == 20
        assert x == 40
        y = f.bar
        assert f.x == 20
        assert y == 40

    def test_can_be_used_as_property(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @property
            @bd.once
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        x = f.bar
        assert f.x == 20
        assert x == 40
        y = f.bar
        assert f.x == 20
        assert y == 40

    def test_respects_multiple_instances(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @bd.once
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        f_x = f.bar()
        assert f.x == 20
        assert f_x == 40
        f_y = f.bar()
        assert f.x == 20
        assert f_y == 40

        g = Foo(100)
        assert g.x == 100
        g_x = g.bar()
        assert g.x == 200
        assert g_x == 400
        g_x = g.bar()
        assert g.x == 200
        assert g_x == 400
        assert f.x == 20
        assert f.bar() == 40

    def test_respects_multiple_instances_when_applied_on_properties(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @bd.once
            @property
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        f_x = f.bar
        assert f.x == 20
        assert f_x == 40
        f_y = f.bar
        assert f.x == 20
        assert f_y == 40

        g = Foo(100)
        assert g.x == 100
        g_x = g.bar
        assert g.x == 200
        assert g_x == 400
        g_x = g.bar
        assert g.x == 200
        assert g_x == 400
        assert f.x == 20
        assert f.bar == 40

    def test_does_not_respect_multiple_instances_when_becomes_a_property(self):
        # The following should not work (as one would expect):
        # @property
        # @bd.once
        # This is expected. bd.once is applied first, thus the fget function
        # (which is always unbound) will only be executed once, even for different
        # instances

        class Foo:
            def __init__(self, val):
                self.x = val

            @property
            @bd.once
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        f_x = f.bar
        assert f.x == 20
        assert f_x == 40
        f_y = f.bar
        assert f.x == 20
        assert f_y == 40

        g = Foo(100)
        assert g.x == 100
        # g.bar should not be called here, so g.bar stays always 40
        g_x = g.bar

        assert g.x == 100
        assert g_x == 40
        g_x = g.bar
        assert g.x == 100
        assert g_x == 40
        assert f.x == 20
        assert f.bar == 40


class TestOnceProperty:
    def test_works(self):
        class Foo:
            def __init__(self, val):
                self.x = val

            @bd.once_property
            def bar(self):
                self.x = self.x * 2
                return self.x * 2

        f = Foo(10)
        assert f.x == 10
        f_x = f.bar
        assert f.x == 20
        assert f_x == 40
        f_y = f.bar
        assert f.x == 20
        assert f_y == 40

        g = Foo(100)
        assert g.x == 100
        g_x = g.bar
        assert g.x == 200
        assert g_x == 400
        g_x = g.bar
        assert g.x == 200
        assert g_x == 400
        assert f.x == 20

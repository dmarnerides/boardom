import pytest
import torch
from torch import nn
import boardom as bd


def add_one(x):
    return x + 1


class TestApply:
    def test_single_value(self):
        assert bd.apply(add_one)(1) == 2

    def test_list(self):
        assert bd.apply(add_one)([1, 2]) == [2, 3]

    def test_tuple(self):
        assert bd.apply(add_one)((1, 2)) == (2, 3)

    def test_dict(self):
        assert bd.apply(add_one)({'a': 1, 'b': 2}) == {'a': 2, 'b': 3}

    def test_single_with_recurse(self):
        assert bd.apply(add_one, recurse=True)(1) == 2

    def test_list_with_recurse(self):
        assert bd.apply(add_one, recurse=True)([1, 2]) == [2, 3]

    def test_tuple_with_recurse(self):
        assert bd.apply(add_one, recurse=True)((1, 2)) == (2, 3)

    def test_dict_with_recurse(self):
        applier = bd.apply(add_one, recurse=True)
        assert applier({'a': 1, 'b': 2}) == {'a': 2, 'b': 3}

    def test_nested_with_recurse(self):
        a = {'a': [1, 2, (3, 4), {'b': 5, 'c': 6}], 'd': 7, 'e': [8, 9]}
        b = {'a': [2, 3, (4, 5), {'b': 6, 'c': 7}], 'd': 8, 'e': [9, 10]}
        assert bd.apply(add_one, recurse=True)(a) == b


class TestSingleton:
    def test_is_single(self):
        class Foo(metaclass=bd.Singleton):
            pass

        class Bar:
            pass

        assert Foo() is Foo()
        assert Bar() is not Bar()


# TODO: Signal Handler


class TestIdentity:
    def test_no_argument_is_none(self):
        assert bd.identity() is None

    def test_single_argument_is_itself(self):
        a = [1, 2, 3]
        assert bd.identity(a) is a

    def test_multiple_arguments_are_a_tuple(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        ans = bd.identity(a, b)
        assert ans == (a, b)
        assert ans[0] is a
        assert ans[1] is b


class TestNull:
    def test_null_is_false(self):
        assert not bd.Null


class TestNullFunction:
    def test_null_function_returns_none(self):
        assert bd.null_function(1, 2, 3) is None


class TestStr2Bool:
    def test_none_is_false(self):
        assert not bd.str2bool(None)

    def test_valid_false_strings(self):
        for x in [
            'f',
            'F',
            'False',
            'false',
            'fAlSe',
            '0',
            'no',
            'not',
            '',
        ]:
            assert bd.str2bool(x) is False

    def test_valid_true_strings(self):
        for x in ['t', 'T', 'True', 'true', '1', 'bonk', 'foo']:
            assert bd.str2bool(x) is True

    def test_raises_typeerror_for_non_string_or_none(self):
        with pytest.raises(TypeError) as e:
            bd.str2bool(42)
            assert 'str2bool function' in str(e)


class TestStrIsInt:
    def test_raises_typeerror_for_non_string(self):
        for x in [None, 42, {}]:
            with pytest.raises(TypeError) as e:
                bd.str_is_int(x)
                assert 'str_is_int function' in str(e)

    def test_works_for_int_str(self):
        assert bd.str_is_int('42')

    def test_float_is_false(self):
        assert not bd.str_is_int('42.5')

    def test_text_is_false(self):
        assert not bd.str_is_int('foo')

import pytest
import torch
import boardom as bd


class TestState:
    def test_is_mapping(self):
        from collections.abc import Mapping

        assert isinstance(bd.State(), Mapping)

    def test_can_assign_member(self):
        s = bd.State()
        s.a = 10
        assert s['a'] == 10

    def test_can_assign_list(self):
        s = bd.State()
        s.a = [10]
        assert s['a'] == [10]

    def test_can_assign_tuple(self):
        s = bd.State()
        s.a = (10,)
        assert s['a'] == (10,)

    def test_can_assign_dict(self):
        s = bd.State()
        s.a = {'foo': 10}
        assert s['a']['foo'] == 10
        assert isinstance(s['a'], bd.State)

    def test_assigned_nested_dicts_become_state(self):
        s = bd.State()
        s.a = {'foo': {'bar': 10}}
        assert s['a']['foo']['bar'] == 10
        assert isinstance(s['a'], bd.State)
        assert isinstance(s['a']['foo'], bd.State)

    def test_can_assign_tensor(self):
        s = bd.State()
        s.a = torch.Tensor((10,))
        assert s['a'] == torch.Tensor((10,))

    def test_dicts_become_state_type(self):
        s = bd.State()
        s.a = {'foo': 10}
        assert isinstance(s['a'], bd.State)

    def test_can_access_member_as_attribute(self):
        s = bd.State()
        s['a'] = 10
        assert s.a == 10

    def test_can_access_nested_attributes(self):
        s = bd.State()
        s.a = {'foo': 10}
        assert s['a']['foo'] == 10
        assert s['a'].foo == 10
        assert s.a.foo == 10

    def test_can_access_nested_with_dotted_string(self):
        s = bd.State()
        s.a = {'foo': 10}
        assert s['a.foo'] == 10

    def test_can_access_nested_with_sequence(self):
        s = bd.State()
        s['a'] = {'b': 10}
        assert s['a', 'b'] == 10
        assert s[['a', 'b']] == 10
        assert s[('a', 'b')] == 10

    def test_non_sequence_key_raises_error(self):
        s = bd.State()
        s['a'] = {'b': 10}
        with pytest.raises(TypeError):
            s[set(('a', 'b'))]

    def test_can_get_using_get_function(self):
        s = bd.State({'foo': {'bar': 10}, 'baz': 3})
        assert s.get('baz') == 3
        assert s.get('foo.bar') == 10
        assert s.get('qux', None) is None
        assert s.get('foo.qux', 5) == 5

    def test_can_create_from_state(self):
        s = bd.State(bd.State())
        assert s == bd.State()

    def test_can_create_from_state_with_data(self):
        d = bd.State()
        d.a = {'b': 10}
        s = bd.State(bd.State(d))
        assert s.a.b == 10

    def test_can_create_from_dict(self):
        s = bd.State({'a': {'b': 10}})
        assert s.a.b == 10

    def test_can_only_use_valid_identifiers_as_keys(self):
        with pytest.raises(KeyError) as e:
            bd.State({'1a': 10})
        assert 'valid Python identifiers' in str(e)

    def test_can_check_membership(self):
        s = bd.State({'a': {'b': 10, 'd': ('f', 'g')}})
        assert 'a' in s
        assert 'a.b' in s
        assert 'c' not in s
        assert 'a.c' not in s
        assert 'a.d' in s
        assert 'a.d.f' not in s

    def test_empty_state_evaluates_to_false(self):
        s_1 = bd.State()
        s_2 = bd.State({})
        s_3 = bd.State({'foo': 10})
        assert not s_1
        assert not s_2
        assert s_3

    def test_update_works_and_is_recursive(self):
        s_1 = bd.State({'foo': {'baz': {'bonk': 5, 'bonkers': 3}, 'qux': 2}, 'bar': 1})
        s_2 = bd.State({'foo': {'baz': {'bonk': 9}}, 'bar': 7})
        assert s_1.foo.baz.bonk == 5
        assert s_1.foo.baz.bonkers == 3
        assert s_1.foo.qux == 2
        assert s_1.bar == 1
        assert s_2.foo.baz.bonk == 9
        assert s_2.bar == 7
        s_1.update(s_2)
        assert s_1.foo.baz.bonk == 9
        assert s_1.foo.baz.bonkers == 3
        assert s_1.foo.qux == 2
        assert s_1.bar == 7

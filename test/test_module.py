import pytest
import torch
from torch import nn
import boardom as bd


class TestModule:
    def test_can_create_module(self):
        mod = bd.Module()
        assert str(mod) == 'Module()'

    def test_module_registry(self):
        assert bd.models.module._subclass_modules.called
        mod = bd.Module()
        assert mod._registry is not None
        assert mod._registry['sequential'] is bd.Sequential
        assert mod._registry['linear'] is nn.Linear


class TestMagicBuilder:
    def test_can_set_builder_context(self):
        with bd.magic_builder():
            pass

    def test_context_is_set_correctly(self):
        class Foo(bd.Module):
            def __init__(self):
                super().__init__()
                self.member = ['relu']

        member_is_list = Foo()
        with bd.magic_builder():
            member_is_module = Foo()

        member_is_list_again = Foo()
        assert member_is_list.member == ['relu']
        assert isinstance(member_is_module.member, nn.ReLU)
        assert member_is_list_again.member == ['relu']

    def test_context_works_in_init(self):
        class Foo(bd.Module):
            def __init__(self):
                super().__init__()
                self.list_member = ['relu']
                with bd.magic_builder():
                    self.module_member = ['relu']
                self.list_member_2 = ['relu']

        only_one_is_module = Foo()
        with bd.magic_builder():
            all_are_modules = Foo()

        assert only_one_is_module.list_member == ['relu']
        assert isinstance(only_one_is_module.module_member, nn.ReLU)
        assert only_one_is_module.list_member_2 == ['relu']

        assert isinstance(all_are_modules.list_member, nn.ReLU)
        assert isinstance(all_are_modules.module_member, nn.ReLU)
        assert isinstance(all_are_modules.list_member_2, nn.ReLU)

    def test_nested_context_works(self):
        class Foo(bd.Module):
            def __init__(self):
                super().__init__()
                self.list_member = ['relu']
                with bd.magic_builder():
                    self.module_member = ['relu']
                    with bd.magic_builder():
                        self.module_member_2 = ['relu']
                    self.module_member_3 = ['relu']
                self.list_member_2 = ['relu']

        f = Foo()

        assert f.list_member == ['relu']
        assert isinstance(f.module_member, nn.ReLU)
        assert isinstance(f.module_member_2, nn.ReLU)
        assert isinstance(f.module_member_3, nn.ReLU)
        assert f.list_member_2 == ['relu']


class TestMagicModule:
    def test_can_accept_module_and_returns_it(self):
        for mod in [nn.ReLU(), nn.Tanh(), bd.Sequential()]:
            assert bd.magic_module(mod) is mod

    def test_only_accepts_module_tuple_and_list(self):
        for x in [nn.Module(), [], tuple()]:
            bd.magic_module(x)
        for x in [{}, property, str()]:
            with pytest.raises(RuntimeError) as e:
                bd.magic_module(x)
            assert 'must be composed of' in str(e)

    def test_empty_list_or_tuple_returns_id(self):
        for mod in [list(), tuple()]:
            assert isinstance(bd.magic_module(mod), nn.Identity)

    def test_list_or_tuple_first_element_creates_list_or_tuple(self):
        rel = nn.ReLU()
        assert bd.magic_module(['list', rel, 'foo', 3]) == [rel, 'foo', 3]
        assert bd.magic_module(['tuple', rel, 'foo', 3]) == (rel, 'foo', 3)

    def test_first_element_dict_means_sequential(self):
        rel = nn.ReLU()
        mod = bd.magic_module([{'relu': rel}])
        assert isinstance(mod, bd.Sequential)
        assert mod[0] == rel
        assert mod['relu'] == rel
        assert mod.relu == rel

    def test_first_element_dict_with_list_makes_module(self):
        mod = bd.magic_module([{'relu': ['relu']}])
        assert isinstance(mod, bd.Sequential)
        assert isinstance(mod[0], nn.ReLU)
        assert isinstance(mod['relu'], nn.ReLU)
        assert isinstance(mod.relu, nn.ReLU)

    def test_when_dict_can_have_another_dict_element(self):
        bd.magic_module([{}])
        bd.magic_module([{}, {}])
        with pytest.raises(RuntimeError) as e:
            bd.magic_module([{}, {}, {}])
        assert 'Invalid module' in str(e)
        with pytest.raises(RuntimeError) as e:
            bd.magic_module([{}, {}, 42])
        assert 'Invalid module' in str(e)
        with pytest.raises(RuntimeError) as e:
            bd.magic_module([{}, 422])
        assert 'Invalid module' in str(e)

    def test_second_dict_is_for_kwargs_of_sequential(self):
        bd.magic_module([{}, {'enum_tag': ''}])
        with pytest.raises(RuntimeError) as e:
            bd.magic_module([{}, {'foo': 42}])
        assert 'Invalid kwargs' in str(e)

    def test_two_dict_form_works(self):
        mod_1 = bd.magic_module([{'relu': ['relu']}, {'enum_tag': ''}])
        assert isinstance(mod_1.relu, nn.ReLU)
        res_1 = mod_1(torch.tensor([-2.0]))
        assert torch.allclose(res_1, torch.tensor([0.0]))

        mod_2 = bd.magic_module([{'relu': ['relu']}, {'enum_tag': 'foo_'}])
        assert isinstance(mod_2.foo_relu, nn.ReLU)
        res_2 = mod_2(torch.tensor([-2.0]))
        assert torch.allclose(res_2, torch.tensor([0.0]))

    def test_single_element_returns_element(self):
        assert isinstance(bd.magic_module(['relu']), nn.ReLU)

    def test_nested_single_element_is_sequential(self):
        magic = bd.magic_module([['relu']])
        assert isinstance(magic, bd.Sequential)
        assert isinstance(magic[0], nn.ReLU)

    def test_doubly_nested_single_element_is_single_sequential(self):
        magic = bd.magic_module([[['relu']]])
        assert isinstance(magic, bd.Sequential)
        assert isinstance(magic[0], nn.ReLU)

    def test_two_elements_without_string_return_sequential(self):
        module = bd.magic_module([['relu'], ['tanh']])
        assert isinstance(module, bd.Sequential)
        assert len(module) == 2
        assert isinstance(module[0], nn.ReLU)
        assert isinstance(module[1], nn.Tanh)

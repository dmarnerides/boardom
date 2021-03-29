import boardom as bd
import numpy as np
import torch


def mul_tensors(x):
    return 5 * x if isinstance(x, torch.Tensor) else x


#  def state_sample(num=1, prep='', just_stuff=False, level=1, max_level=2):
#      if just_stuff:
#          return {
#              f'{prep}num': num,
#              f'{prep}str': 'foo',
#              f'{prep}ten': torch.tensor([num]),
#              f'{prep}arr': np.array([num]),
#              f'{prep}lin': torch.nn.Linear(num, num),
#              f'{prep}mod': bd.Module(),
#          }
#      else:
#          ret = {
#              **state_sample(num, prep, True),
#              f'{prep}list': list(state_sample(2 * num, '', True).values()),
#              f'{prep}tuple': tuple(state_sample(3 * num, '', True).values()),
#          }
#          if level <= max_level:
#              key = f'lvl{level}'
#              ret.update({key: state_sample(5 * num, '', False, level + 1)})
#          return ret
#
#
#  def test_sample_sanity():
#      eng = bd.StateUtils()
#      eng.update(state_sample(max_level=2))
#      assert eng.num == 1
#      assert eng.list[0] == 2
#      assert eng.tuple[0] == 3
#      assert eng.lvl1.num == 5
#      assert eng.lvl1.lvl2.num == 25


class TestStateUtils:
    def test_can_create_testutils(self):
        bd.StateUtils()

    def test_can_get_members(self):
        x = bd.StateUtils()
        x.foo = 1
        x.bar = {'a': 2, 'b': 3}
        x.baz = {'c': {'d': 4, 'e': 5}}

        sequence = {
            'foo': 1,
            'bar': x.bar,
            'baz': x.baz,
        }

        for ((k, v), (key, val)) in zip(sequence.items(), x.named_members()):
            assert v == val
            assert k == key

    def test_can_get_members_starting_from(self):
        x = bd.StateUtils()
        x.foo = 1
        x.bar = {'a': 2, 'b': 3}
        x.baz = {'c': {'d': 4, 'e': 5, 'f': {'g': 6, 'h': 7}}}

        baz_sequence = {
            'baz.c.d': 4,
            'baz.c.e': 5,
            'baz.c.f': x.baz.c.f,
        }

        for ((k, v), (key, val)) in zip(
            baz_sequence.items(), x.named_members(start_from='baz.c')
        ):
            assert v == val
            assert k == key

    def test_can_get_members_recursive(self):
        x = bd.StateUtils()
        x.foo = 1
        x.bar = {'a': 2, 'b': 3}
        x.baz = {'c': {'d': 4, 'e': 5}}

        recurse_sequence = {
            'foo': 1,
            'bar': x.bar,
            'bar.a': 2,
            'bar.b': 3,
            'baz': x.baz,
            'baz.c': x.baz.c,
            'baz.c.d': 4,
            'baz.c.e': 5,
        }

        for ((k, v), (key, val)) in zip(
            recurse_sequence.items(), x.named_members(recurse=True)
        ):
            assert v == val
            assert k == key

    def test_can_get_members_recursive_starting_from(self):
        x = bd.StateUtils()
        x.foo = 1
        x.bar = {'a': 2, 'b': 3}
        x.baz = {'c': {'d': 4, 'e': 5, 'f': {'g': 6, 'h': 7}}}

        baz_sequence = {
            'baz.c.d': 4,
            'baz.c.e': 5,
            'baz.c.f': x.baz.c.f,
            'baz.c.f.g': 6,
            'baz.c.f.h': 7,
        }

        for ((k, v), (key, val)) in zip(
            baz_sequence.items(), x.named_members(start_from='baz.c', recurse=True)
        ):
            assert v == val
            assert k == key

    def test_can_apply(self):
        eng = bd.StateUtils()
        eng.state = {
            'a': torch.tensor([2.0]),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0]),
                'e': {'f': torch.tensor([7.0]), 'g': 11},
                'h': 13,
            },
        }
        eng.apply(mul_tensors, recurse=False)
        assert torch.allclose(eng.a, 5 * torch.tensor([2.0]))
        assert eng.b == 3.0
        assert torch.allclose(eng.c.d, torch.tensor([5.0]))
        assert torch.allclose(eng.c.e.f, torch.tensor([7.0]))
        assert eng.c.e.g == 11
        assert eng.c.h == 13.0

    def test_can_recurse_apply(self):
        eng = bd.StateUtils()
        eng.state = {
            'a': torch.tensor([2.0]),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0]),
                'e': {'f': torch.tensor([7.0]), 'g': 11},
                'h': 13,
            },
        }
        eng.apply(mul_tensors, recurse=True)
        assert torch.allclose(eng.a, 5 * torch.tensor([2.0]))
        assert eng.b == 3.0
        assert torch.allclose(eng.c.d, 5 * torch.tensor([5.0]))
        assert torch.allclose(eng.c.e.f, 5 * torch.tensor([7.0]))
        assert eng.c.e.g == 11
        assert eng.c.h == 13.0

    #  def test_can_recurse_apply_fully(self):
    #      eng = bd.StateUtils()
    #      eng.state = {
    #          'a': torch.tensor([2.0]),
    #          'b': 3.0,
    #          'c': {
    #              'd': torch.tensor([5.0]),
    #              'e': {'f': torch.tensor([7.0]), 'g': 11},
    #              'h': 13,
    #          },
    #          'i': [
    #              17,
    #              19,
    #              torch.tensor([23]),
    #              {'j': torch.tensor([29]), 'k': (4, torch.tensor([31]))},
    #          ],
    #          'l': {'m': {'n': {'o': [37, {'p': torch.tensor([39])}]}}},
    #      }
    #      eng.apply(mul_tensors, recurse=True)
    #      assert torch.allclose(eng.a, 5 * torch.tensor([2.0]))
    #      assert eng.b == 3.0
    #      assert torch.allclose(eng.c.d, 5 * torch.tensor([5.0]))
    #      assert torch.allclose(eng.c.e.f, 5 * torch.tensor([7.0]))
    #      assert eng.c.e.g == 11
    #      assert eng.c.h == 13.0
    #      assert eng.i[0] == 17.0
    #      assert eng.i[1] == 19.0
    #      assert torch.allclose(eng.i[2], 5 * torch.tensor([23]))
    #      # Nested dicts in lists and tuples do not become State
    #      assert torch.allclose(eng.i[3]['j'], 5 * torch.tensor([29]))
    #      assert eng.i[3]['k'][0] == 4
    #      assert torch.allclose(eng.i[3]['k'][1], 5 * torch.tensor([31]))
    #      assert eng.l.m.n.o[0] == 37
    #      assert torch.allclose(eng.l.m.n.o[1]['p'], 5 * torch.tensor([39]))

    def test_train_and_eval_works(self):
        eng = bd.StateUtils()
        eng.state = {
            'a': torch.nn.Linear(2, 2),
            'b': 3.0,
            'c': {
                'd': torch.nn.Linear(3, 3),
                'e': {
                    'f': torch.nn.Linear(7, 7),
                    'g': 11,
                    'j': {'k': torch.nn.Linear(17, 17)},
                },
                'h': 13,
            },
            'i': torch.nn.Linear(10, 10),
        }
        assert eng.a.training
        assert eng.c.d.training
        assert eng.c.e.f.training
        assert eng.c.e.j.k.training
        assert eng.i.training
        eng.eval()
        assert not eng.a.training
        assert not eng.c.d.training
        assert not eng.c.e.f.training
        assert not eng.c.e.j.k.training
        assert not eng.i.training
        eng.train()
        assert eng.a.training
        assert eng.c.d.training
        assert eng.c.e.f.training
        assert eng.c.e.j.k.training
        assert eng.i.training

import boardom as bd
from torch import nn


class TestSequential:
    def test_can_accept_no_element(self):
        bd.Sequential()

    def test_single_element(self):
        coll = bd.Sequential(nn.ReLU())
        assert len(coll) == 1
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(getattr(coll, '0'), nn.ReLU)

    def test_single_element_list(self):
        coll = bd.Sequential([nn.ReLU()])
        assert len(coll) == 1
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(getattr(coll, '0/0'), nn.ReLU)

    def test_single_element_tuple(self):
        coll = bd.Sequential((nn.ReLU(),))
        assert len(coll) == 1
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(getattr(coll, '0/0'), nn.ReLU)

    def test_single_element_dict(self):
        coll = bd.Sequential({'relu': nn.ReLU()})
        assert len(coll) == 1
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(getattr(coll, 'relu'), nn.ReLU)

    def test_single_element_cfg(self):
        coll = bd.Sequential(['relu'])
        assert len(coll) == 1
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(getattr(coll, '0'), nn.ReLU)

    #  def test_multi_element_cfg(self):
    #      coll = bd.Sequential(['relu'], ['tanh'])
    #      assert len(coll) == 2
    #      assert isinstance(coll[0], nn.ReLU)
    #      assert isinstance(coll[1], nn.Tanh)
    #      assert isinstance(getattr(coll, '0'), nn.ReLU)
    #      assert isinstance(getattr(coll, '1'), nn.Tanh)

    def test_multi_element_dict(self):
        coll = bd.Sequential({'relu': nn.ReLU(), 'tanh': nn.Tanh()})
        assert len(coll) == 2
        assert isinstance(coll[0], nn.ReLU)
        assert isinstance(coll[1], nn.Tanh)
        assert isinstance(getattr(coll, 'relu'), nn.ReLU)
        assert isinstance(getattr(coll, 'tanh'), nn.Tanh)

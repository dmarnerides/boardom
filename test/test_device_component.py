import boardom as bd
import torch


class TestDevice:
    def test_can_create_device_component(self):
        bd.Device()

    def test_to_works(self):
        eng = bd.Device()
        cpu_dev = torch.device('cpu')
        eng.state = {
            'a': torch.tensor([2.0], device=cpu_dev),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0], device=cpu_dev),
                'e': {'f': torch.tensor([7.0], device=cpu_dev), 'g': 11},
                'h': 13,
            },
            'i': torch.nn.Linear(10, 10),
            'j': [torch.nn.Linear(12, 12)],
        }
        assert eng.i.weight.device == cpu_dev
        assert eng.j[0].weight.device == cpu_dev
        dev = torch.device('cuda', index=0)
        eng.to(dev)
        assert eng.a.device == dev
        assert eng.c.d.device == dev
        assert eng.c.e.f.device == dev
        assert eng.i.weight.device == dev
        assert eng.j[0].weight.device == dev

    def test_to_no_recurse_works(self):
        eng = bd.Device()
        cpu_dev = torch.device('cpu')
        eng.state = {
            'a': torch.tensor([2.0], device=cpu_dev),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0], device=cpu_dev),
                'e': {'f': torch.tensor([7.0], device=cpu_dev), 'g': 11},
                'h': 13,
            },
            'i': torch.nn.Linear(10, 10),
        }
        dev = torch.device('cuda', index=0)
        eng.to(dev, recurse=False)
        assert eng.a.device == dev
        assert eng.c.d.device == cpu_dev
        assert eng.c.e.f.device == cpu_dev

    def test_to_works_starting_from_member(self):
        eng = bd.Device()
        cpu_dev = torch.device('cpu')
        eng.state = {
            'a': torch.tensor([2.0], device=cpu_dev),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0], device=cpu_dev),
                'e': {
                    'f': torch.tensor([7.0], device=cpu_dev),
                    'g': 11,
                    'j': {'k': torch.tensor([17], device=cpu_dev)},
                },
                'h': 13,
            },
            'i': torch.nn.Linear(10, 10),
        }
        dev = torch.device('cuda', index=0)
        eng.to(dev, start_from='c.e')
        assert eng.a.device == cpu_dev
        assert eng.c.d.device == cpu_dev
        assert eng.c.e.f.device == dev
        assert eng.c.e.j.k.device == dev

    def test_to_works_starting_from_member_no_recurse(self):
        eng = bd.Device()
        cpu_dev = torch.device('cpu')
        eng.state = {
            'a': torch.tensor([2.0], device=cpu_dev),
            'b': 3.0,
            'c': {
                'd': torch.tensor([5.0], device=cpu_dev),
                'e': {
                    'f': torch.tensor([7.0], device=cpu_dev),
                    'g': 11,
                    'j': {'k': torch.tensor([17], device=cpu_dev)},
                },
                'h': 13,
            },
            'i': torch.nn.Linear(10, 10),
        }
        dev = torch.device('cuda', index=0)
        eng.to(dev, start_from='c.e', recurse=False)
        assert eng.a.device == cpu_dev
        assert eng.c.d.device == cpu_dev
        assert eng.c.e.f.device == dev
        assert eng.c.e.j.k.device == cpu_dev

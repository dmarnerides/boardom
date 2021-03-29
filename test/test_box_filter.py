import pytest
import torch
import boardom as bd


class TestBoxFilter:
    def test_boxfilter1d_works(self):

        odd = torch.Tensor([1, 2, 3, 2, 1])
        assert torch.equal(bd.box_filter1d(odd, k=1), odd)
        assert torch.equal(bd.box_filter1d(odd, k=2), torch.Tensor([3, 5, 5, 3, 1]))
        assert torch.equal(bd.box_filter1d(odd, k=3), torch.Tensor([3, 6, 7, 6, 3]))
        assert torch.equal(bd.box_filter1d(odd, k=4), torch.Tensor([6, 8, 8, 6, 3]))
        assert torch.equal(bd.box_filter1d(odd, k=5), torch.Tensor([6, 8, 9, 8, 6]))
        with pytest.raises(RuntimeError):
            bd.box_filter1d(odd, k=6)
        with pytest.raises(RuntimeError):
            bd.box_filter1d(odd, k=0)

        even = torch.Tensor([1, 2, 3, 2])
        assert torch.equal(bd.box_filter1d(even, k=1), even)
        assert torch.equal(bd.box_filter1d(even, k=2), torch.Tensor([3, 5, 5, 2]))
        assert torch.equal(bd.box_filter1d(even, k=3), torch.Tensor([3, 6, 7, 5]))
        assert torch.equal(bd.box_filter1d(even, k=4), torch.Tensor([6, 8, 7, 5]))

        # Check a 2d tensor input example
        even_2d = torch.Tensor([1, 2, 3, 2]).view(1, 4)
        assert torch.equal(bd.box_filter1d(even_2d, k=1), even_2d)
        assert torch.equal(
            bd.box_filter1d(even_2d, k=2), torch.Tensor([3, 5, 5, 2]).view(1, 4)
        )
        assert torch.equal(
            bd.box_filter1d(even_2d, k=3), torch.Tensor([3, 6, 7, 5]).view(1, 4)
        )
        assert torch.equal(
            bd.box_filter1d(even_2d, k=4), torch.Tensor([6, 8, 7, 5]).view(1, 4)
        )

        even_2d2 = torch.Tensor([1, 2, 3, 2]).view(4, 1)
        assert torch.equal(bd.box_filter1d(even_2d2, k=1), even_2d2)
        assert torch.equal(
            bd.box_filter1d(even_2d2, k=2, dim=0), torch.Tensor([3, 5, 5, 2]).view(4, 1)
        )
        assert torch.equal(
            bd.box_filter1d(even_2d2, k=3, dim=0), torch.Tensor([3, 6, 7, 5]).view(4, 1)
        )
        assert torch.equal(
            bd.box_filter1d(even_2d2, k=4, dim=0), torch.Tensor([6, 8, 7, 5]).view(4, 1)
        )

    def test_box_filter_2d_works(self):
        a = torch.ones(3, 3)
        assert torch.equal(
            bd.box_filter2d(a, (2, 2)),
            torch.Tensor([[4, 4, 2], [4, 4, 2], [2, 2, 1]]),
        )

    def test_box_filter_nd_works(self):
        a = torch.ones(3, 3, 3)
        assert torch.equal(
            bd.box_filternd(a, (2, 2, 2), dims=(-1, -2, -3)),
            torch.Tensor(
                [
                    [[8, 8, 4], [8, 8, 4], [4, 4, 2]],
                    [[8, 8, 4], [8, 8, 4], [4, 4, 2]],
                    [[4, 4, 2], [4, 4, 2], [2, 2, 1]],
                ]
            ),
        )

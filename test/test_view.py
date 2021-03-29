import boardom as bd
from importlib import reload


class TestView:
    def test_bd_default_view_is_torch(self):
        reload(bd)

        assert bd.default_view == 'torch'

    def test_view_context_manager_works(self):
        reload(bd)

        assert bd.default_view == 'torch'
        with bd.view('cv'):
            assert bd.default_view == 'opencv'
        assert bd.default_view == 'torch'

    def test_view_decorator_works(self):
        reload(bd)

        @bd.view('cv')
        def foo():
            return bd.default_view

        assert bd.default_view == 'torch'
        assert foo() == 'opencv'
        assert bd.default_view == 'torch'

import pytest
import boardom as bd
import numpy as np

#
#  class TestResize:
#      def test_can_fully_resize(self):
#          h, w, c = (50, 100, 3)
#          x = np.ones((h, w, c)).astype('float32')
#          square_1 = bd.resize_keep_ratio(x, (50, 50))
#          bd.imshow(square_1)
#          assert square_1.shape == (50, 50, 3)
#          assert square_1[:, :25, :].sum() == 0

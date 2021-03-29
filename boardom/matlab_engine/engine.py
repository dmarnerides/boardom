import time
import os
import tempfile
import scipy.io
import numpy as np
import matlab.engine
import boardom as bd


class Matlab:
    def __init__(self, *paths, recurse=False):
        bd.log('Launching Matlab Engine...')
        main_path = bd.main_file_path()
        start = time.time()
        self.engine = matlab.engine.start_matlab(f'-sd {main_path}')
        end = time.time()
        bd.log(f'Matlab launch done. Time taken: {end-start:.2f}s.')
        # Add the current path to access .m functions defined here
        self.add_path(os.path.dirname(os.path.abspath(__file__)), recurse=True)
        self.add_path(*paths, recurse=recurse)

    def _add_path(self, path):
        path = bd.process_path(path)
        self.engine.addpath(self.engine.genpath(path))

    def add_path(self, *paths, recurse=False):
        if recurse:
            paths = [x[0] for path in paths for x in os.walk(path)]
        for path in paths:
            self._add_path(path)

    def mat2np(self, x):
        return np.array(x._data).reshape(x.size, order='F')

    def mat2cv(self, x):
        return bd.rgb2bgr(self.mat2np(x), dim=-1)

    # Converting to list is too slow, so saving to temporary file
    def np2mat(self, x, var_name=None):
        fname = tempfile.NamedTemporaryFile().name
        fname = f'{fname}.mat'
        if var_name is None:
            var_name = 'np_img'
        scipy.io.savemat(
            fname,
            {var_name: x},
            appendmat=False,
            format='5',  # pylint: disable=E1101
            long_field_names=False,
            do_compression=False,
            oned_as='row',
        )
        return fname, var_name

    def cv2mat(self, x):
        return self.np2mat(bd.bgr2rgb(x, dim=-1))

    def __getattr__(self, attr):
        return getattr(self.engine, attr)

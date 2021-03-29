# matlab.engine needs to be imported before everything (before torch.utils.data.DataLoader)
import sys
import importlib


_matlab = importlib.util.find_spec('matlab')
if _matlab is not None:
    _engine = importlib.util.find_spec('matlab.engine')
    if _engine is not None:
        if 'matplotlib.pyplot' not in sys.modules:
            from .engine import Matlab
        else:
            print(
                '[boardom - warning] Not importing matlab engine. as matplotlib.pyplot '
                'is already imported. To use matlab engine import boardom first.'
            )
            Matlab = None
    else:
        Matlab = None
else:
    Matlab = None

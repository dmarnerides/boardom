import boardom as bd
from wrapt import decorator
from contextlib import contextmanager
from .tensor_ops import permute, channel_flip, to_array, to_tensor, is_array

default_view = 'torch'

# Getting images from one library to the other
# Always assuming the last three dimensions are the images
# opencv is hwc - BGR
# torch is chw - RGB
# plt is hwc - RGB
VIEW_NAMES = {
    'opencv': [
        'hwcbgr',
        'hwc-bgr',
        'bgrhwc',
        'bgr-hwc',
        'opencv',
        'open-cv',
        'cv',
        'cv2',
    ],
    'torch': ['chwrgb', 'chw-rgb', 'rgbchw', 'rgb-chw', 'torch', 'pytorch'],
    'plt': ['hwcrgb', 'hwc-rgb', 'rgbhwc', 'rgb-hwc', 'plt', 'pyplot', 'matplotlib'],
    'other': ['chwbgr', 'chw-bgr', 'bgrchw', 'bgr-chw'],
}


def determine_view(v):
    if not isinstance(v, str):
        raise ValueError(f'Invalid view {v}')
    v_low = v.lower()
    for view, names in VIEW_NAMES.items():
        if v_low in names:
            return view
    raise ValueError('Could not determine {v} view.')


class view:
    def __init__(self, new_view):
        self.new_view = determine_view(new_view)

    def __enter__(self):
        self.old_view = bd.default_view
        bd.default_view = self.new_view

    def __exit__(self, type, value, traceback):
        bd.default_view = self.old_view

    @decorator
    def __call__(self, wrapped, instance, args, kwargs):
        with self:
            return wrapped(*args, **kwargs)


def hwc2chw(x):
    """Permutes the last three dimensions of the hwc input to become chw.

    Args:
        x (Tensor or Array): Input to be permuted.
    """
    return permute(x, (2, 0, 1))


def chw2hwc(x):
    """Permutes the last three dimensions of the chw input to become hwc.

    Args:
        x (Tensor or Array): Input to be permuted.
    """
    return permute(x, (1, 2, 0))


# Default is dimension -3 (e.g. for bchw)
def rgb2bgr(x, dim=-3):
    """Reverses the channel dimension. See :func:`channel_flip`"""
    return channel_flip(x, dim)


def bgr2rgb(x, dim=-3):
    """Reverses the channel dimension. See :func:`channel_flip`"""
    return channel_flip(x, dim)


# This is not elegant but at least it's clear and does its job
def change_view(x, current, new):
    """Changes the view of the input. Returns a copy.

    Args:
        x (Tensor or Array): Input whose view is to be changed.
        current (str): Current view.
        new (str): New view.

    Possible views:

    ======== ==============================================================
      View     Aliases
    ======== ==============================================================
     opencv   hwcbgr, hwc-bgr, bgrhwc, bgr-hwc, opencv, open-cv, cv, cv2
    -------- --------------------------------------------------------------
     torch    chwrgb, chw-rgb, rgbchw, rgb-chw, torch, pytorch
    -------- --------------------------------------------------------------
     plt      hwcrgb, hwc-rgb, rgbhwc, rgb-hwc, plt, pyplot, matplotlib
    -------- --------------------------------------------------------------
     other    chwbgr, chw-bgr, bgrchw, bgr-chw
    ======== ==============================================================

    Note:
        If the input has less than three dimensions a copy is returned.

    """
    curr_name, new_name = determine_view(current), determine_view(new)
    if new_name == curr_name:
        if is_array(x):
            return x.copy()
        else:
            return x.data.clone()

    if curr_name == 'opencv':
        if new_name == 'torch':
            return bgr2rgb(hwc2chw(x), -3)
        elif new_name == 'plt':
            return bgr2rgb(x, -1)
        elif new_name == 'other':
            return hwc2chw(x)
    if curr_name == 'torch':
        if new_name == 'opencv':
            return chw2hwc(rgb2bgr(x, -3))
        elif new_name == 'plt':
            return chw2hwc(x)
        elif new_name == 'other':
            return rgb2bgr(x, -3)
    if curr_name == 'plt':
        if new_name == 'torch':
            return hwc2chw(x)
        elif new_name == 'opencv':
            return rgb2bgr(x, -1)
        elif new_name == 'other':
            return hwc2chw(rgb2bgr(x, -1))
    if curr_name == 'other':
        if new_name == 'torch':
            return bgr2rgb(x, -3)
        elif new_name == 'plt':
            return chw2hwc(rgb2bgr(x, -3))
        elif new_name == 'opencv':
            return chw2hwc(x)


# These functions also convert!
def cv2torch(x):
    """Converts input to Tensor and changes view from cv (hwc-bgr) to torch (chw-rgb).

    For more detail see :func:`change_view`
    """
    return change_view(to_tensor(x), 'cv', 'torch')


def torch2cv(x):
    """Converts input to Array and changes view from torch (chw-rgb) to cv (hwc-bgr).

    For more detail see :func:`change_view`
    """
    return change_view(to_array(x), 'torch', 'cv')


def cv2plt(x):
    """Changes view from cv (hwc-bgr) to plt (hwc-rgb).

    For more detail see :func:`change_view`
    """
    return change_view(x, 'cv', 'plt')


def plt2cv(x):
    """Changes view from plt (hwc-rgb) to cv (hwc-bgr).

    For more detail see :func:`change_view`
    """
    return change_view(x, 'plt', 'cv')


def plt2torch(x):
    """Converts input to Tensor and changes view from plt (hwc-rgb) to torch (chw-rgb).

    For more detail see :func:`change_view`
    """
    return change_view(to_tensor(x), 'plt', 'torch')


def torch2plt(x):
    """Converts input to Array and changes view from torch (chw-rgb) to plt (hwc-rgb) .

    For more detail see :func:`change_view`
    """
    return change_view(to_array(x), 'torch', 'plt')

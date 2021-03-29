import numpy as np
import cv2

# Returns a resized img with size size keeping the aspect ratio
# by padding appropriately if pad is True, else, it will use either
# the height or width (only one must be provided, e.g. size=(256, None))
# Assumes a numpy array of shape h w c
# Size is (w,h)
def resize_keep_ratio(img, size, pad_val=0, interpolation=cv2.INTER_AREA):
    if not isinstance(size, (list, tuple)):
        raise RuntimeError(f'Expected size to be list or tuple. Got: {type(size)}')
    if len(size) != 2:
        raise RuntimeError(f'Size must have two elements. Got len(size)={len(size)}')
    h, w, c = img.shape
    ar = w / h
    end_w, end_h = size
    if end_w is None:
        end_w = int(end_h * ar)
    elif end_h is None:
        end_h = int(end_w / ar)
    end_ar = end_w / end_h
    result = np.zeros((end_h, end_w, c)).astype(img.dtype)
    if ar > end_ar:
        res_dim = (end_w, int(end_w / ar))
        start = max(0, int((end_h - res_dim[-1]) / 2))
        resized = cv2.resize(img, res_dim, interpolation=interpolation)

        result[start : start + resized.shape[0], :, :] = resized
    else:
        res_dim = (int(end_h * ar), end_h)
        start = max(0, int((end_w - res_dim[0]) / 2))
        resized = cv2.resize(img, res_dim, interpolation=interpolation)

        result[:, start : start + resized.shape[1], :] = resized
    return result

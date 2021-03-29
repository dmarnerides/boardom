import torch

# RGB is assumed to be in BT.709 format with D65 whitepoint, similarly to OpenCV
# In conversions, images are assumed linear, without gamma and in the range [0,1]
# if the function is prepended with batch_ it expects 4 dimensional tensors [b, c, h, w]
# Otherwise it's [c,h,w]
# Gray / single channel images preserve their channels dimension. I.e. [1,h,w]


# These are from opencv
mat_rgb2xyz = torch.Tensor(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
mat_xyz2rgb = torch.Tensor(
    [
        [3.240479, -1.53715, -0.498535],
        [-0.969256, 1.875991, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ]
)

# Matlab RGB2GRAY uses these values:
# 0.2989 * R + 0.5870 * G + 0.1140 * B

# Y channel of XYZ
def luminance(img):
    return (img * mat_rgb2xyz[1][:, None, None].to(img.device)).sum(-3, keepdim=True)


# Y channel of XYZ
def batch_luminance(img):
    return (img * mat_rgb2xyz[1][None, :, None, None].to(img.device)).sum(
        -3, keepdim=True
    )


# img is torch chw rgb
def rgb2xyz(img):
    c, h, w = img.shape
    return torch.matmul(mat_rgb2xyz, img.view(c, h * w)).view(c, h, w)


def batch_rgb2xyz(img):
    b, c, h, w = img.shape
    img = img.permute(1, 0, 2, 3).contiguous().view(c, b * h * w)
    return (
        torch.matmul(mat_rgb2xyz.to(img.device), img)
        .view(c, b, h, w)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def xyz2rgb(img):
    c, h, w = img.shape
    return torch.matmul(mat_xyz2rgb, img.view(c, h * w)).view(c, h, w).clamp(0, 1)


def batch_xyz2rgb(img):
    b, c, h, w = img.shape
    img = img.permute(1, 0, 2, 3).contiguous().view(c, b * h * w)
    return (
        torch.matmul(mat_xyz2rgb.to(img.device), img)
        .view(c, b, h, w)
        .permute(1, 0, 2, 3)
        .contiguous()
        .clamp(0, 1)
    )


_thresh = 0.008856


def _f(t):
    larger_slice = t > _thresh
    smaller_slice = larger_slice.logical_not()
    result = t.clone()
    result[larger_slice] = t[larger_slice].pow(1 / 3)
    result[smaller_slice] = t[smaller_slice] * 7.787 + 16 / 116
    return result


def _f_L(t):
    larger_slice = t > _thresh
    smaller_slice = larger_slice.logical_not()
    result = t.clone()
    result[larger_slice] = 116 * (t[larger_slice].pow(1 / 3)) - 16
    result[smaller_slice] = t[smaller_slice] * 903.3
    return result


X_n, Z_n = 0.950456, 1.088754


# It's here
# https://github.com/opencv/opencv/blob/43467a2ac77207afd7bbc348e63d89692f838ad6/modules/imgproc/src/color_lab.cpp#L1100
def xyz2lab(img):
    x, y, z = img[0:1, :, :] / X_n, img[1:2, :, :], img[2:3, :, :] / Z_n
    #  x, y, z = img[0:1,:,:], img[1:2,:,:], img[2:3,:,:]
    L = _f_L(y)
    f_y = _f(y)
    a = 500 * (_f(x) - f_y)
    b = 200 * (f_y - _f(z))
    return torch.cat((L, a, b), 0)


def batch_xyz2lab(img):
    x, y, z = img[:, 0:1, :, :] / X_n, img[:, 1:2, :, :], img[:, 2:3, :, :] / Z_n
    #  x, y, z = img[0:1,:,:], img[1:2,:,:], img[2:3,:,:]
    L = _f_L(y)
    f_y = _f(y)
    a = 500 * (_f(x) - f_y)
    b = 200 * (f_y - _f(z))
    return torch.cat((L, a, b), -3)


def _inv_f(t):
    larger_slice = t > _thresh
    smaller_slice = larger_slice.logical_not()
    result = t.clone()
    result[larger_slice] = t[larger_slice].pow(3)
    result[smaller_slice] = (t[smaller_slice] - 16 / 116) / 7.787
    return result


def _inv_f_L(t):
    larger_slice = t > _thresh
    smaller_slice = larger_slice.logical_not()
    result = t.clone()
    result[larger_slice] = ((t[larger_slice] + 16) / 116).pow(3)
    result[smaller_slice] = t[smaller_slice] / 903.3
    return result


def lab2xyz(img):
    L, a, b = img[0:1, :, :], img[1:2, :, :], img[2:3, :, :]
    y = _inv_f_L(L)
    f_y = _f(y)
    x = X_n * _inv_f(f_y + a / 500)
    z = Z_n * _inv_f(f_y - b / 200)
    return torch.cat((x, y, z), 0)


def batch_lab2xyz(img):
    L, a, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    y = _inv_f_L(L)
    f_y = _f(y)
    #  inv_y = inf_f(L)
    x = X_n * _inv_f(f_y + a / 500)
    z = Z_n * _inv_f(f_y - b / 200)
    return torch.cat((x, y, z), -3)


def lab2rgb(img):
    return xyz2rgb(lab2xyz(img))


def batch_lab2rgb(img):
    return batch_xyz2rgb(batch_lab2xyz(img))


def rgb2lab(img):
    return xyz2lab(rgb2xyz(img))


def batch_rgb2lab(img):
    return batch_xyz2lab(batch_rgb2xyz(img))

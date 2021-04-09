import math
import torch
import numpy as np
import pandas as pd
import boardom as bd


def slide_window_(a, kernel, stride=None):
    """Expands last dimension to help compute sliding windows.

    Args:
        a (Tensor or Array): The Tensor or Array to view as a sliding window.
        kernel (int): The size of the sliding window.
        stride (tuple or int, optional): Strides for viewing the expanded dimension (default 1)

    The new dimension is added at the end of the Tensor or Array.

    Returns:
        The expanded Tensor or Array.

    Running Sum Example::

        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_slided = boardom.util.slide_window_(a.clone(), kernel=3, stride=1)
         1  2  3
         2  3  4
         3  4  5
         4  5  6
        [torch.FloatTensor of size 4x3]
        >>> running_total = (a_slided*torch.Tensor([1,1,1])).sum(-1)
          6
          9
         12
         15
        [torch.FloatTensor of size 4]

    Averaging Example::

        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_sub_slide = boardom.util.slide_window_(a.clone(), kernel=3, stride=3)
         1  2  3
         4  5  6
        [torch.FloatTensor of size 2x3]
        >>> a_sub_avg = (a_sub_slide*torch.Tensor([1,1,1])).sum(-1) / 3.0
         2
         5
        [torch.FloatTensor of size 2]
    """

    if isinstance(kernel, int):
        kernel = (kernel,)
    if stride is None:
        stride = tuple(1 for i in kernel)
    elif isinstance(stride, int):
        stride = (stride,)
    window_dim = len(kernel)
    if is_array(a):
        new_shape = (
            a.shape[:-window_dim]
            + tuple(
                int(np.floor((s - kernel[i]) / stride[i]) + 1)
                for i, s in enumerate(a.shape[-window_dim:])
            )
            + kernel
        )
        new_stride = (
            a.strides[:-window_dim]
            + tuple(s * k for s, k in zip(a.strides[-window_dim:], stride))
            + a.strides[-window_dim:]
        )
        return np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=new_stride)
    else:
        new_shape = (
            a.size()[:-window_dim]
            + tuple(
                int(np.floor((s - kernel[i]) / stride[i]) + 1)
                for i, s in enumerate(a.size()[-window_dim:])
            )
            + kernel
        )
        new_stride = (
            a.stride()[:-window_dim]
            + tuple(s * k for s, k in zip(a.stride()[-window_dim:], stride))
            + a.stride()[-window_dim:]
        )
        a.set_(a.storage(), storage_offset=0, size=new_shape, stride=new_stride)
        return a


def re_stride(a, kernel, stride=None):
    """Returns a re-shaped and re-strided Rensor given a kernel (uses as_strided).

    Args:
        a (Tensor): The Tensor to re-stride.
        kernel (tuple or int): The size of the new dimension(s).
        stride (tuple or int, optional): Strides for viewing the expanded dimension(s) (default 1)
    """
    if isinstance(kernel, int):
        kernel = (kernel,)
    if stride is None:
        stride = tuple(1 for i in kernel)
    elif isinstance(stride, int):
        stride = (stride,)
    window_dim = len(kernel)
    new_shape = (
        a.size()[:-window_dim]
        + kernel
        + tuple(
            int(math.floor((s - kernel[i]) / stride[i]) + 1)
            for i, s in enumerate(a.size()[-window_dim:])
        )
    )
    new_stride = (
        a.stride()[:-window_dim]
        + a.stride()[-window_dim:]
        + tuple(s * k for s, k in zip(a.stride()[-window_dim:], stride))
    )
    return a.as_strided(new_shape, new_stride)


def clamped_gaussian(mean, std, min_value, max_value):
    if max_value <= min_value:
        return mean
    factor = 0.99
    while True:
        ret = np.random.normal(mean, std)
        if ret > min_value and ret < max_value:
            break
        else:
            std = std * factor
            ret = np.random.normal(mean, std)

    return ret


def exponential_size(val):
    return val * (np.exp(-np.random.uniform())) / (np.exp(0) + 1)


# Accepts hwc-bgr image
def index_gauss(
    img,
    precision=None,
    crop_size=None,
    random_size=True,
    ratio=None,
    seed=None,
):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a gaussian distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        precision (list or tuple, optional): Floats representing the precision
            of the Gaussians (default [1, 4])
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:

                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {"w": img.shape[1], "h": img.shape[0]}
    if precision is None:
        precision = {"w": 1, "h": 4}
    else:
        precision = {"w": precision[0], "h": precision[1]}

    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    else:
        crop_size = {"w": crop_size[0], "h": crop_size[1]}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(max(crop_size['h'], exponential_size(dims['h'])))
            crop_size['w'] = int(np.round(crop_size['h'] * ratio))
        else:
            if random_size:
                crop_size['w'] = int(max(crop_size['w'], exponential_size(dims['w'])))
            crop_size['h'] = int(np.round(crop_size['w'] / ratio))
    else:
        if random_size:
            crop_size = {
                key: int(max(val, exponential_size(dims[key])))
                for key, val in crop_size.items()
            }

    centers = {
        key: int(
            clamped_gaussian(
                dim / 2,
                crop_size[key] / precision[key],
                min(int(crop_size[key] / 2), dim),
                max(int(dim - crop_size[key] / 2), 0),
            )
        )
        for key, dim in dims.items()
    }
    starts = {
        key: max(center - int(crop_size[key] / 2), 0) for key, center in centers.items()
    }
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts["h"] : ends["h"], starts["w"] : ends["w"], :]


def slice_gauss(
    img,
    precision=None,
    crop_size=None,
    random_size=True,
    ratio=None,
    seed=None,
):
    """Returns a cropped sample from an image array using :func:`index_gauss`"""
    return img[index_gauss(img, precision, crop_size, random_size, ratio)]


# Accepts hwc-bgr image
def index_uniform(img, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a uniform distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:

                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {"w": img.shape[1], "h": img.shape[0]}
    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    else:
        crop_size = {"w": crop_size[0], "h": crop_size[1]}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(max(crop_size['h'], exponential_size(dims['h'])))
            crop_size['w'] = int(np.round(crop_size['h'] * ratio))
        else:
            if random_size:
                crop_size['w'] = int(max(crop_size['w'], exponential_size(dims['w'])))
            crop_size['h'] = int(np.round(crop_size['w'] / ratio))
    else:
        if random_size:
            crop_size = {
                key: int(max(val, exponential_size(dims[key])))
                for key, val in crop_size.items()
            }

    centers = {
        key: int(
            np.random.uniform(
                int(crop_size[key] / 2), int(dims[key] - crop_size[key] / 2)
            )
        )
        for key, dim in dims.items()
    }
    starts = {
        key: max(center - int(crop_size[key] / 2), 0) for key, center in centers.items()
    }
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts["h"] : ends["h"], starts["w"] : ends["w"], :]


def slice_uniform(img, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns a cropped sample from an image array using :func:`index_uniform`"""

    return img[index_uniform(img, crop_size, random_size, ratio)]


def replicate(x, dim=-3, nrep=3):
    """Replicates Tensor/Array in a new dimension.

    Args:
        x (Tensor or Array): Tensor to replicate.
        dim (int, optional): New dimension where replication happens.
        nrep (int, optional): Number of replications.
    """
    if is_tensor(x):
        return x.unsqueeze(dim).expand(*x.size()[: dim + 1], nrep, *x.size()[dim + 1 :])
    else:
        return np.repeat(np.expand_dims(x, dim), nrep, axis=dim)


def moving_avg(x, width=5):
    """Performes moving average of a one dimensional Tensor or Array

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.mean(slide_window_(x, width, 1), -1)
        else:
            return torch.mean(slide_window_(x, width, 1), -1)
    else:
        return x.mean()


def moving_var(x, width=5):
    """Performes moving variance of a one dimensional Tensor or Array

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.var(slide_window_(x, width, 1), -1)
        else:
            return torch.var(slide_window_(x, width, 1), -1)
    else:
        return x.var()


def sub_avg(x, width=5):
    """Performs averaging of a one dimensional Tensor or Array every `width` elements.

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.mean(slide_window_(x, width, width), -1)
        else:
            return torch.mean(slide_window_(x, width, width), -1)
    else:
        return x.mean()


def sub_var(x, width=5):
    """Calculates variance of a one dimensional Tensor or Array every `width` elements.

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.var(slide_window_(x, width, width), -1)
        else:
            return torch.var(slide_window_(x, width, width), -1)
    else:
        return x.var()


def has(x, val):
    """Checks if a Tensor/Array has a value val. Does not work with nan (use :func:`has_nan`)."""
    return bool((x == val).sum() != 0)


def has_nan(x):
    """Checks if a Tensor/Array has NaNs."""
    return bool((x != x).sum() > 0)


def has_inf(x):
    """Checks if a Tensor/Array array has Infs."""
    return has(x, float('inf'))


def replace_specials_(x, val=0):
    """Replaces NaNs and Infs from a Tensor/Array.

    Args:
        x (Tensor or Array): The Tensor/Array (gets replaced in place).
        val (int, optional): Value to replace NaNs and Infs with (default 0).
    """
    x[(x == float('inf')) | (x != x)] = val
    return x


def replace_inf_(x, val=0):
    """Replaces Infs from a Numpy Array.

    Args:
        x (Array): The Array (gets replaced in place).
        val (int, optional): Value to replace Infs with (default 0).
    """
    x[x == float('inf')] = val
    return x


def replace_nan_(x, val=0):
    """Replaces NaNs from a Numpy Array.

    Args:
        x (Array): The Array (gets replaced in place).
        val (int, optional): Value to replace Infs with (default 0).
    """
    x[x != x] = val
    return x


def map_range(x, low=0, high=1):
    """Maps the range of a Numpy Array to [low, high] globally."""
    if is_array(x):
        return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)
    else:
        xmax, xmin = x.max(), x.min()
        if xmax - xmin == 0:
            return torch.zeros_like(x)
        return (
            x.add(-xmin).div_(xmax - xmin).mul_(high - low).add_(low).clamp_(low, high)
        )


# This was added to torch in v0.3. Keeping it here too.
def is_tensor(x):
    """Checks if input is a Tensor"""
    return torch.is_tensor(x)


def is_cuda(x):
    """Checks if input is a cuda Tensor."""
    return torch.is_tensor(x) and x.is_cuda


def is_array(x):
    """Checks if input is a numpy array or a pandas Series."""
    return isinstance(x, np.ndarray) or isinstance(x, pd.Series)


# Returns a numpy array version of x
def to_array(x):
    """Converts x to a Numpy Array. Returns a copy of the data.

    Args:
        x (Tensor or Array): Input to be converted. Can also be on the GPU.

    Automatically gets the data from torch Tensors and casts GPU Tensors
    to CPU.
    """
    if is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.copy()


# Returns a cpu tensor COPY version of x
def to_tensor(x):
    """Converts x to a Torch Tensor. Returns a copy of the data if x is a Tensor.

    Args:
        x (Tensor or Array): Input to be converted. Can also be on the GPU.

    Automatically casts GPU Tensors to CPU.
    """
    # if is_cuda(x):
    #     return x.cpu()
    if is_array(x):
        return torch.from_numpy(x)
    else:
        return x.clone()


########
# Tensors, arrays, cuda, cpu, image views etc
########
def permute(x, perm):
    """Permutes the last three dimensions of the input Tensor or Array.

    Args:
        x (Tensor or Array): Input to be permuted.
        perm (tuple or list): Permutation.

    Note:
        If the input has fewer than three dimensions a copy is returned.
    """
    if is_tensor(x):
        if x.dim() < 3:
            return x.data.clone()
        else:
            s = tuple(range(0, x.dim()))
            permutation = s[:-3] + tuple(s[-3:][i] for i in perm)
        return x.permute(*permutation).contiguous()
    elif is_array(x):
        if x.ndim < 3:
            return x.copy()
        else:
            s = tuple(range(0, x.ndim))
            permutation = s[:-3] + tuple(s[-3:][i] for i in perm)
        # Copying to get rid of negative strides
        return np.transpose(x, permutation).copy()
    else:
        raise TypeError(f'Uknown type {torch.typename(x)} encountered.')


# TODO: CHANGE DEFAULT DIMENSION ACCORDING TO DEFAULT VIEW
def channel_flip(x, dim=-3):
    """Reverses the channel dimension.

    Args:
        x (Tensor or Array): Input to have its channels flipped.
        dim (int, optional): Channels dimension (default -3).

    Note:
        If the input has fewer than three dimensions a copy is returned.
    """

    if is_tensor(x):
        # ADAPT TO THIS
        # return torch.flip(x, dim)
        dim = x.dim() + dim if dim < 0 else dim
        if x.dim() < 3:
            return x.data.clone()
        return x[
            tuple(
                slice(None, None)
                if i != dim
                else torch.arange(x.size(i) - 1, -1, -1).long()
                for i in range(x.dim())
            )
        ]
    elif is_array(x):
        dim = x.ndim + dim if dim < 0 else dim
        if x.ndim < 3:
            return x.copy()
        return np.ascontiguousarray(np.flip(x, dim))
    else:
        raise TypeError('Uknown type {0} encountered.'.format(torch.typename(x)))


def _make_tensor_4d_with_3_channels(x, orig_view):
    x = bd.to_tensor(x)
    x = bd.change_view(x, orig_view, 'torch')
    ndim = x.ndimension()
    if ndim == 1:
        raise RuntimeError('Received an image with less than 2 dimensions')
    elif ndim == 2:
        x = x.view(1, 1, *x.shape)
    elif ndim == 3:
        x = x.unsqueeze(0)
    x = x.view(-1, *x.shape[-3:])
    b, c, h, w = x.shape
    # If image is not color, assume greyscale
    if c != 3:
        x = x.view(b * c, 1, h, w).expand(b * c, 3, h, w)
    return x


def make_grid(
    images,
    view=None,
    size=None,
    inter_pad=0,
    fill_value=0,
    scale_each=False,
):
    """Creates a single image grid from a set of images.

    Args:
        images (Tensor, Array, list or tuple): Torch Tensor(s) and/or Numpy Array(s).
        view (str, optional): The image view e.g. 'hwc-bgr' or 'torch'
            (default 'torch').
        color (bool, optional): Treat images as colored or not (default True).
        size (list or tuple, optional): Grid dimensions, rows x columns. (default None).
        inter_pad (int or list/tuple, optional): Padding separating the images (default None).
        fill_value (int, optional): Fill value for inter-padding (default 0).
        scale_each (bool, optional): Scale each image to [0-1] (default False).

    Returns:
        Tensor or Array: The resulting grid. If any of the inputs is an Array
        then the result is an Array, otherwise a Tensor.

    Notes:
        - Images of **different sizes are padded** to match the largest.
        - Works for **color** (3 channels) or **grey** (1 channel/0 channel)
          images.
        - Images must have the same view (e.g. chw-rgb (torch))
        - The Tensors/Arrays can be of **any dimension >= 2**. The last 2 (grey)
          or last 3 (color) dimensions are the images and all other dimensions
          are stacked. E.g. a 4x5x3x256x256 (torch view) input will be treated:

            - As 20 3x256x256 color images if color is True.
            - As 60 256x256 grey images if color is False.

        - If color is False, then only the last two channels are considered
          (as hw) thus any colored images will be split into their channels.
        - The image list can contain both **Torch Tensors and Numpy Arrays**.
          at the same time as long as they have the same view.
        - If size is not given, the resulting grid will be the smallest square
          in which all the images fit. If the images are more than the given
          size then the default smallest square is used.

    Raises:
        TypeError: If images are not Arrays, Tensors, a list or a tuple
        ValueError: If channels or dimensions are wrong.

    """

    view = view or bd.default_view
    # Determine view
    orig_view = bd.determine_view(view)

    # Flag if we need to convert back to array
    should_convert_to_array = False

    images = [x for x in bd.recurse_get_elements(images)]
    images = [x for x in images if bd.is_array(x) or bd.is_tensor(x)]
    should_convert_to_array = any([bd.is_array(x) for x in images])
    images = [_make_tensor_4d_with_3_channels(x, orig_view) for x in images]
    if not images:
        return None
    # Pad images to match largest
    if len(images) > 1:
        maxh, maxw = (
            max([x.size(-2) for x in images]),
            max([x.size(-1) for x in images]),
        )
        for i, img in enumerate(images):
            imgh, imgw = img.size(-2), img.size(-1)
            if (img.size(-2) < maxh) or (img.size(-1) < maxw):
                padhl = int((maxh - imgh) / 2)
                padhr = maxh - imgh - padhl
                padwl = int((maxw - imgw) / 2)
                padwr = maxw - imgw - padwl
                images[i] = torch.nn.functional.pad(
                    img, (padwl, padwr, padhl, padhr), value=fill_value
                )
    images = torch.cat(images, 0)

    # Scale each
    if scale_each:
        for i in range(images.size(0)):
            images[i] = bd.map_range(images[i])

    # Create grid
    b, c, im_w, im_h = (
        images.size()[0],
        images.size()[1],
        images.size()[2],
        images.size()[3],
    )
    # Get number of columns and rows (width and height)
    if (size is not None and b > size[0] * size[1]) or size is None:
        n_row = int(np.ceil(np.sqrt(b)))
        n_col = int(np.ceil(b / n_row))
    else:
        n_col = size[0]
        n_row = size[1]

    if isinstance(inter_pad, int):
        inter_pad = (inter_pad, inter_pad)

    w_pad, h_pad = inter_pad[1], inter_pad[0]
    total_w_padding = max(w_pad, 0) * (n_col - 1)
    total_h_padding = max(h_pad, 0) * (n_row - 1)

    w = int(im_w * n_col) + total_w_padding
    h = int(im_h * n_row) + total_h_padding
    grid = torch.Tensor(c, w, h).type_as(images).fill_(fill_value)
    for i in range(b):
        i_row = i % n_row
        i_col = int(i / n_row)
        grid[
            :,
            i_col * (im_w + w_pad) : (i_col) * (im_w + w_pad) + im_w,
            i_row * (im_h + h_pad) : (i_row) * (im_h + h_pad) + im_h,
        ].copy_(images[i])

    if should_convert_to_array:
        grid = bd.to_array(grid)
    if orig_view != 'torch':
        grid = bd.change_view(grid, 'torch', orig_view)
    return grid

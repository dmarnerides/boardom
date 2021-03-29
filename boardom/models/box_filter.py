import torch

# By default this zero pads and returns same size
def box_filter1d(x, k, dim=-1):
    size = x.shape[dim]
    if not (1 <= k <= size):
        raise RuntimeError(
            f'Box filter kernel size must be between 1 and {size}, but got {k}'
        )
    dim = dim % x.ndim
    x = x.cumsum(dim=dim)
    half_k = k // 2
    slice_list = [
        (half_k, k),
        (k, None),
        (None, -k),
        (-1, None),
        (-k, -half_k - (k % 2)),
    ]
    sl = [
        [slice(None) if i != dim else slice(*curr_slice) for i in range(x.ndim)]
        for curr_slice in slice_list
    ]
    return torch.cat((x[sl[0]], x[sl[1]] - x[sl[2]], x[sl[3]] - x[sl[4]]), dim=dim)


# kernel size corresponds to dims.
# so if *_, h, w = x.shape
# and dims=(-1,-2)
# then kernel_size = (k_w, k_h)
def box_filter2d(x, kernel_size, dims=(-1, -2)):
    return box_filternd(x, kernel_size, dims)


def box_filternd(x, kernel_size, dims):
    for k, dim in zip(kernel_size[-1::-1], dims[-1::-1]):
        x = box_filter1d(x, k, dim)
    return x

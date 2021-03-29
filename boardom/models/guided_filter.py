from functools import partial
import torch
from torch.nn import functional as F
from .module import Module
from .box_filter import box_filter2d

# If kernel_size = 0 then it's dynamically adjusted to be equal to the input size
class GuidedFilter(Module):
    def __init__(
        self,
        epsilon=1e-3,
        kernel_size=0,
        division_epsilon=1e-8,
        channel_adapter=None,
        grouped=False,
    ):
        super().__init__()
        epsilon = max(epsilon, division_epsilon)
        self.register_buffer('epsilon', torch.Tensor([epsilon]))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if any(k < 0 for k in kernel_size):
            raise ValueError(f'Invalid kernel size: {kernel_size}.')
        self.base_kernel_size = kernel_size
        self.channel_adapter = channel_adapter
        self.grouped = grouped

    def get_kernel_size(self, x):
        *_, h, w = x.shape
        kernel_size = tuple(
            d if k <= 0 or k >= d else k for k, d in zip(self.base_kernel_size, (h, w))
        )
        return kernel_size

    def forward(self, x):
        guide_hr, guide, target = x

        kernel_size = self.get_kernel_size(target)
        if self.channel_adapter is not None:
            guide = self.channel_adapter(guide)
            target = self.channel_adapter(target)

        norm = torch.ones((1, 1, *target.shape[-2:]), device=target.device)
        norm = box_filter2d(norm, kernel_size)
        target_mean = box_filter2d(target, kernel_size) / norm
        guide_mean = box_filter2d(guide, kernel_size) / norm
        covariance = box_filter2d(target * guide, kernel_size) / norm
        covariance = covariance - target_mean * guide_mean
        guide_variance = box_filter2d(guide.pow(2), kernel_size) / norm
        guide_variance = guide_variance - guide_mean.pow(2)

        A = covariance / (guide_variance + self.epsilon)
        b = target_mean - A * guide_mean

        guide = guide if guide_hr is None else guide_hr
        ups = partial(
            F.interpolate, size=guide.shape[-2:], mode='bilinear', align_corners=False
        )
        A, b = ups(A), ups(b)
        # In case num channels of guide are different from A, b,
        # we assume grouped multiplication
        if self.grouped:
            bs, c, h, w = guide.shape
            A, b = A.view(bs, c, -1, h, w), b.view(bs, c, -1, h, w)
            guide = guide.view(bs, c, 1, h, w)
            return (A * guide + b).view(bs, -1, h, w)
        else:
            return A * guide + b

    def extra_repr(self):
        ret = f'GIF - epsilon: {self.epsilon.item():.2e}, '
        ret += f'kernel_size={self.base_kernel_size}'
        if self.grouped:
            ret += ', grouped'
        return ret


# Size must be (w, h) as is standard in opencv
def _resize(x, size):
    *b, c, h, w = x.shape
    h_new, w_new = size
    if not b:
        x = x.unsqueeze(0)
    else:
        x = x.view(-1, c, h, w)
    # F.interpolate size is reversed..
    ret = F.interpolate(
        x, size=(size[1], size[0]), mode='bilinear', align_corners=False
    )
    return ret.view(*b, c, h_new, w_new)


# TODO: Improve covariance estimation to avoid catastrophic cancellation!!
#       (note) this is alleviated by the epsilon value
# TODO: Adjust the epsilon value to be channel dependent! (e.g. epsilon=(1e-3, 1, 1e-2))
# Note this does guided filter with outer product!
# x is *b, c_x, h, w
# guide is *b, c_g, h, w
# kernel_size is a tuple
# resize is (w,h)
def guided_filter(x, guide=None, kernel_size=(5, 5), epsilon=1e-3, resize=None):
    h_orig, w_orig = x.shape[-2:]
    orig_size = (w_orig, h_orig)
    if resize is not None:
        x = _resize(x, size=resize)

    *b, c_x, h, w = x.shape

    if guide is None:
        guide_orig = x
        guide = x
    else:
        guide_orig = guide
        if resize is not None:
            guide = _resize(guide_orig, size=resize)

    *_, c_g, _, _ = guide.shape

    norm = torch.ones((*([1] * len(b)), 1, h, w), device=x.device)
    norm = box_filter2d(norm, kernel_size)
    x_mean = box_filter2d(x, kernel_size) / norm
    guide_mean = box_filter2d(guide, kernel_size) / norm

    # Compute covariance matrix (in channel dimension)
    x_view = x.view(*b, 1, c_x, h, w)
    x_mean_view = x_mean.view(*b, 1, c_x, h, w)
    guide_view = guide.view(*b, c_g, 1, h, w)
    guide_mean_view = guide_mean.view(*b, c_g, 1, h, w)
    covariance = box_filter2d(x_view * guide_view, kernel_size) / norm
    covariance = covariance - x_mean_view * guide_mean_view
    bdims = list(range(len(b)))
    covariance = covariance.permute(*bdims, -2, -1, -4, -3)
    covariance = covariance.view(-1, c_g, c_x)

    guide_other_view = guide.view(*b, 1, c_g, h, w)
    guide_mean_other_view = guide_mean.view(*b, 1, c_g, h, w)

    guide_covariance = box_filter2d(guide_other_view * guide_view, kernel_size) / norm
    guide_covariance = guide_covariance - guide_mean_other_view * guide_mean_view
    eye = torch.eye(c_g, c_g, device=x.device).view(*b, c_g, c_g, 1, 1)
    guide_covariance = guide_covariance + eye * epsilon
    guide_covariance = guide_covariance.permute(*bdims, -2, -1, -4, -3).view(
        -1, c_g, c_g
    )

    A = torch.bmm(guide_covariance.inverse(), covariance)
    A_transpose = A.transpose(-1, -2)

    a_times_g_mean = torch.bmm(
        A_transpose, guide_mean.permute(*bdims, -2, -1, -3).view(-1, c_g, 1)
    )

    a_times_g_mean = (
        a_times_g_mean[..., 0].view(*b, h, w, c_x).permute(*bdims, -1, -3, -2)
    )

    b_term = x_mean - a_times_g_mean

    if resize is not None:
        dim_0, mat_dim_1, mat_dim_2 = A_transpose.shape
        A_transpose = A_transpose.reshape(*b, h * w, -1).transpose(-1, -2)
        A_transpose = A_transpose.view(-1, mat_dim_1 * mat_dim_2, h, w)
        A_transpose = F.interpolate(
            A_transpose, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        A_transpose = A_transpose.permute(0, -2, -1, -3).view(-1, mat_dim_1, mat_dim_2)

        b_term = _resize(b_term, size=orig_size)

    a_times_guide = torch.bmm(
        A_transpose, guide_orig.permute(*bdims, -2, -1, -3).view(-1, c_g, 1)
    )
    a_times_guide = (
        a_times_guide[..., 0].view(*b, h_orig, w_orig, c_x).permute(*bdims, -1, -3, -2)
    )

    return a_times_guide + b_term


# x is low resolution, same as guide_lr
def multi_guided_filter(x, guide_hr, guide_lr=None, kernel_size=(5, 5), epsilon=1e-3):
    *_, c_g, h_orig, w_orig = guide_hr.shape
    *b, c_x, h, w = x.shape
    orig_size = (w_orig, h_orig)

    if guide_lr is None:
        guide_lr = _resize(guide_hr, (w, h))

    norm = torch.ones((*([1] * len(b)), 1, h, w), device=x.device)
    norm = box_filter2d(norm, kernel_size)
    x_mean = box_filter2d(x, kernel_size) / norm
    guide_mean = box_filter2d(guide_lr, kernel_size) / norm

    # Compute covariance matrix (in channel dimension)
    x_view = x.view(*b, 1, c_x, h, w)
    x_mean_view = x_mean.view(*b, 1, c_x, h, w)
    guide_view = guide_lr.view(*b, c_g, 1, h, w)
    guide_mean_view = guide_mean.view(*b, c_g, 1, h, w)
    covariance = box_filter2d(x_view * guide_view, kernel_size) / norm
    covariance = covariance - x_mean_view * guide_mean_view
    bdims = list(range(len(b)))
    covariance = covariance.permute(*bdims, -2, -1, -4, -3)
    covariance = covariance.view(-1, c_g, c_x)

    guide_other_view = guide_lr.view(*b, 1, c_g, h, w)
    guide_mean_other_view = guide_mean.view(*b, 1, c_g, h, w)

    guide_covariance = box_filter2d(guide_other_view * guide_view, kernel_size) / norm
    guide_covariance = guide_covariance - guide_mean_other_view * guide_mean_view
    eye = torch.eye(c_g, c_g, device=x.device).view(*b, c_g, c_g, 1, 1)
    guide_covariance = guide_covariance + eye * epsilon
    guide_covariance = guide_covariance.permute(*bdims, -2, -1, -4, -3).view(
        -1, c_g, c_g
    )

    A = torch.bmm(guide_covariance.inverse(), covariance)
    A_transpose = A.transpose(-1, -2)

    a_times_g_mean = torch.bmm(
        A_transpose, guide_mean.permute(*bdims, -2, -1, -3).view(-1, c_g, 1)
    )

    a_times_g_mean = (
        a_times_g_mean[..., 0].view(*b, h, w, c_x).permute(*bdims, -1, -3, -2)
    )

    b_term = x_mean - a_times_g_mean

    dim_0, mat_dim_1, mat_dim_2 = A_transpose.shape
    A_transpose = A_transpose.reshape(*b, h * w, -1).transpose(-1, -2)
    A_transpose = A_transpose.view(-1, mat_dim_1 * mat_dim_2, h, w)
    A_transpose = F.interpolate(
        A_transpose, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )
    A_transpose = A_transpose.permute(0, -2, -1, -3).view(-1, mat_dim_1, mat_dim_2)

    b_term = _resize(b_term, size=orig_size)

    a_times_guide = torch.bmm(
        A_transpose, guide_hr.permute(*bdims, -2, -1, -3).view(-1, c_g, 1)
    )
    a_times_guide = (
        a_times_guide[..., 0].view(*b, h_orig, w_orig, c_x).permute(*bdims, -1, -3, -2)
    )

    return a_times_guide + b_term

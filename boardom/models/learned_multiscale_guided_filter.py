from functools import partial
import torch
from torch.nn import functional as F
from .guided_filter import box_filter2d
from .module import Module, magic_off


# If kernel_sizes = [0] then it's dynamically adjusted to be equal to the input size
# Epsilon is a learned parameter
class LearnedMultiScaleGuidedFilter(Module):
    def __init__(
        self,
        kernel_sizes=[0, 0, 0],
        epsilon=1e-3,
        channel_adapter=None,
        grouped=False,
        small_eps=1e-5,
        learn_epsilon=True,
    ):
        super().__init__()
        with magic_off():
            self.kernel_sizes = []
        for kernel_size in kernel_sizes:
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if any(k < 0 for k in kernel_size):
                raise ValueError(f'Invalid kernel size: {kernel_size}.')
            if any(k != 0 for k in kernel_size):
                raise NotImplementedError('Fixed kernels not implemented!')
            self.kernel_sizes.append(kernel_size)

        num_scales = len(kernel_sizes)
        if num_scales <= 0:
            raise RuntimeError('Invalid kernels')
        self.num_scales = num_scales
        epsilon = max(epsilon, small_eps)
        eps_param = torch.ones(num_scales).float() * epsilon
        if learn_epsilon:
            self.register_parameter('log_epsilon', torch.nn.Parameter(eps_param.log()))
        else:
            self.register_buffer('log_epsilon', eps_param.log())
        weight_param = torch.ones(num_scales).float()
        self.register_parameter('weights', torch.nn.Parameter(weight_param))
        self.channel_adapter = channel_adapter
        self.grouped = grouped
        self.learn_epsilon = learn_epsilon
        self.log_small_eps = torch.Tensor([small_eps]).log().item()

    def get_kernel_sizes(self, x):
        *_, h, w = x.shape
        ns = self.num_scales
        kernel_sizes = [(min(h, 3), min(w, 3))]
        if ns > 1:
            dh, dw = abs(h - 3) // (ns - 1), abs(w - 3) // (ns - 1)
            kernel_sizes += [
                (min(h, 3 + i * dh), min(w, 3 + i * dw)) for i in range(1, ns - 1)
            ]
            kernel_sizes += [(h, w)]

        return kernel_sizes

    def forward(self, x):
        guide_hr, guide, target = x

        kernel_sizes = self.get_kernel_sizes(target)
        if self.channel_adapter is not None:
            guide = self.channel_adapter(guide)
            target = self.channel_adapter(target)

        norm = torch.ones((1, 1, *target.shape[-2:]), device=target.device)
        norms = torch.stack([box_filter2d(norm, k) for k in kernel_sizes], 0)
        target_means = (
            torch.stack([box_filter2d(target, k) for k in kernel_sizes], 0) / norms
        )
        guide_means = (
            torch.stack([box_filter2d(guide, k) for k in kernel_sizes], 0) / norms
        )

        covariances = (
            torch.stack([box_filter2d(target * guide, k) for k in kernel_sizes], 0)
            / norms
        )
        covariances = covariances - target_means * guide_means
        guide_variances = (
            torch.stack([box_filter2d(guide.pow(2), k) for k in kernel_sizes], 0)
            / norms
        )
        guide_variances = guide_variances - guide_means.pow(2)
        # Set a minimum limit for small eps
        with torch.no_grad():
            self.log_epsilon.clamp_(self.log_small_eps, 100)
        eps = self.log_epsilon.exp().view(
            self.num_scales, *[1] * (len(guide_variances.shape) - 1)
        )
        A = covariances / (guide_variances + eps)
        b = target_means - A * guide_means

        weights = torch.softmax(self.weights, 0)
        weights = weights.view(self.num_scales, *[1] * (len(A.shape) - 1))
        A = (A * weights).sum(0, keepdim=False)
        b = (b * weights).sum(0, keepdim=False)

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
        ret = 'LMS-GIF '
        ret += f'kernel_sizes={self.kernel_sizes}'
        if self.grouped:
            ret += ', grouped'
        if self.learn_epsilon:
            ret += ', learned epsilon'
        else:
            ret += ', fixed epsilon'
        return ret

from .module import Module, magic_module, magic_builder, magic_off

from .collections import (
    BaseCollection,
    Sequential,
    Split,
    Branch,
    Sum,
    Cat,
    Prod,
    CropCat2d,
    SingleSelect,
    MultiSelect,
    Parallel,
    Map,
)
from .padding import (
    Pad2dSame,
    pad2dsame,
    Pad2dMultiple,
    pad2dmultiple,
    UnPad2dMultiple,
    unpad2dmultiple,
)

from .resize import Resize

from .conv import conv, convsame, conv2dsame, Conv2dSame, ConvTranspose2dSame

from .self_attention import SelfAttention

from .gaussian import gaussian_kernel_nd, gaussian_blur2d, GaussianBlur2d

from .psnr import psnr, PSNR

from .frozen_batchnorm import (
    freeze_bn_running_stats,
    unfreeze_bn_running_stats,
)

from .box_filter import box_filter1d, box_filter2d, box_filternd

from .guided_filter import GuidedFilter, guided_filter, multi_guided_filter
from .learned_multiscale_guided_filter import LearnedMultiScaleGuidedFilter

from .patch_repr import pretty_print

from .resnet import PretrainedResnet, pretrained_resnet_layers, determine_channel_sizes

from .unet import OriginalUNet

from .init_filters import (
    module_is_conv,
    module_is_linear,
    module_is_norm,
    param_is_bias,
    param_is_initialized,
    param_is_not_none,
    param_is_not_frozen,
    param_is_weight,
)

from .init import (
    Initializer,
    InitUniform,
    InitNormal,
    InitConstant,
    InitOnes,
    InitZeros,
    InitEye,
    InitDirac,
    InitXavierUniform,
    InitXavierNormal,
    InitKaimingUniform,
    InitKaimingNormal,
    InitOrthogonal,
    InitSparse,
    InitSELU,
)

from .ssim import SSIM, SSIMLoss


from .cosine_loss import CosineLoss

from .multiloss import MultiLoss

from .gram import GramMatrix, GramLoss

from .total_variation import (
    IsotropicTotalVariation,
    AnisotropicTotalVariation,
    TotalVariation,
)

from .perceptual_loss import StyleLoss, FeatureLoss, PerceptualLoss, FeatureMatchingLoss

from .interp1d import PiecewiseMLP, Interp1d

from .utils import (
    is_frozen,
    is_trainable,
    freeze,
    frozen_parameters,
    named_frozen_parameters,
    frozen_modules,
    named_frozen_modules,
    trainable_parameters,
    named_trainable_parameters,
    count_parameters,
)

from .spectral_norm import spectral_norm, remove_spectral_norm

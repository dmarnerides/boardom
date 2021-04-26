import torch
from torch import nn
import boardom as bd
from torchvision.models import vgg
from .gram import GramMatrix
from .module import Module
from .collections import Sequential

# This is from Gaty's et al and Johnson et al.
# Also known as style loss


class PretrainedVggFeatures(Sequential):
    # extract_list is a list of integers (starting from 0) indicating the VGG layers to extract
    def __init__(self, size=16, bn=False, extract_list=None):
        bn = '_bn' if bn else ''
        version = f'vgg{size}{bn}'
        features = getattr(vgg, version)(pretrained=True).features
        if extract_list is not None:
            if not isinstance(extract_list, list):
                raise RuntimeError('Expected list type for extract_list')
            max_extract = max(extract_list)
            if max_extract >= len(features):
                raise RuntimeError('Exceeded maximum feature size in extract_list')
            features = features[: max_extract + 1]
        else:
            extract_list = list(range(len(features)))
        super().__init__(*features)
        with bd.magic_off():
            self._extract_set = set(extract_list)

    def forward(self, x):
        ret = {}
        for i, feat in enumerate(self):
            x = feat(x)
            if i in self._extract_set:
                ret[i] = x
        return ret

    def normalise_01_input(self, x):
        with torch.no_grad():
            if (x.min().item() < 0) or (x.max().item() > 1):
                raise RuntimeError(
                    'Expected input to VGG features to be in [0,1] range'
                )
        # Apply normalisation
        return bd.normalize_torchvision_imagenet(x)


_MODES = ['l1', 'l2']
_VERSIONS = ['gatys', 'johnson']
_REDUCTIONS = ['sum', 'mean', 'none']


def _check_params(version, mode, reduction):
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f'Unknown reduction for PerceptualLoss: {reduction}'
            f'\nAvailable reductions: {_REDUCTIONS}'
        )
    if version not in _VERSIONS:
        raise ValueError(
            f'Unknown PerceptualLoss version: {version}'
            f'\nAvailable versions. {_VERSIONS}'
        )
    if mode not in _MODES:
        raise ValueError(
            f'Unknown PerceptualLoss mode: {version}' f'\nAvailable modes: {_MODES}'
        )


class StyleLoss(Module):
    def __init__(
        self,
        version='johnson',
        mode='l2',
        reduction='mean',
        aggregate_scales=True,
        autonormalise=True,
    ):
        super().__init__()
        _check_params(version, mode, reduction)
        self.autonormalise = autonormalise

        if mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)

        if version == 'gatys':
            self.vgg_features = PretrainedVggFeatures(
                size=19, extract_list=[1, 6, 11, 20, 29]
            )
        elif version == 'johnson':
            self.vgg_features = PretrainedVggFeatures(
                size=16, extract_list=[3, 8, 15, 22]
            )

        self.gram = GramMatrix()
        self.aggregate_scales = aggregate_scales

    def forward(self, x, y):
        norm_x, norm_y = x, y
        if self.autonormalise:
            norm_x = self.vgg_features.normalise_01_input(x)
            norm_y = self.vgg_features.normalise_01_input(y)
        xy_feats = zip(self.vgg_features(norm_x), self.vgg_features(norm_y))
        ret = [self.loss(self.gram(xf), self.gram(yf)) for xf, yf in xy_feats]

        if self.aggregate_scales:
            return sum(ret) / len(ret)
        else:
            return ret


def FeatureLoss(Module):
    def __init__(
        self, version='johnson', mode='l2', reduction='mean', autonormalise=True
    ):
        super().__init__()
        _check_params(version, mode, reduction)
        self.autonormalise = autonormalise

        if mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)

        if version == 'gatys':
            self.vgg_features = PretrainedVggFeatures(size=19, extract_list=[22])
        elif version == 'johnson':
            self.vgg_features = PretrainedVggFeatures(size=16, extract_list=[8])

    def forward(self, x, y):
        norm_x, norm_y = x, y
        if self.autonormalise:
            norm_x = self.vgg_features.normalise_01_input(x)
            norm_y = self.vgg_features.normalise_01_input(y)

        xf = list(self.vgg_features(norm_x).values())[0]
        yf = list(self.vgg_features(norm_y).values())[0]
        return self.loss(xf, yf)


# Perceptual loss uses both the style losses AND the feature loss
class PerceptualLoss(Module):
    def __init__(
        self,
        version='johnson',
        style_mode='l2',
        feature_mode='l2',
        reduction='mean',
        aggregate_scales=True,
        autonormalise=True,
    ):
        super().__init__()
        _check_params(version, style_mode, reduction)
        _check_params(version, feature_mode, reduction)
        self.autonormalise = autonormalise

        if style_mode == 'l1':
            self.style_loss = nn.L1Loss(reduction=reduction)
        elif style_mode == 'l2':
            self.style_loss = nn.MSELoss(reduction=reduction)
        if feature_mode == 'l1':
            self.feature_loss = nn.L1Loss(reduction=reduction)
        elif feature_mode == 'l2':
            self.feature_loss = nn.MSELoss(reduction=reduction)

        if version == 'gatys':
            with bd.magic_off():
                self.style_layers = [1, 6, 11, 20, 29]
                self.content_layers = [22]
            self.vgg_features = PretrainedVggFeatures(
                size=19, extract_list=self.style_layers + self.content_layers
            )
        elif version == 'johnson':
            with bd.magic_off():
                self.style_layers = [3, 8, 15, 22]
                self.content_layers = [8]
            self.vgg_features = PretrainedVggFeatures(
                size=16, extract_list=self.style_layers + self.content_layers
            )

        self.gram = GramMatrix()
        self.aggregate_scales = aggregate_scales

    def forward(self, x, y):
        norm_x, norm_y = x, y
        if self.autonormalise:
            norm_x = self.vgg_features.normalise_01_input(x)
            norm_y = self.vgg_features.normalise_01_input(y)
        x_feats = self.vgg_features(norm_x)
        y_feats = self.vgg_features(norm_y)
        style_loss = [
            self.style_loss(self.gram(x_feats[key]), self.gram(y_feats[key]))
            for key in self.style_layers
        ]

        if self.aggregate_scales:
            style_loss = sum(style_loss) / len(style_loss)

        xf = x_feats[self.content_layers[0]]
        yf = y_feats[self.content_layers[0]]
        feature_loss = self.feature_loss(xf, yf)

        if self.aggregate_scales:
            return style_loss + feature_loss
        else:
            # Returns a list
            return style_loss + [feature_loss]


#  # This loss is between features of the discriminator between real and fake samples
#  # when training GANS
class FeatureMatchingLoss(Module):
    def __init__(
        self,
        mode='l2',
        reduction='mean',
        aggregate_scales=True,
        use_gram_matrix=False,
    ):
        super().__init__()
        _check_params('gatys', mode, reduction)

        if mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)
        if use_gram_matrix:
            self.gram = GramMatrix()
        else:
            self.gram = None
        self.aggregate_scales = aggregate_scales

    def forward(self, x_feature_list, y_feature_list):
        xy_feats = zip(x_feature_list, y_feature_list)
        if self.gram is not None:
            ret = [self.loss(self.gram(xf), self.gram(yf)) for xf, yf in xy_feats]
        else:
            ret = [self.loss(xf, yf) for xf, yf in xy_feats]

        if self.aggregate_scales:
            return sum(ret) / len(ret)
        else:
            return ret

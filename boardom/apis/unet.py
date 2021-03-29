import boardom as bd
from torch import nn
from .layers import activation, normalization, residual, block
from boardom.models.resnet import pretrained_resnet_layers, determine_channel_sizes

# For consistency with the lowercase API
def pretrained_unet():
    return PretrainedUNet()


def custom_unet():
    return CustomUnet()


def _build_guided_filter_layer(n_in, n_down, n_final):
    n_return = n_in
    with bd.cfg.g.guided_filter:
        kernel_size = bd.cfg.kernel_size
        epsilon = bd.cfg.epsilon
        should_bottlenect = bd.cfg.bottleneck_gf_adapter
        grouped = bd.cfg.grouped_gf
    adapter = None
    if grouped:
        n_return = n_final
        if (n_down > 512) and should_bottlenect:
            adapter = [
                ['conv2d', n_down, n_down // 4, 1],
                ['conv2d', n_down // 4, n_final, 1],
            ]
        else:
            adapter = ['conv2d', n_down, n_final, 1]
    elif n_in != n_down:
        if (n_down > 512) and should_bottlenect:
            adapter = [
                ['conv2d', n_down, n_down // 4, 1],
                ['conv2d', n_down // 4, n_in, 1],
            ]
        else:
            adapter = ['conv2d', n_down, n_in, 1]
    args = {
        'epsilon': epsilon,
        'kernel_size': kernel_size,
        'channel_adapter': adapter,
        'grouped': grouped,
    }
    return ['guidedfilter', args], n_return


def _build_lms_guided_filter_layer(n_in, n_down, n_final):
    n_return = n_in
    grp = bd.cfg.g.lms_guided_filter
    should_bottlenect = grp.bottleneck_gf_adapter
    grouped = grp.grouped_gf
    learn_epsilon = grp.learn_epsilon
    adapter = None
    if grouped:
        n_return = n_final
        if (n_down > 512) and should_bottlenect:
            adapter = [
                ['conv2d', n_down, n_down // 4, 1],
                ['conv2d', n_down // 4, n_final, 1],
            ]
        else:
            adapter = ['conv2d', n_down, n_final, 1]
    elif n_in != n_down:
        if (n_down > 512) and should_bottlenect:
            adapter = [
                ['conv2d', n_down, n_down // 4, 1],
                ['conv2d', n_down // 4, n_in, 1],
            ]
        else:
            adapter = ['conv2d', n_down, n_in, 1]
    args = {
        'kernel_sizes': grp.kernel_sizes,
        'epsilon': grp.epsilon,
        'channel_adapter': adapter,
        'grouped': grouped,
        'learn_epsilon': learn_epsilon,
    }
    return ['learnedmultiscaleguidedfilter', args], n_return


def fusion(n_in, n_down, n_final):
    fusion_type = bd.cfg.fusion
    if fusion_type == 'guided_filter':
        return _build_guided_filter_layer(n_in, n_down, n_final)
    elif fusion_type == 'lms_guided_filter':
        return _build_lms_guided_filter_layer(n_in, n_down, n_final)
    else:
        upsample_type = bd.cfg.upsample
        if upsample_type == 'transpose':
            with bd.cfg.g.upsample:
                kernel_size = bd.cfg.kernel_size
            up = ['convtranspose2dsame', n_down, n_in, kernel_size, 2]
        else:
            if bd.cfg.first_conv_then_resize:
                up = [['conv2d', n_down, n_in, 1], ['resize', 2, upsample_type]]
            else:
                up = [['resize', 2, upsample_type], ['conv2d', n_down, n_in, 1]]
        up = [['multiselect', (0, 2)], ['parallel', ['identity'], up]]
        return [{'upsample': up, fusion_type: [fusion_type]}], n_in


def _down(n_in, down_type, kernel_size):
    if down_type == 'strided':
        down = ['conv2dsame', n_in, n_in, kernel_size, 2]
    elif down_type in ['nearest', 'bilinear']:
        down = ['resize', 0.5, down_type]
    else:
        raise ValueError(f'Unknown downsample type {down_type}')
    return down


def down_block(n_in):
    with bd.cfg.g.down:
        kernel_size = bd.cfg.kernel_size
        downsample_type = bd.cfg.downsample
    down = _down(n_in, downsample_type.split('_')[0], kernel_size)
    if downsample_type in [
        'strided_preact_bbn',
        'nearest_preact_bbn',
        'bilinear_preact_bbn',
    ]:
        main = [normalization(n_in), activation(), down]
        b, _ = block(n_in, n_in, 'nac', kernel_size)
        main = main + [b]
        shortcut = [normalization(n_in), down]
        return residual(main, shortcut)
    else:
        return down


def build_blocks(n_in, n_out):
    block_types = bd.cfg.block_types
    kernel_size = bd.cfg.kernel_size
    blocks = []
    for bt in block_types:
        b, new_out = block(n_in, n_out, bt, kernel_size)
        blocks.append(b)
        n_in = new_out
    return blocks


def pre_down(n_in, n_out):
    with bd.cfg.g.pre_down:
        return build_blocks(n_in, n_out)


def post_fuse(n_in, n_out):
    fusion_type = bd.cfg.fusion
    if fusion_type == 'cat':
        n_in = 2 * n_in
    with bd.cfg.g.post_fuse:
        return build_blocks(n_in, n_out)


def build_pretrained_level(sublevel, n_in, n_down, n_final, encoder_module):
    branch = [
        'branch',
        {'skip': nn.Identity(), 'down': encoder_module, 'child': sublevel},
    ]
    fusion_module, nfuse = fusion(n_in, n_down, n_final)
    ret = [
        {
            'branch': branch,
            'fusion': fusion_module,
            'post_fuse': post_fuse(nfuse, n_final),
        }
    ]
    return ret


def build_custom_level(sublevel, n_in, n_down, n_final):
    branches = {
        'pre_down': pre_down(n_in, n_down),
        'down': down_block(n_down),
        'child': sublevel,
    }
    branch = ['branch', branches]
    fusion_module, nfuse = fusion(n_down, n_down, n_final)
    ret = [
        {
            'branch': branch,
            'fusion': fusion_module,
            'post_fuse': post_fuse(nfuse, n_final),
        }
    ]
    return ret


class PretrainedUNet(bd.Module):
    def __init__(self):
        super().__init__()
        with bd.cfg.g.pretrained_unet:
            self.main = self._build()

    def _build(self):
        in_features = 3
        out_features = bd.cfg.out_features
        encoder_modules = pretrained_resnet_layers(
            network_name=bd.cfg.network_name,
            num_pretrained_layers=bd.cfg.num_pretrained_layers,
            freeze_batchnorm=bd.cfg.freeze_batchnorm,
            split_before_relus=bd.cfg.split_before_relus,
        )
        encoder_modules = [bd.magic_module(m) for m in encoder_modules[:-1]]
        nf = determine_channel_sizes(encoder_modules)
        num_levels = len(encoder_modules)
        #  ret = [nn.Identity()]
        with bd.cfg.g.bottleneck:
            ret = build_blocks(nf[-1], nf[-1])

        for i, lvl in enumerate(range(num_levels, 0, -1)):
            with bd.cfg.g[f'level_{lvl}']:
                n_down = nf[lvl - 1]
                if lvl == 1:
                    n_in = in_features
                    n_final = out_features
                else:
                    n_in = n_final = nf[lvl - 2]
                ret = build_pretrained_level(
                    ret, n_in, n_down, n_final, encoder_modules[lvl - 1]
                )

        with bd.cfg.g.final:
            ret[0]['final_activation'] = activation()
        return bd.magic_module(ret)

    def forward(self, x):
        return self.main(x)


class CustomUnet(bd.Module):
    def __init__(self):
        super().__init__()
        with bd.cfg.g.custom_unet:
            self.main = self._build()

    def _build(self):
        nf = bd.cfg.hidden_features
        in_features = bd.cfg.in_features
        out_features = bd.cfg.out_features
        num_levels = len(nf)
        with bd.cfg.g.bottleneck:
            ret = build_blocks(nf[-1], nf[-1])
        for i, lvl in enumerate(range(num_levels, 0, -1)):
            with bd.cfg.g[f'level_{lvl}']:
                n_down = nf[lvl - 1]
                if lvl == 1:
                    n_in = in_features
                    n_final = out_features
                else:
                    n_in = n_final = nf[lvl - 2]
                ret = build_custom_level(ret, n_in, n_down, n_final)

        with bd.cfg.g.final:
            ret[0]['final_activation'] = activation()
        return bd.magic_module(ret)

    def forward(self, x):
        return self.main(x)

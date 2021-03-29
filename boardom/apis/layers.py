import boardom as bd


def activation():
    act = bd.cfg.activation.lower()
    if act is None or act == 'none':
        ret = []
    elif act in ['relu', 'elu', 'selu', 'sigmoid', 'tanh']:
        return [act]
    elif act == 'leakyrelu':
        ret = [act, bd.cfg.leaky_relu_slope]
    else:
        raise NotImplementedError(f'API for activation {act} not yet implemented.')
    return bd.magic_module(ret)


def normalization(num_channels, num_groups=None):
    norm = bd.cfg.normalization.lower()
    if norm is None or norm == 'none':
        ret = []
    elif norm in (
        'batchnorm1d',
        'batchnorm2d',
        'batchnorm3d',
        'instancenorm1d',
        'instancenorm2d',
        'instancenorm3d',
    ):
        kwargs = dict(
            eps=bd.cfg.norm_eps,
            momentum=bd.cfg.norm_momentum,
            affine=bd.cfg.norm_affine,
            track_running_stats=bd.cfg.norm_track_running_stats,
        )
        ret = [norm, num_channels, kwargs]
    else:
        raise NotImplementedError(f'API for {norm} not implemented.')
    return bd.magic_module(ret)


def residual(module_a, module_b):
    return bd.magic_module([['split', module_a, module_b], ['sum']])


def block(n_in, n_out, btype, kernel_size):
    conv_1 = ['conv2dsame', n_in, n_out, kernel_size, {'pad_mode': 'reflect'}]
    conv_2 = ['conv2dsame', n_out, n_out, kernel_size, {'pad_mode': 'reflect'}]
    if btype == 'r_preact_bbn':
        branch = [normalization(n_in), ['conv2d', n_in, n_out, 1]]
    else:
        branch = ['identity'] if n_in == n_out else ['conv2d', n_in, n_out, 1]

    if btype == 'c':
        ret = [conv_1]
        new_out = n_out
    elif btype == 'n':
        ret = [normalization(n_in)]
        new_out = n_in
    elif btype == 'a':
        ret = [activation()]
        new_out = n_in
    else:
        act = activation()
        norm_in = normalization(n_in)
        norm_out = normalization(n_out)
        if btype == 'cna':
            ret = [conv_1, norm_out, act]
            new_out = n_out

        elif btype == 'nac':
            ret = [norm_in, act, conv_1]
            new_out = n_out

        elif btype == 'r_orig':
            main = [conv_1, norm_out, act, conv_2, norm_out]
            ret = [residual(main, branch), act]
            new_out = n_out

        elif btype in ['r_preact', 'r_preact_bbn']:
            main = [norm_in, act, conv_1, norm_out, act, conv_2]
            ret = residual(main, branch)
            new_out = n_out

        elif btype == 'r_preact_c':
            main = [conv_1, norm_out, act, conv_2]
            ret = residual(main, branch)
            new_out = n_out
        else:
            raise NotImplementedError(f'API for {btype} not yet implemented.')

    return bd.magic_module(ret), new_out

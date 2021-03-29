import boardom as bd
from .layers import activation


def pretrained_resnet(**kwargs):
    ret = bd.autoconfig(bd.PretrainedResnet, ignore=['final_activation'])
    if 'final_activation' not in kwargs:
        with bd.cfg.g.final:
            final_activation = activation()
        return ret(final_activation=final_activation, **kwargs)
    else:
        return ret(**kwargs)

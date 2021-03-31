import types
import torch
from torch import nn
import torchvision as tv
from torchvision.models.resnet import BasicBlock, Bottleneck
import boardom as bd
from .module import Module

# This is adapted from torchvision
def forward_no_relu_basic(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    #  out = self.relu(out)

    return out


def forward_no_relu_bottleneck(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


def pretrained_resnet_layers(
    network_name,
    num_pretrained_layers=6,
    freeze_bn_running_stats=False,
    split_before_relus=False,
):
    if (num_pretrained_layers < 0) or (num_pretrained_layers > 6):
        raise ValueError('Expected num_pretrained_layers to be in the range [0,6]')
    avail = tv.models.resnet.__all__
    if network_name not in avail:
        raise ValueError(f'Expected name to be one of {avail}, got "{network_name}"')
    bd.print_separator()
    bd.log(f'Fetching (pretrained) {network_name}.')
    bd.log(f'Number of pretrained layers: {num_pretrained_layers}.')
    model = tv.models.__dict__[network_name](pretrained=True)
    modules = [
        [model.conv1, model.bn1, model.relu],
        [model.maxpool, model.layer1],
        [model.layer2],
        [model.layer3],
        [model.layer4],
        [model.avgpool, nn.Flatten(1), model.fc],
    ]

    for i, module_list in enumerate(modules, 1):
        if i > num_pretrained_layers:
            break
        for mod in module_list:
            bd.set_pretrained(mod)
            if freeze_bn_running_stats:
                bd.log('Freezing batchnorm for module {name}.')
                bd.freeze_bn_running_stats(mod)
    if split_before_relus:
        for m in [model.layer1, model.layer2, model.layer3, model.layer4]:
            block = m[-1]
            if isinstance(block, BasicBlock):
                block.forward = types.MethodType(forward_no_relu_basic, block)
            elif isinstance(block, Bottleneck):
                block.forward = types.MethodType(forward_no_relu_bottleneck, block)
            else:
                raise RuntimeError(
                    f'Attempted to split before relu from module type: {torch.typename(block)}'
                )

        modules = [
            [model.conv1, model.bn1],
            [model.relu, model.maxpool, model.layer1],
            [model.relu, model.layer2],
            [model.relu, model.layer3],
            [model.relu, model.layer4],
            [model.relu, model.avgpool, nn.Flatten(1), model.fc],
        ]

    return modules


def determine_channel_sizes(module_list, x_in=None):
    if x_in is None:
        x_in = torch.ones(1, 3, 256, 256)
    nf = []
    bd.print_separator()
    bd.log('Calculating channel sizes.')
    for module in module_list:
        x_in = module(x_in)
        nf.append(x_in.shape[1])
    bd.log(f'Sizes are {nf}')
    bd.print_separator()
    return nf


# 5 is up to the fc head
class PretrainedResnet(Module):
    def __init__(
        self,
        network_name,
        num_classes=1000,
        num_pretrained_layers=5,
        freeze_bn_running_stats=False,
        final_activation=None,
    ):
        super().__init__()
        module_list = pretrained_resnet_layers(
            network_name, num_pretrained_layers, freeze_bn_running_stats
        )
        if num_classes != 1000:
            module_list[-1][-1] = nn.Linear(
                module_list[-1][-1].in_features, num_classes
            )
        modules = {'base': module_list[0] + [module_list[1][0]]}
        modules['layer1'] = module_list[1][1]
        modules['layer2'] = module_list[2]
        modules['layer3'] = module_list[3]
        modules['layer4'] = module_list[4]
        modules['head'] = module_list[5]
        if final_activation is not None:
            modules['final_activation'] = final_activation

        self.main = bd.magic_module([modules])

    def forward(self, x):
        return self.main(x)

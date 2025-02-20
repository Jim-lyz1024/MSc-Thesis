import torch
import torch.nn as nn
from torch.nn import functional as F


def init_network_weights(model, init_type="normal", gain=0.02):
    def _init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, "weight") and (
            class_name.find("Conv") != -1 or class_name.find("linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            else:
                raise NotImplementedError(
                    "Initialization Method {} is not Implemented.".format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif class_name.find("InstanceNorm2d") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
        elif class_name.find("BatchNorm2d") != -1:
            raise NotImplementedError("BatchNorm2d is Not Implemented.")

    model.apply(_init_func)


class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                bias=False,
            )
        ]
        conv_block += [nn.InstanceNorm2d(num_features=num_features)]
        conv_block += [nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                bias=False,
            )
        ]
        conv_block += [nn.InstanceNorm2d(num_features=num_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class FCN(nn.Module):
    """Fully convolutional network."""

    def __init__(self, in_channels, out_channels, num_features=32, n_blocks=3):
        super().__init__()

        backbone = []
        backbone += [nn.ReflectionPad2d(1)]
        backbone += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        backbone += [nn.InstanceNorm2d(num_features=num_features)]
        backbone += [nn.ReLU(True)]

        for _ in range(n_blocks):
            backbone += [ResnetBlock(num_features)]

        self.backbone = nn.Sequential(*backbone)

        self.gctx_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * num_features,
                out_channels=num_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(num_features=num_features),
            nn.ReLU(True),
        )

        self.regress = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, x, lmda=1.0, return_p=False):
        original_x = x

        x = self.backbone(x)
        c = F.adaptive_avg_pool2d(x, (1, 1))
        c = c.expand_as(x)
        x = torch.cat([x, c], 1)
        x = self.gctx_fusion(x)

        perturbation = self.regress(x)
        perturbated_x = original_x + lmda * perturbation

        if return_p:
            return perturbated_x, perturbation

        return perturbated_x


def build_network(network_type):
    if network_type == "fcn_3x32_gctx":
        net = FCN(in_channels=3, out_channels=3, num_features=32, n_blocks=3)
    elif network_type == "fcn_3x64_gctx":
        net = FCN(in_channels=3, out_channels=3, num_features=64, n_blocks=5)
    else:
        raise NotImplementedError
    init_network_weights(net, init_type="normal", gain=0.02)

    return net

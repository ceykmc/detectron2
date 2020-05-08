# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1,
                                    bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetv2(Backbone):
    def __init__(self, cfg: CfgNode):
        super(ShuffleNetv2, self).__init__()

        ratio = cfg.MODEL.SHUFFLENETV2.RATIO
        stages_repeats = [4, 8, 4]
        if ratio == 0.5:
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif ratio == 1.0:
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif ratio == 1.5:
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif ratio == 2.0:
            stages_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError(F"ShuffleNetV2 unsupported ratio number: {ratio}")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = [F"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(
                    InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        if cfg.MODEL.SHUFFLENETV2.IMAGENET_PRETRAINED_MODEL:
            state_dict = torch.load(
                cfg.MODEL.SHUFFLENETV2.IMAGENET_PRETRAINED_MODEL)
            for name in list(state_dict.keys()):
                if "fc" in name:
                    state_dict.pop(name)
            self.load_state_dict(state_dict, strict=True)
            print(F"ShuffleNetv2 load imagenet pretrained model: "
                  F"{cfg.MODEL.SHUFFLENETV2.IMAGENET_PRETRAINED_MODEL}")
        else:
            self.init_params()

        self._out_features = ["stage2", "stage3", "stage4"]
        self._out_feature_strides = {
            self._out_features[0]: 8,
            self._out_features[1]: 16,
            self._out_features[2]: 32
        }
        self._out_feature_channels = {
            self._out_features[0]: stages_out_channels[1],
            self._out_features[1]: stages_out_channels[2],
            self._out_features[2]: stages_out_channels[4]
        }

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = dict()

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        outputs["stage2"] = x

        x = self.stage3(x)
        outputs["stage3"] = x

        x = self.stage4(x)
        x = self.conv5(x)
        outputs["stage4"] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name],
                            stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, input_shape=None):
    return ShuffleNetv2(cfg)

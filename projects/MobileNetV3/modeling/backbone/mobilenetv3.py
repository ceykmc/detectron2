# -*- coding: utf-8 -*-
"""code from https://github.com/xiaolai-sqlai/mobilenetv3/blob/master/mobilenetv3.py
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size,
                      in_size // reduction,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction,
                      in_size,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), nn.BatchNorm2d(in_size), hsigmoid())

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear,
                 semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.out_channel = out_size

        self.conv1 = nn.Conv2d(in_size,
                               expand_size,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size,
                               expand_size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=expand_size,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size,
                               out_size,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size,
                          out_size,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160,
                               960,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


class MobileNetV3_Small(Backbone):
    def __init__(self, out_features=None, pretrained_model_file_path=None):
        super(MobileNetV3_Small, self).__init__()

        stem_out_channel = 16
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=stem_out_channel,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=16),
            hswish(),
        )
        current_stride = 2
        self._out_feature_strides = {"stem": 2}
        self._out_feature_channels = {"stem": stem_out_channel}

        bneck = [
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        ]
        self.stages_and_names = []
        for i, block in enumerate(bneck):
            name = F"res{i+1}"
            self.add_module(name, block)
            self.stages_and_names.append((block, name))
            self._out_feature_strides[name] = current_stride = \
                current_stride * block.stride
            self._out_feature_channels[name] = block.out_channel

        last_block = nn.Sequential(
            nn.Conv2d(96,
                      576,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        name = F"res{len(bneck)+1}"
        self.add_module(name, last_block)
        self.stages_and_names.append((last_block, name))
        self._out_feature_strides[name] = current_stride
        self._out_feature_channels[name] = 576

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, \
                "Available children: {}".format(", ".join(children))

        self.init_params()
        if pretrained_model_file_path is not None:
            self.load_pretrained_model(pretrained_model_file_path)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def load_pretrained_model(self, pretrained_model_file_path):
        print(self.load_pretrained_model.__name__)
        pretrained_model = _convert_mobilenetv3_small_pretrained_model(pretrained_model_file_path)
        self.load_state_dict(pretrained_model, strict=False)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _convert_mobilenetv3_small_pretrained_model(pretrained_model_file_path):
    pretrained_model = torch.load(pretrained_model_file_path)["state_dict"]
    origin_keys = list(pretrained_model.keys())
    for origin_key in origin_keys:
        if "bneck" in origin_key:
            bneck_index = int(re.findall("([0-9]{1,2})", origin_key)[0])
            if ".se" in origin_key:
                new_key = origin_key[origin_key.find(".se"):]
                new_key = F"res{bneck_index+1}{new_key}"
            else:
                new_key = origin_key.replace(F"module.bneck.{bneck_index}", F"res{bneck_index+1}")
            pretrained_model[new_key] = pretrained_model.pop(origin_key)
        elif "conv1" in origin_key:  # stem conv
            new_key = "stem.0.weight"
            pretrained_model[new_key] = pretrained_model.pop(origin_key)
        elif "bn1" in origin_key:  # stem bn
            new_key = origin_key.replace("module.bn1", "stem.1")
            pretrained_model[new_key] = pretrained_model.pop(origin_key)
        elif "conv2" in origin_key:  # last block conv
            new_key = "res12.0.weight"
            pretrained_model[new_key] = pretrained_model.pop(origin_key)
        elif "bn2" in origin_key:  # last block bn
            new_key = origin_key.replace("module.bn2", "res12.1")
            pretrained_model[new_key] = pretrained_model.pop(origin_key)
        else:
            continue
    return pretrained_model

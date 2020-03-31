# -*- coding: utf-8 -*-

import math
from typing import List
import torch.nn as nn

from detectron2.config import CfgNode
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from ..efficientnet import build_efficientnet
from .bifpn_module import BiFPNModule


class BiFPN(Backbone):
    def __init__(self, bottom_up: Backbone, in_features: List[str],
                 bifpn_w: int, bifpn_d: int):
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        self.bottom_up = bottom_up
        self.lateral_convs = nn.ModuleList()
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = Conv2d(in_channels=in_channels,
                                  out_channels=bifpn_w,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  norm=nn.BatchNorm2d(num_features=bifpn_w,
                                                      eps=1e-4,
                                                      momentum=0.003),
                                  activation=nn.ReLU(inplace=True))
            self.lateral_convs.append(lateral_conv)
        self.bifpn_modules = nn.ModuleList([
            BiFPNModule(levels=len(in_features), channels=bifpn_w)
            for _ in range(bifpn_d)
        ])

        self.in_features = in_features
        self._out_feature_strides = {
            F"p{int(math.log2(s))}": s
            for s in in_strides
        }
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: bifpn_w for k in self._out_features}
        self._size_divisibility = in_strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = [
            lateral_conv(bottom_up_features[f])
            for lateral_conv, f in zip(self.lateral_convs, self.in_features)
        ]
        for bifpn_module in self.bifpn_modules:
            results = bifpn_module(results)
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name],
                            stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_retinanet_efficientnet_bifpn_backbone(cfg: CfgNode,
                                                input_shape=None):
    bottom_up = build_efficientnet(cfg)

    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    bifpn_w = cfg.MODEL.BIFPN.BIFPN_W
    bifpn_d = cfg.MODEL.BIFPN.BIFPN_D

    bifpn = BiFPN(bottom_up=bottom_up,
                  in_features=in_features,
                  bifpn_w=bifpn_w,
                  bifpn_d=bifpn_d)
    return bifpn

# -*- coding: utf-8 -*-

import math
from typing import List

import torch.nn as nn

from detectron2.modeling import Backbone
from detectron2.layers import Conv2d

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
                                  norm=nn.BatchNorm2d(num_features=bifpn_w),
                                  activation=nn.ReLU(inplace=True))
            self.lateral_convs.append(lateral_conv)
        self.bifpn_modules = nn.ModuleList([
            BiFPNModule(levels=len(in_features), channels=bifpn_w)
            for _ in range(bifpn_d)
        ])

        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: bifpn_w for k in self._out_features}

    def forward(self, x):
        pass

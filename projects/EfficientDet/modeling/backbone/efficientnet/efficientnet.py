# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone

from .utils import Swish, drop_connect, get_same_padding_conv2d, round_filters, round_repeats

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "batch_norm_momentum", "batch_norm_epsilon", "dropout_rate", "num_classes",
    "width_coefficient", "depth_coefficient", "depth_divisor", "min_depth",
    "drop_connect_rate", "image_size"
])

# Parameters for an individual model block
BlockArgs = collections.namedtuple("BlockArgs", [
    "kernel_size", "num_repeat", "input_filters", "output_filters",
    "expand_ratio", "id_skip", "stride", "se_ratio"
])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is
                       not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp,
                                       out_channels=oup,
                                       kernel_size=1,
                                       bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup,
                                       momentum=self._bn_mom,
                                       eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup,
                                   momentum=self._bn_mom,
                                   eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1,
                int(self._block_args.input_filters
                    * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup,
                                     out_channels=num_squeezed_channels,
                                     kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels,
                                     out_channels=oup,
                                     kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup,
                                    out_channels=final_oup,
                                    kernel_size=1,
                                    bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup,
                                   momentum=self._bn_mom,
                                   eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = \
            self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x,
                                 p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(Backbone):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        blocks_args = self.read_blocks_args(cfg)
        global_params = self.read_global_params(cfg)
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._out_features = cfg.MODEL.EFFICIENTNET.OUT_FEATURES

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=2,
                                 bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels,
                                   momentum=bn_mom,
                                   eps=bn_eps)

        current_stride = 2
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": out_channels}

        # Build blocks
        count = 0
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat,
                                         self._global_params))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))

            name = F"res{count}"
            current_stride *= block_args.stride
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = block_args.output_filters
            count += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params))
                name = F"res{count}"
                current_stride *= block_args.stride
                self._out_feature_strides[name] = current_stride
                self._out_feature_channels[name] = block_args.output_filters
                count += 1

        self._swish = Swish()

    def extract_features(self, inputs):
        outputs = dict()
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        if "stem" in self._out_features:
            outputs["stem"] = x
        # Blocks
        for idx, block in enumerate(self._blocks):
            name = F"res{idx}"
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    @staticmethod
    def read_blocks_args(cfg: CfgNode):
        block_args = list()
        kernel_sizes = cfg.MODEL.EFFICIENTNET.BLOCKARGS.KERNEL_SIZES
        repeat_nums = cfg.MODEL.EFFICIENTNET.BLOCKARGS.REPEAT_NUMS
        input_filters = cfg.MODEL.EFFICIENTNET.BLOCKARGS.INPUT_FILTERS
        output_filters = cfg.MODEL.EFFICIENTNET.BLOCKARGS.OUTPUT_FILTERS
        expand_ratios = cfg.MODEL.EFFICIENTNET.BLOCKARGS.EXPAND_RATIOS
        se_ratios = cfg.MODEL.EFFICIENTNET.BLOCKARGS.SE_RATIOS
        strides = cfg.MODEL.EFFICIENTNET.BLOCKARGS.STRIDES
        for k, r, i, o, e, se, st in zip(kernel_sizes, repeat_nums,
                                         input_filters, output_filters,
                                         expand_ratios, se_ratios, strides):
            block_args.append(
                BlockArgs(kernel_size=k,
                          num_repeat=r,
                          input_filters=i,
                          output_filters=o,
                          expand_ratio=e,
                          id_skip=True,
                          stride=st,
                          se_ratio=se))
        return block_args

    @staticmethod
    def read_global_params(cfg: CfgNode):
        batch_norm_momentum = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.BATCH_NORM_MOMENTUM
        batch_norm_epsilon = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.BATCH_NORM_EPSILON
        dropout_rate = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.DROPOUT_RATE
        drop_connect_rate = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.DROPOUT_CONNECT_RATE
        num_classes = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.NUM_CLASSES
        width_coefficient = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.WIDTH_COEFFICIENT
        depth_coefficient = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.DEPTH_COEFFICIENT
        depth_divisor = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.DEPTH_DIVISOR
        image_size = cfg.MODEL.EFFICIENTNET.GLOBAL_PARAMS.IMAGE_SIZE
        global_params = GlobalParams(batch_norm_momentum=batch_norm_momentum,
                                     batch_norm_epsilon=batch_norm_epsilon,
                                     dropout_rate=dropout_rate,
                                     drop_connect_rate=drop_connect_rate,
                                     num_classes=num_classes,
                                     width_coefficient=width_coefficient,
                                     depth_coefficient=depth_coefficient,
                                     depth_divisor=depth_divisor,
                                     min_depth=None,
                                     image_size=list(image_size))
        return global_params

    def forward(self, inputs):
        return self.extract_features(inputs)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def build_efficientnet(cfg: CfgNode):
    model = EfficientNet(cfg)

    model_url = "http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth"
    state_dict = model_zoo.load_url(model_url)
    model.load_state_dict(state_dict, strict=False)
    print("load efficientnet pretrained model")
    model.freeze_bn()
    print("fozen batchnorm layers")
    return model

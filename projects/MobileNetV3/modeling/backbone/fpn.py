# -*- coding: utf-8 -*-

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.modeling import FPN, BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelP6P7

from .mobilenetv3 import MobileNetV3_Small


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_fpn_backbone(cfg: CfgNode,
                                   input_shape: ShapeSpec = None):
    mobilenetv3_out_features = cfg.MODEL.MOBILENETV3.OUT_FEATURES
    pretrained_model_file_path = cfg.MODEL.MOBILENETV3.PRETRAINED_MODEL_PATH
    if len(pretrained_model_file_path) == 0:
        pretrained_model_file_path = None
    bottom_up = MobileNetV3_Small(
        mobilenetv3_out_features,
        pretrained_model_file_path=pretrained_model_file_path)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()[in_features[-1]].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7,
                                out_channels,
                                in_feature=in_features[-1]),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

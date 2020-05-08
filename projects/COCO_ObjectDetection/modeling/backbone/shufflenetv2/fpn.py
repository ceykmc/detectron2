# -*- coding: utf-8 -*-

from detectron2.config import CfgNode
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling import FPN

from .shufflenetv2 import build_shufflenetv2_backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_shufflenetv2_fpn_backbone(cfg: CfgNode, input_shape=None):
    bottom_up = build_shufflenetv2_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

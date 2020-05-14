# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_shufflenetv2_config(cfg: CN):
    _C = cfg

    _C.MODEL.SHUFFLENETV2 = CN()
    _C.MODEL.SHUFFLENETV2.RATIO = 0.5
    _C.MODEL.SHUFFLENETV2.IMAGENET_PRETRAINED_MODEL = ""

    _C.SOLVER.OPTIMIZER_TYPE = "SGD"

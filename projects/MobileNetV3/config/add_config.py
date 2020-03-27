# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_mobilenetv3_config(cfg: CN):
    _C = cfg

    _C.MODEL.MOBILENETV3 = CN()
    _C.MODEL.MOBILENETV3.PRETRAINED_MODEL_PATH = ""
    _C.MODEL.MOBILENETV3.OUT_FEATURES = ["res3", "res4", "res5"]

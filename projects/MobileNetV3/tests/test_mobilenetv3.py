# -*- coding: utf-8 -*-

import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from config import add_mobilenetv3_config
from modeling import build_mobilenetv3_fpn_backbone
from modeling.backbone import MobileNetV3_Small


def setup_cfg():
    config_file_path = "./configs/retinanet_M_50_FPN_1x.yaml"
    cfg = get_cfg()
    add_mobilenetv3_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


class TestMobilenetv3(unittest.TestCase):
    def test_mobilenetv3(self):
        cfg = setup_cfg()
        input_x = torch.randn(4, 3, 288, 320)
        out_features = cfg.MODEL.MOBILENETV3.OUT_FEATURES
        pretrained_model_file_path = cfg.MODEL.MOBILENETV3.PRETRAINED_MODEL_PATH
        model = MobileNetV3_Small(
            out_features=out_features,
            pretrained_model_file_path=pretrained_model_file_path)
        output_y = model(input_x)
        for k, v in output_y.items():
            print(F"mobilenetv3, {k}, {v.shape}")

    def test_fpn(self):
        cfg = setup_cfg()
        model = build_mobilenetv3_fpn_backbone(cfg)
        input_x = torch.randn(4, 3, 288, 320)
        output_y = model(input_x)
        for k, v in output_y.items():
            print(F"FPN, {k}, {v.shape}")

    def test_retinanet(self):
        cfg = setup_cfg()
        retina_model = build_model(cfg)
        # print(retina_model)


def main():
    unittest.main()


if __name__ == "__main__":
    main()

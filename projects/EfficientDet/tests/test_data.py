# -*- coding: utf-8 -*-

import argparse
import unittest

import torch

from modeling.backbone.bifpn import build_retinanet_efficientnet_bifpn_backbone
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
from tests.parameterized_test_case import parameterizedTestCase


class TestData(parameterizedTestCase):
    def test_train_loader(self):
        train_loader = build_detection_train_loader(cfg=self.cfg)
        for batched_inputs in train_loader:
            for batched_input in batched_inputs:
                for k, v in batched_input.items():
                    print(F"{k}, {v.size() if isinstance(v, torch.Tensor) else v}")
            break

    def test_predict(self):
        train_loader = build_detection_train_loader(cfg=self.cfg)
        model = build_model(self.cfg)
        with EventStorage():
            for batched_inputs in train_loader:
                loss = model(batched_inputs)
                print(loss)
                break


def argument_parser():
    parser = argparse.ArgumentParser(description="imagenet training")
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    config_file_path = args.config_file
    param = {"config_file_path": config_file_path}
    suite = unittest.TestSuite()
    suite.addTest(parameterizedTestCase.parameterize(TestData, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

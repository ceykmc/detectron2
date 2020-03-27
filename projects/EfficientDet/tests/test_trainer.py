# -*- coding: utf-8 -*-

import argparse
import unittest

import torch

from trainer import ImageNetTrainer
from modeling import build_model
from tests.parameterized_test_case import parameterizedTestCase


class TestTrainer(parameterizedTestCase):
    def test_evaluation(self):
        trainer = ImageNetTrainer(self.cfg)

        model = build_model(self.cfg)
        check_point_file_path = "/home/lijun/.cache/torch/checkpoints/efficientnet-b0-355c32eb.pth"
        state_dict = torch.load(check_point_file_path)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.cfg.MODEL.DEVICE)
        model.eval()

        trainer.test(self.cfg, model)


def argument_parser():
    parser = argparse.ArgumentParser(description="imagenet training")
    parser.add_argument("--config_file",
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
    suite.addTest(parameterizedTestCase.parameterize(TestTrainer, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

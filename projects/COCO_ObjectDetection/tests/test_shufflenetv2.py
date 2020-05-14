# -*- coding: utf-8 -*-

import os
import argparse
import unittest

from config import add_shufflenetv2_config
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from modeling import build_model


def argument_parser():
    parser = argparse.ArgumentParser(description="ShuffleNetv2 test")
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def setup(args: argparse.ArgumentParser):
    cfg = get_cfg()
    add_shufflenetv2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


class TestShuffleNetv2(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super(TestShuffleNetv2, self).__init__(methodName)
        self.cfg = setup(param["args"])

    def test_create_model(self):
        build_model(self.cfg)

    def test_train_forward(self):
        model = build_model(self.cfg)
        train_data_loader = build_detection_train_loader(self.cfg, mapper=None)
        with EventStorage():
            for batched_inputs in train_data_loader:
                loss = model(batched_inputs)
                print(loss)
                break


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7, 8"
    args = argument_parser()
    param = {"args": args}

    testloader = unittest.TestLoader()
    testnames = testloader.getTestCaseNames(TestShuffleNetv2)
    suite = unittest.TestSuite()
    for name in testnames:
        suite.addTest(TestShuffleNetv2(name, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

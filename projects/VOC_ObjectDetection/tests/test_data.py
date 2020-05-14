# -*- coding: utf-8 -*-

import argparse
import unittest

from config import add_shufflenetv2_config
from detectron2.config import get_cfg
from detectron2.data import get_detection_dataset_dicts, build_detection_train_loader


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


class TestData(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super(TestData, self).__init__(methodName)
        self.cfg = setup(param["args"])

    def test_train_dataset(self):
        train_dataset_names = self.cfg.DATASETS.TRAIN
        train_dataset = get_detection_dataset_dicts(dataset_names=train_dataset_names)
        print(F"train data sample number: {len(train_dataset)}")

    def test_train_dataloader(self):
        train_data_loader = build_detection_train_loader(self.cfg)
        for step, batched_inputs in enumerate(train_data_loader):
            for data in batched_inputs:
                print(data.keys(), data["image"].size())
            break


def main():
    args = argument_parser()
    param = {"args": args}

    testloader = unittest.TestLoader()
    testnames = testloader.getTestCaseNames(TestData)
    suite = unittest.TestSuite()
    for name in testnames:
        suite.addTest(TestData(name, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

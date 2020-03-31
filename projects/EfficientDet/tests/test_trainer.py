# -*- coding: utf-8 -*-

import argparse
import unittest

from tests.parameterized_test_case import parameterizedTestCase
from trainer import EfficientDetTrainer


class TestTrainer(parameterizedTestCase):
    def test_evaluation(self):
        trainer = EfficientDetTrainer(self.cfg)


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
    suite.addTest(parameterizedTestCase.parameterize(TestTrainer, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import unittest

from detectron2.config import get_cfg
from config import add_config


def setup(config_file_path: str):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


class parameterizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName="runTest", param=None):
        super(parameterizedTestCase, self).__init__(methodName)
        assert isinstance(param, dict)
        assert "config_file_path" in param
        config_file_path = param["config_file_path"]
        self.cfg = setup(config_file_path)

    @staticmethod
    def parameterize(testcase_class, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter "param".
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_class(name, param=param))
        return suite

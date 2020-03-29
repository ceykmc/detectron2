# -*- coding: utf-8 -*-

import argparse
import unittest

from PIL import Image
import torch
from torchvision import transforms

from modeling.backbone.bifpn import build_retinanet_efficientnet_bifpn_backbone
from tests.parameterized_test_case import parameterizedTestCase


def read_image(image_path: str, image_size: tuple):
    image = Image.open(image_path).convert("RGB")
    tfms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = tfms(image).unsqueeze(0)
    return image


class TestBiFPN(parameterizedTestCase):
    def test_create_model(self):
        model = build_retinanet_efficientnet_bifpn_backbone(self.cfg)

    def test_predict(self):
        model = build_retinanet_efficientnet_bifpn_backbone(self.cfg)

        image_path = R"D:\dataset\VOC\VOC2007\JPEGImages\000015.jpg"
        image = read_image(image_path, 512)
        with torch.no_grad():
            output_features = model(image)
        for k, output_feature in output_features.items():
            print(F"{k}, {output_feature.shape}, {torch.sum(output_feature).item()}")

    def test_output_shape(self):
        model = build_retinanet_efficientnet_bifpn_backbone(self.cfg)
        print(F"output shape: {model.output_shape()}")


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
    suite.addTest(parameterizedTestCase.parameterize(TestBiFPN, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

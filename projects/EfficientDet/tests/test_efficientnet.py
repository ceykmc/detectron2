# -*- coding: utf-8 -*-

import argparse
import unittest

from PIL import Image
import torch
from torchvision import transforms

from modeling.backbone.efficientnet import EfficientNet
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


class TestEfficientNet(parameterizedTestCase):
    def test_create_model(self):
        model = EfficientNet(self.cfg)
        check_point_file_path = \
            R"C:\Users\xzy_lijun\.cache\torch\checkpoints\efficientnet-b0-355c32eb.pth"
        state_dict = torch.load(check_point_file_path)
        model.load_state_dict(state_dict, strict=False)

    def test_predict(self):
        model = EfficientNet(self.cfg)
        check_point_file_path = \
            R"C:\Users\xzy_lijun\.cache\torch\checkpoints\efficientnet-b0-355c32eb.pth"
        state_dict = torch.load(check_point_file_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        image_path = R"\\10.20.0.60\public_workspace\ImageNet\val\ILSVRC2012_val_00001981.JPEG"
        image = read_image(image_path, 512)
        torch.save(image, "test_image.pth")
        print(torch.sum(image))
        with torch.no_grad():
            output_features = model(image)
        for k, output_feature in output_features.items():
            print(F"{k}, {output_feature.shape}, {torch.sum(output_feature).item()}")


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
    suite.addTest(parameterizedTestCase.parameterize(TestEfficientNet, param=param))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import argparse
import logging

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.utils.analysis import flop_count_operators, parameter_count, parameter_count_table

from config import add_config
from modeling import build_retinanet_efficientnet_bifpn_backbone

logging.basicConfig(level=logging.INFO)


def setup_cfg(config_file_path: str,
              model_file_path: str,
              score_thresh: float = 0.5):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--model-file", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file_path = args.config_file
    model_file_path = args.model_file
    cfg = setup_cfg(config_file_path, model_file_path, score_thresh=0.5)

    model = build_model(cfg)
    # print(parameter_count_table(model))
    # for k, v in parameter_count(model).items():
    #     print(k, v)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    for step, batched_inputs in enumerate(data_loader):
        res = flop_count_operators(model, batched_inputs)
        print(res)
        break


if __name__ == "__main__":
    main()

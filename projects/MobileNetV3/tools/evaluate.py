# -*- coding: utf-8 -*-

import argparse
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import PascalVOCDetectionEvaluator

from config import add_mobilenetv3_config
from modeling import build_mobilenetv3_fpn_backbone

logging.basicConfig(level=logging.INFO)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator(dataset_name)


def setup_cfg(config_file_path: str,
              model_file_path: str,
              score_thresh: float = 0.5):
    cfg = get_cfg()
    add_mobilenetv3_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--model_file", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file_path = args.config_file
    model_file_path = args.model_file
    cfg = setup_cfg(config_file_path, model_file_path, score_thresh=0.05)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True)
    Trainer.test(cfg, model)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import argparse
import logging

import distiller
from tabulate import tabulate
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader

from config import add_config
from modeling import build_model

logging.basicConfig(level=logging.INFO)


def setup_cfg(config_file_path: str):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file_path = args.config_file
    cfg = setup_cfg(config_file_path)

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    batched_input = next(iter(data_loader))
    df = distiller.model_performance_summary(model, batched_input, 1)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    total_macs = df['MACs'].sum()

    with open("./output/model_summary.txt", mode="w") as summary_file:
        summary_file.write(t)
        summary_file.writelines("\nTotal MACs: " + "{:,}".format(total_macs))


if __name__ == "__main__":
    main()

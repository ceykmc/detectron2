# -*- coding: utf-8 -*-

import logging
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch, default_setup

from config import add_config
from trainer import EfficientDetTrainer

logging.basicConfig(level=logging.INFO)


def setup(config_file_path: str, args):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train(cfg, args):
    trainer = EfficientDetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = default_argument_parser().parse_args()
    config_file_path = args.config_file
    cfg = setup(config_file_path, args)
    if args.eval_only:
        model = EfficientDetTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        EfficientDetTrainer.test(cfg, model)
    else:
        launch(
            train,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(cfg, args),
        )


if __name__ == "__main__":
    main()

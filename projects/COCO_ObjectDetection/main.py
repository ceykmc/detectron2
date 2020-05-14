# -*- coding: utf-8 -*-

import argparse
import os
import logging

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from config import add_shufflenetv2_config
from modeling import build_retinanet_shufflenetv2_fpn_backbone

logging.basicConfig(level=logging.INFO)


def setup(args: argparse.ArgumentParser):
    cfg = get_cfg()
    add_shufflenetv2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    return cfg


def train(args: argparse.ArgumentParser):
    cfg = setup(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    args = default_argument_parser().parse_args()
    if args.eval_only:
        # torch.backends.cudnn.benchmark = True
        # config_file_path = args.config_file
        # cfg = setup(config_file_path)
        # model = PersonTrainer.build_model(cfg)
        # if cfg.MODEL.WEIGHTS:
        #     checkpointer = DetectionCheckpointer(model)
        #     checkpointer.load(cfg.MODEL.WEIGHTS)
        # PersonTrainer.test(cfg, model)
        pass
    else:
        launch(
            train,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args, ),
        )


if __name__ == "__main__":
    main()

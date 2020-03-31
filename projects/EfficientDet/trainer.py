# -*- coding: utf-8 -*-

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer

from modeling import build_model


class EfficientDetTrainer(DefaultTrainer):
    def run_step(self):
        print(F"iter: {self.iter}")
        return super().run_step()

    @classmethod
    def build_model(cls, cfg: CfgNode):
        return build_model(cfg)

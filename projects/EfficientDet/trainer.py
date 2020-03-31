# -*- coding: utf-8 -*-

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer

from modeling import build_model
from utils import EMA


class EfficientDetTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_hooks(self):
        ret = super().build_hooks()
        decay = self.cfg.SOLVER.EMA.DECAY
        period = self.cfg.SOLVER.EMA.PERIOD
        ema_hook = EMA(model=self.model, decay=decay, period=period)
        ret.insert(0, ema_hook)  # 需要先将参数值修改成ema值，再进行保存
        return ret

    @classmethod
    def build_model(cls, cfg: CfgNode):
        return build_model(cfg)

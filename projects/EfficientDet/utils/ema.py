# -*- coding: utf-8 -*-

import torch.nn as nn

from detectron2.engine import HookBase


class EMA(HookBase):
    def __init__(self, model: nn.Module, decay: float, period: int):
        super().__init__()
        self.model = model
        self.decay = decay
        self.period = period
        self.shadow = {}
        self.backup = {}

    def before_train(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        return super().before_train()

    def after_train(self):
        return super().after_train()

    def before_step(self):
        return super().before_step()

    def after_step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        if self.trainer.iter >= self.trainer.max_iter - 1:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data
                    param.data = self.shadow[name]
        return super().after_step()

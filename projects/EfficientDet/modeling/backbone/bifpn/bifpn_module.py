# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d


class FastNormalizedFusion(nn.Module):
    def __init__(self,
                 levels: int,
                 init_value: float = 0.5,
                 eps: float = 0.0001):
        super().__init__()
        self.top_down_weights_p = nn.Parameter(
            data=torch.Tensor(levels, 2).fill_(init_value))
        self.bottom_up_weights_p = nn.Parameter(
            data=torch.Tensor(levels - 2, 3).fill_(init_value))
        self.eps = eps
        self.top_down_weights = None
        self.bottom_up_weights = None

    def weight_normalize(self):
        self.top_down_weights = F.relu(self.top_down_weights_p)
        self.bottom_up_weights = F.relu(self.bottom_up_weights_p)
        self.top_down_weights /= torch.unsqueeze(
            torch.sum(self.top_down_weights, dim=1), dim=1) + self.eps
        self.bottom_up_weights /= torch.unsqueeze(
            torch.sum(self.bottom_up_weights, dim=1), dim=1) + self.eps

    def top_down_process(self, level: int, up_level_feature: torch.Tensor,
                         current_level_feature: torch.Tensor):
        a = self.top_down_weights[level, 0] * current_level_feature
        b = self.top_down_weights[level, 1] * \
            F.interpolate(up_level_feature, scale_factor=2, mode="nearest")
        return a + b

    def bottom_up_process(self, level: int, down_level_feature: torch.Tensor,
                          current_level_origin_feature: torch.Tensor,
                          current_level_inter_feature: torch.Tensor):
        a = self.bottom_up_weights[level - 1, 0] * current_level_origin_feature
        b = self.bottom_up_weights[level - 1, 1] * current_level_inter_feature
        c = self.bottom_up_weights[level - 1, 2] * \
            F.max_pool2d(down_level_feature, kernel_size=2)
        return a + b + c

    def bottom_up_process_top_level(self, level: int, down_level_feature: torch.Tensor,
                                    current_level_feature: torch.Tensor):
        a = self.top_down_weights[level, 0] * current_level_feature
        b = self.top_down_weights[level, 1] * \
            F.max_pool2d(down_level_feature, kernel_size=2)
        return a + b


class BiFPNModule(nn.Module):
    def __init__(self, levels, channels):
        super().__init__()
        self.levels = levels
        self.fusion = FastNormalizedFusion(levels=levels)
        self.convs = nn.ModuleList()

        # there are total 2 * (levels - 1) convolutions in BiFPNModule
        for _ in range(2 * (levels - 1)):
            self.convs.append(
                Conv2d(in_channels=channels,
                       out_channels=channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       groups=channels,
                       norm=nn.BatchNorm2d(num_features=channels),
                       activation=nn.ReLU(inplace=True)))

    def forward(self, inputs):
        self.fusion.weight_normalize()
        inputs_clone = [e.clone() for e in inputs]

        count = 0
        for i in range(self.levels - 2, -1, -1):
            inputs[i] = self.fusion.top_down_process(
                level=i,
                up_level_feature=inputs[i + 1],
                current_level_feature=inputs[i])
            inputs[i] = self.convs[count](inputs[i])
            count += 1
        for i in range(1, self.levels - 1, 1):
            inputs[i] = self.fusion.bottom_up_process(
                level=i,
                down_level_feature=inputs[i - 1],
                current_level_origin_feature=inputs_clone[i],
                current_level_inter_feature=inputs[i])
            inputs[i] = self.convs[count](inputs[i])
            count += 1
        inputs[self.levels - 1] = self.fusion.bottom_up_process_top_level(
            level=self.levels - 1,
            down_level_feature=inputs[self.levels - 2],
            current_level_feature=inputs[self.levels - 1])
        inputs[self.levels - 1] = self.convs[count](inputs[self.levels - 1])
        assert count == len(self.convs) - 1, "not all conv module are used"
        return inputs


def main():
    model = BiFPNModule(levels=5, channels=32)

    inputs = [torch.randn(4, 32, 64, 64),
              torch.randn(4, 32, 32, 32),
              torch.randn(4, 32, 16, 16),
              torch.randn(4, 32, 8, 8),
              torch.randn(4, 32, 4, 4)]
    outputs = model(inputs)
    for i, output in enumerate(outputs):
        print(i, output.size())


if __name__ == "__main__":
    main()

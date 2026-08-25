"""最小线性回归模型。"""

import torch
from torch import nn


class LinearRegression(nn.Module):
    """学习公式 y_hat = weight * x + bias。"""

    def __init__(self) -> None:
        super().__init__()
        # 输入特征数为 1，输出值数也为 1；weight 和 bias 会在训练中自动学习。
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

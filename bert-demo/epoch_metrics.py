"""训练和验证共用的整轮指标。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    """一轮的样本平均损失和准确率；准确率范围为 0 到 1。"""

    loss: float
    accuracy: float

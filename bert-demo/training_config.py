"""训练参数及本地资源路径配置。"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

# 根据脚本所在目录定位资源，避免 PyCharm 工作目录不同导致路径失效。
PROJECT_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    """集中保存训练配置；frozen=True 表示创建后不允许直接修改字段。"""

    # 模型和分词器共用一个本地目录，由下载脚本提前准备。
    model_path: Path = PROJECT_DIRECTORY / "model" / "bert-base-chinese"
    # 保存训练得到的分类层参数，不覆盖原始 BERT 权重。
    checkpoint_directory: Path = PROJECT_DIRECTORY / "params"
    # 完整遍历一次训练集称为一轮；默认保留原代码的 30000 轮，可通过 --epochs 调整。
    epochs: int = 30000
    # 每次前向计算处理的评论数量，越大通常需要越多内存或显存。
    batch_size: int = 50
    # 编码后包含特殊标记的最大长度；超长截断，短句补齐。
    maximum_length: int = 512
    # 控制优化器每次更新参数的步长，仅作用于可训练的分类层。
    learning_rate: float = 1e-3
    # 每隔指定数量的批次输出训练日志。
    log_interval: int = 5
    # 训练损失下降且验证损失上升连续达到此轮数，才判断出现过拟合趋势。
    patience: int = 3
    # 相邻轮次损失变化必须超过此阈值；默认 0，严格比较升降方向。
    minimum_loss_change: float = 0.0

    @staticmethod
    def parse_arguments() -> "TrainingConfig":
        """允许从启动参数调整轮数和批大小，默认保留原训练设置。"""
        parser = argparse.ArgumentParser(description="使用本地中文 BERT 和评论数据集训练、验证二分类模型。")
        parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
        parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
        parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
        parser.add_argument("--minimum-loss-change", type=float, default=TrainingConfig.minimum_loss_change)
        arguments = parser.parse_args()
        # 在加载模型和数据之前检查输入，避免无效参数启动训练。
        if arguments.epochs <= 0 or arguments.batch_size <= 0:
            parser.error("epochs 和 batch-size 必须为正整数")
        if arguments.patience <= 0:
            parser.error("patience 必须为正整数")
        if not math.isfinite(arguments.minimum_loss_change) or arguments.minimum_loss_change < 0:
            parser.error("minimum-loss-change 必须为有限的非负数")
        return TrainingConfig(
            epochs=arguments.epochs, batch_size=arguments.batch_size,
            patience=arguments.patience, minimum_loss_change=arguments.minimum_loss_change,
        )

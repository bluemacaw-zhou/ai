"""入口：使用训练好的分类层评估独立测试集，不更新模型参数。

用法：
    python evaluate.py                                    # 评估 params/ 下全部 best_classifier*.pth
    python evaluate.py --checkpoint params/a.pth params/b.pth  # 评估指定的一个或多个检查点
"""
import argparse
from pathlib import Path

import torch

from evaluator import TestSetEvaluator
from model_loader import TrainedModelLoader
from training_config import TrainingConfig


def main() -> None:
    """每个检查点的结果写入同名 .evaluation.json；多个检查点时额外输出对比表。"""
    parser = argparse.ArgumentParser(description="使用训练好的分类层评估独立测试集。")
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+", default=None,
        help="一个或多个分类层文件；省略时评估 params/ 下全部 best_classifier*.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        parser.error("batch-size 必须为正整数")

    checkpoints = arguments.checkpoint or sorted(
        TrainingConfig().checkpoint_directory.glob("best_classifier*.pth")
    )
    missing = [str(path) for path in checkpoints if not path.exists()]
    if not checkpoints or missing:
        parser.error(f"检查点文件不存在：{missing or 'params/best_classifier*.pth'}")

    # 本地评估固定使用 CPU，CPU 与 GPU 训练出的参数在同一环境下比较。
    # BERT 只加载一次，之后每个检查点只替换分类层；测试集 DataLoader 同样复用。
    loader = TrainedModelLoader(checkpoints[0], device=torch.device("cpu"))
    evaluator = TestSetEvaluator(loader, arguments.batch_size)
    results = []
    for checkpoint in checkpoints:
        if checkpoint != loader.checkpoint_path:
            loader.switch_checkpoint(checkpoint)
        result = evaluator.evaluate()
        TestSetEvaluator.save(result, checkpoint.with_suffix(".evaluation.json"))
        results.append(result)

    if len(results) > 1:
        TestSetEvaluator.log_comparison(results)


if __name__ == "__main__":
    main()

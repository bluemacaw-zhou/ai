"""使用本地最佳分类层参数评估独立测试集，不更新模型参数。"""

import argparse
import json
from pathlib import Path

import torch
from ai_starter import get_logger

from review_dataset import ReviewDataset
from post_processing_model import PostProcessingModel
from train import validate
from training_config import TrainingConfig

log = get_logger(__name__)


def load_checkpoint(model, checkpoint_path: Path) -> None:
    """仅恢复分类层参数；冻结的 BERT 继续从本地模型目录加载。"""
    parameters = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(parameters)
    log.info("已加载分类层：%s", checkpoint_path)


def evaluate_checkpoint(checkpoint_path: Path, batch_size: int) -> dict:
    """在未用于训练和选择参数的 test 划分上计算平均损失与准确率。"""
    config = TrainingConfig(batch_size=batch_size)
    # 本地阶段评估使用 CPU，便于与后续 Runpod 的评估结果比较。
    device = torch.device("cpu")
    tokenizer = PostProcessingModel.load_tokenizer(config.model_path)
    data_loader = ReviewDataset.create_data_loader(
        "test", tokenizer, batch_size=config.batch_size, maximum_length=config.maximum_length
    )
    model = PostProcessingModel.create_model(config.model_path, device)
    load_checkpoint(model, checkpoint_path)
    # validate 关闭梯度并使用评估模式；这里只评估，不调用训练或优化器。
    metrics = validate(model, data_loader, torch.nn.CrossEntropyLoss(), device)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": "test",
        "sample_count": len(data_loader.dataset),
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
    }
    log.info("阶段测试结果：%s", result)
    return result


def main() -> None:
    """可指定分类层文件；默认评估验证损失最低的阶段参数。"""
    config = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.checkpoint_directory / "best_classifier.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        parser.error("batch-size 必须为正整数")
    result = evaluate_checkpoint(arguments.checkpoint, arguments.batch_size)
    # 保存结果，供之后在 Runpod 训练完成后比较；测试集不参与早停决策。
    result_path = arguments.checkpoint.with_suffix(".evaluation.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    log.info("测试结果已保存：%s", result_path)


if __name__ == "__main__":
    main()

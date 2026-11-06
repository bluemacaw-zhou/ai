"""测试集评估器：在独立 test 划分上计算损失、准确率及分类指标。"""
import json
from pathlib import Path

import torch
from ai_starter import get_logger

from model_loader import TrainedModelLoader
from review_dataset import ReviewDataset
from tool import TrainingTool

log = get_logger(__name__)


class TestSetEvaluator:
    """使用已加载的模型评估未参与训练和选模型的 test 划分，不更新参数。"""

    def __init__(self, loader: TrainedModelLoader, batch_size: int):
        self._loader = loader
        self._data_loader = ReviewDataset.create_data_loader(
            "test", loader.tokenizer, batch_size=batch_size, maximum_length=loader.config.maximum_length
        )

    def evaluate(self) -> dict:
        """遍历一次测试集，返回损失、准确率、各类别与宏平均指标及混淆矩阵。"""
        labels, predictions, loss = self._collect_predictions()
        result = {
            "checkpoint": str(self._loader.checkpoint_path.resolve()),
            "split": "test",
            "sample_count": len(labels),
            "loss": loss,
            "accuracy": (labels == predictions).float().mean().item(),
            **TestSetEvaluator._compute_class_metrics(labels, predictions),
        }
        TestSetEvaluator._log_result(result)
        return result

    @staticmethod
    def save(result: dict, result_path: Path) -> None:
        """以 JSON 保存评估结果，便于和 Runpod 训练结果比较。"""
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info("测试结果已保存：%s", result_path)

    @staticmethod
    def log_comparison(results: list[dict]) -> None:
        """多个检查点评估完成后，按同一测试集逐行对比主要指标。"""
        log.info("多检查点测试集对比（样本=%s）：", results[0]["sample_count"])
        log.info("%-32s %8s %9s %9s  %s", "checkpoint", "loss", "accuracy", "macro_f1", "confusion_matrix")
        for result in results:
            log.info(
                "%-32s %8.4f %9.4f %9.4f  %s",
                Path(result["checkpoint"]).name, result["loss"], result["accuracy"],
                result["macro"]["f1"], result["confusion_matrix"],
            )

    @torch.no_grad()
    def _collect_predictions(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """返回全部真实标签、预测标签（CPU 上）和样本平均损失。"""
        model, device = self._loader.model, self._loader.device
        model.eval()
        loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        all_labels, all_predictions = [], []
        for batch in self._data_loader:
            inputs, labels = TrainingTool.move_batch_to_device(batch, device)
            logits = model(**inputs)
            total_loss += loss_function(logits, labels).item()
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
        labels = torch.cat(all_labels)
        return labels, torch.cat(all_predictions), total_loss / len(labels)

    @staticmethod
    def _compute_class_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
        """由混淆矩阵计算各类别及宏平均的精确率、召回率和 F1。"""
        label_names = TrainedModelLoader.LABEL_NAMES
        class_count = len(label_names)
        # 行为真实类别，列为预测类别。
        confusion = torch.bincount(labels * class_count + predictions, minlength=class_count ** 2)
        confusion = confusion.reshape(class_count, class_count)

        per_class = {}
        for index, name in enumerate(label_names):
            true_positive = confusion[index, index].item()
            predicted_count = confusion[:, index].sum().item()
            actual_count = confusion[index, :].sum().item()
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / actual_count if actual_count else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": actual_count}

        macro = {
            key: sum(metrics[key] for metrics in per_class.values()) / class_count
            for key in ("precision", "recall", "f1")
        }
        return {"per_class": per_class, "macro": macro, "confusion_matrix": confusion.tolist()}

    @staticmethod
    def _log_result(result: dict) -> None:
        log.info(
            "测试集结果：样本=%s loss=%.4f accuracy=%.4f macro_f1=%.4f 混淆矩阵=%s",
            result["sample_count"], result["loss"], result["accuracy"],
            result["macro"]["f1"], result["confusion_matrix"],
        )
        for name, class_metrics in result["per_class"].items():
            log.info(
                "%s：precision=%.4f recall=%.4f f1=%.4f support=%s",
                name, class_metrics["precision"], class_metrics["recall"],
                class_metrics["f1"], class_metrics["support"],
            )

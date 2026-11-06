"""加载训练好的分类模型，供测试集评估和单条语句预测共用。"""
from dataclasses import dataclass
from pathlib import Path

import torch
from ai_starter import get_logger
from transformers import BertTokenizer

from post_processing_model import PostProcessingModel
from tool import TrainingTool
from training_config import TrainingConfig

log = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """单条语句的预测结果；confidence 是预测类别的 softmax 概率。"""

    text: str
    label: int
    label_name: str
    confidence: float


class TrainedModelLoader:
    """加载本地 BERT、分词器和训练好的分类层参数，加载后始终处于评估模式。"""

    # 类别编号与 ChnSentiCorp 标签一致：0 为负面评论，1 为正面评论。
    LABEL_NAMES: tuple[str, ...] = ("负面评价", "正面评价")

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: torch.device | None = None,
        config: TrainingConfig | None = None,
    ):
        self._config = config or TrainingConfig()
        self._checkpoint_path = checkpoint_path or self._config.checkpoint_directory / "best_classifier.pth"
        self._device = device or TrainingTool.select_device()
        self._tokenizer = PostProcessingModel.load_tokenizer(self._config.model_path)
        self._model = PostProcessingModel.create_model(self._config.model_path, self._device)
        TrainingTool.load_checkpoint(self._model, self._checkpoint_path, self._device)

    @property
    def config(self) -> TrainingConfig:
        return self._config

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self) -> BertTokenizer:
        return self._tokenizer

    @property
    def model(self) -> PostProcessingModel:
        return self._model

    def switch_checkpoint(self, checkpoint_path: Path) -> None:
        """只替换分类层参数；BERT 与分词器复用已加载的实例，避免重复加载约 400 MB 权重。"""
        TrainingTool.load_checkpoint(self._model, checkpoint_path, self._device)
        self._checkpoint_path = checkpoint_path

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[Prediction]:
        """对一组语句做情感判断；只做前向计算，不更新参数。"""
        # 推理时无需补齐到 512，按本批最长语句动态补齐即可，结果与训练时一致。
        encoded_inputs = self._tokenizer(
            texts, truncation=True, max_length=self._config.maximum_length,
            padding=True, return_tensors="pt",
        )
        inputs = {name: tensor.to(self._device) for name, tensor in encoded_inputs.items()}
        probabilities = torch.softmax(self._model(**inputs), dim=1)
        confidences, labels = probabilities.max(dim=1)
        return [
            Prediction(text, label, self.LABEL_NAMES[label], confidence)
            for text, label, confidence in zip(texts, labels.tolist(), confidences.tolist())
        ]

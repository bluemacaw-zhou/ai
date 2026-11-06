"""训练、验证和测试共用的工具方法。"""
from pathlib import Path

import torch
from ai_starter import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from epoch_metrics import EpochMetrics
from post_processing_model import PostProcessingModel

log = get_logger(__name__)


class TrainingTool:
    """无状态的公共方法集合，全部为静态方法，直接通过类名调用。"""

    @staticmethod
    def select_device() -> torch.device:
        """优先使用 NVIDIA CUDA；没有 CUDA 时使用 CPU。"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("计算设备：%s", device)
        return device

    @staticmethod
    def create_optimizer(model: PostProcessingModel, learning_rate: float) -> AdamW:
        """只将 requires_grad=True 的参数交给优化器，排除冻结的 BERT。"""
        # 显式保留旧 Transformers AdamW 的 eps 和权重衰减默认值。
        return AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate, eps=1e-6, weight_decay=0.0,
        )

    @staticmethod
    def move_batch_to_device(
        batch: tuple[dict[str, torch.Tensor], torch.Tensor], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """输入张量、标签和模型必须位于同一个设备才能参与计算。"""
        inputs, labels = batch
        return {name: tensor.to(device) for name, tensor in inputs.items()}, labels.to(device)

    @staticmethod
    def train_one_epoch(
        model: PostProcessingModel, data_loader: DataLoader, optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module, device: torch.device, epoch: int, log_interval: int,
    ) -> EpochMetrics:
        """遍历训练集一次，完成前向计算、损失计算、反向传播和参数更新。"""
        # 上一轮验证会切换到 eval；每轮开始必须重新开启分类层的训练模式。
        # PostProcessingModel.train() 会让冻结的 BERT 继续保持评估模式。
        model.train()

        # 累计损失总和、预测正确数量及样本数，用于计算整轮指标。
        total_loss = 0.0
        correct_predictions = 0
        sample_count = 0
        for batch_index, batch in enumerate(data_loader):
            inputs, labels = TrainingTool.move_batch_to_device(batch, device)
            # 清除上一批梯度，避免 PyTorch 默认的梯度累加影响本批更新。
            optimizer.zero_grad(set_to_none=True)

            # 输出形状为 [批大小, 2]；交叉熵内部处理 logits，不需要提前调用 softmax。
            logits = model(**inputs)
            loss = loss_function(logits, labels)

            # 反向传播计算分类层梯度，再由优化器更新参数。
            loss.backward()
            optimizer.step()

            # 默认损失是批内平均值，乘以实际批大小还原损失总和；最后一批可能更小。
            batch_size = labels.size(0)
            batch_correct = (logits.argmax(dim=1) == labels).sum().item()
            total_loss += loss.item() * batch_size
            correct_predictions += batch_correct
            sample_count += batch_size

            if batch_index % log_interval == 0:
                log.info(
                    "epoch=%s batch=%s/%s loss=%.4f accuracy=%.4f",
                    epoch, batch_index + 1, len(data_loader), loss.item(), batch_correct / batch_size,
                )

        # 准确率必须除以样本总数，不能除以批次数。
        return EpochMetrics(total_loss / sample_count, correct_predictions / sample_count)

    @staticmethod
    @torch.no_grad()
    def validate(
        model: PostProcessingModel, data_loader: DataLoader,
        loss_function: torch.nn.Module, device: torch.device,
    ) -> EpochMetrics:
        """在完整数据划分上计算平均损失与准确率；不反向传播、不更新参数。"""
        # eval 控制 dropout 等层的行为，与 no_grad 的作用不同，两者都需要。
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        sample_count = 0
        for batch in data_loader:
            inputs, labels = TrainingTool.move_batch_to_device(batch, device)
            logits = model(**inputs)
            batch_size = labels.size(0)
            # 与训练相同，按实际样本数加权累计损失和准确率。
            total_loss += loss_function(logits, labels).item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

        return EpochMetrics(total_loss / sample_count, correct_predictions / sample_count)

    @staticmethod
    def save_checkpoint(model: PostProcessingModel, directory: Path, filename: str) -> None:
        """BERT 固定不训练，只保存分类层，避免每轮重复写入整个 BERT。"""
        # 此文件只含分类层参数，不包含优化器状态，不能完整恢复中断时的训练状态。
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / filename
        torch.save(model.classifier.state_dict(), checkpoint_path)
        log.info("分类层参数已保存：%s", checkpoint_path)

    @staticmethod
    def load_checkpoint(model: PostProcessingModel, checkpoint_path: Path, device: torch.device) -> None:
        """仅恢复分类层参数并切换到评估模式；冻结的 BERT 继续从本地模型目录加载。"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"分类层参数文件不存在：{checkpoint_path}")
        parameters = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.classifier.load_state_dict(parameters)
        model.eval()
        log.info("已加载分类层：%s", checkpoint_path)

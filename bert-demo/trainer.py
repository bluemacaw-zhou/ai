"""训练器：组织一次完整的训练、验证、保存和早停流程。"""
import math

import torch
from ai_starter import get_logger

from epoch_metrics import EpochMetrics
from overfitting_monitor import OverfittingMonitor
from post_processing_model import PostProcessingModel
from review_dataset import ReviewDataset
from tool import TrainingTool
from training_config import TrainingConfig

log = get_logger(__name__)


class Trainer:
    """准备资源 → 逐轮训练与验证 → 保存参数 → 判断停止 → 恢复最佳参数。

    训练的是新建的二分类层，BERT 仅提取特征；不使用独立测试集。
    结果通过检查点文件保留，不从已有检查点自动续训。
    """

    def __init__(self, config: TrainingConfig):
        """按配置准备设备、分词器、数据、模型、优化器、损失函数和过拟合监控器。"""
        self._config = config

        # 1. 确定计算设备：有 NVIDIA CUDA 则使用 GPU，否则使用 CPU。
        # 后续模型参数、输入张量和真实标签必须在同一个设备上才能计算。
        self._device = TrainingTool.select_device()

        # 2. 从 config.model_path 加载本地词表和分词器配置，不联网下载。
        # 分词器负责把评论文本变成词元编号、有效位置掩码和句段编号，不参与参数训练。
        tokenizer = PostProcessingModel.load_tokenizer(config.model_path)

        # 3. 准备两份可按批次遍历的数据：train 用于参数更新，validation 用于评估和选模型。
        # DataLoader 每次取 config.batch_size 条评论，并在取出时调用 ReviewDataset.encode_batch 静态方法编码。
        # 一批返回 (inputs, labels)：inputs 是模型输入张量字典，labels 是 0/1 真实类别。
        # 训练集每轮打乱，验证集不打乱；两者都保留最后不足一批的样本。
        # 此处不加载 test，避免让独立测试结果参与每轮调参和停止判断。
        self._training_loader = ReviewDataset.create_data_loader(
            "train", tokenizer, batch_size=config.batch_size, maximum_length=config.maximum_length
        )
        self._validation_loader = ReviewDataset.create_data_loader(
            "validation", tokenizer, batch_size=config.batch_size, maximum_length=config.maximum_length
        )

        # 4. 创建 PostProcessingModel 并移动到上述设备。
        # 它使用本地 BERT 提取每条评论的 [CLS] 特征，再通过 Linear(768, 2) 输出负面/正面分数。
        # BERT 参数冻结且保持评估模式；新建分类层的权重、偏置是本次要学习的参数。
        self._model = PostProcessingModel.create_model(config.model_path, self._device)

        # 5. 创建 AdamW，只接收 requires_grad=True 的分类层参数。
        # 优化器利用反向传播得到的梯度，按 learning_rate 控制的步长更新这些参数。
        self._optimizer = TrainingTool.create_optimizer(self._model, config.learning_rate)

        # 6. 定义交叉熵损失：输入是每条评论的两个原始分数，以及它的真实类别编号。
        # 不需要先调用 softmax；损失函数内部完成数值稳定的计算，默认返回本批平均损失。
        # 给真实类别的概率越低，损失越大；它不等于错误率，准确率另行统计。
        self._loss_function = torch.nn.CrossEntropyLoss()

        # 7. 初始化本次训练的最佳记录，不读取历史 best 文件中的成绩。
        # 正无穷保证第一轮有限的验证损失一定更小，从而写入本次训练的首个最佳检查点。
        # best_epoch 记录最佳参数来自哪一轮；不是已训练轮数，也不是剩余轮数。
        # 用验证损失选择最优模型，比离散的准确率更容易观察概率预测的细微变化。
        self._best_validation_loss = math.inf
        self._best_epoch = 0

        # 8. 创建跨轮次保留状态的过拟合监控器，不能在每轮重新创建。
        # 它记录上一轮的训练/验证损失、是否曾出现验证损失下降，以及连续满足趋势的次数。
        # patience 默认 3；minimum_loss_change 默认 0，表示严格比较相邻轮次的升降方向。
        self._overfitting_monitor = OverfittingMonitor(config.patience, config.minimum_loss_change)

    @property
    def config(self) -> TrainingConfig:
        return self._config

    @property
    def model(self) -> PostProcessingModel:
        return self._model

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def best_validation_loss(self) -> float:
        return self._best_validation_loss

    def run(self) -> None:
        """逐轮训练直至达到最大轮数或检测到持续过拟合，最后恢复最佳分类层。"""
        # 9. epoch 从 1 开始计数，config.epochs 只是最大上限。
        # 一轮会完整遍历训练集一次，再完整评估验证集一次；满足停止条件可以提前退出。
        for epoch in range(1, self._config.epochs + 1):
            if self._run_epoch(epoch):
                break

        # 10. 早停或达到最大轮数后，都从 best_classifier.pth 恢复本次训练的最佳分类层并设为 eval。
        # 内存中的模型因此回到验证损失最低时的参数；磁盘 last 文件仍保留最后完成轮次的参数。
        # 这里不自动评估测试集；独立测试由 evaluate.py 通过 TrainedModelLoader 加载 best 参数完成。
        # 本段仅在正常到达此处时执行；外部强制终止进程不会保证执行，但已保存文件仍可加载。
        TrainingTool.load_checkpoint(
            self._model, self._config.checkpoint_directory / "best_classifier.pth", self._device
        )
        log.info("已恢复验证损失最低的分类层（第 %s 轮）", self._best_epoch)

    def _run_epoch(self, epoch: int) -> bool:
        """训练并验证一轮，保存检查点；返回 True 表示应停止训练。"""
        # 9.1 训练一轮：开启分类层训练模式，逐批清梯度、前向计算、求损失、反向传播和更新参数。
        # 训练指标是在参数边更新时累计的，不是用最终参数重新评估一次训练集的结果。
        training_metrics = TrainingTool.train_one_epoch(
            self._model, self._training_loader, self._optimizer, self._loss_function,
            self._device, epoch, self._config.log_interval,
        )

        # 9.2 使用本轮更新后的同一份模型参数评估完整验证集。
        # validate 切换到 eval 并关闭梯度；下一轮训练会重新切换模式。
        validation_metrics = TrainingTool.validate(
            self._model, self._validation_loader, self._loss_function, self._device
        )

        # 9.3 检查计算是否有效。NaN/Inf 不能用于比较最佳损失或判断上升下降。
        # 出现异常立即报错，不用本轮参数覆盖此前可用的检查点。
        if not math.isfinite(training_metrics.loss) or not math.isfinite(validation_metrics.loss):
            raise RuntimeError("训练或验证损失出现 NaN/Inf，已终止训练；请检查数据和学习率")

        # 9.4 输出本轮汇总。accuracy=0.9 表示 90%，loss 则不是百分比。
        log.info(
            "epoch=%s/%s train_loss=%.4f train_accuracy=%.4f validation_loss=%.4f validation_accuracy=%.4f",
            epoch, self._config.epochs, training_metrics.loss, training_metrics.accuracy,
            validation_metrics.loss, validation_metrics.accuracy,
        )

        self._save_checkpoints(epoch, validation_metrics)
        return self._should_stop(epoch, training_metrics, validation_metrics)

    def _save_checkpoints(self, epoch: int, validation_metrics: EpochMetrics) -> None:
        """验证损失创新低时覆盖 best；每轮都覆盖 last。"""
        # 9.5 始终保留验证损失最低时的参数，以便发现拐点后回退；这里不以准确率选最佳。
        # 检查点只包含分类层权重和偏置，BERT 继续使用原本地文件，不重复保存约 400 MB 权重。
        directory = self._config.checkpoint_directory
        if validation_metrics.loss < self._best_validation_loss:
            self._best_validation_loss = validation_metrics.loss
            self._best_epoch = epoch
            TrainingTool.save_checkpoint(self._model, directory, "best_classifier.pth")

        # 9.6 last 和 best 可能来自不同轮次；两个文件都不包含优化器状态，不能完整恢复训练现场。
        TrainingTool.save_checkpoint(self._model, directory, "last_classifier.pth")

    def _should_stop(
        self, epoch: int, training_metrics: EpochMetrics, validation_metrics: EpochMetrics
    ) -> bool:
        """交给过拟合监控器联合比较相邻两轮的训练与验证损失。"""
        # 9.7 必须先观察到验证损失下降，再连续 patience 轮同时满足：训练损失下降、验证损失上升。
        # 变化幅度须超过 minimum_loss_change；一轮不满足就清零，第一轮只建立比较基准。
        # 停滞或单独的验证损失上升不会触发停止。这是趋势规则，不保证完全避免过拟合。
        stop_training = self._overfitting_monitor.should_stop(
            training_metrics.loss, validation_metrics.loss
        )
        # 例如 2/3 表示连续两轮符合条件，下一轮仍符合才停止；不是累计出现两次。
        log.info(
            "过拟合监控：连续训练损失下降且验证损失上升=%s/%s，最低验证损失=%.6f（第 %s 轮）",
            self._overfitting_monitor.consecutive_overfitting_epochs, self._config.patience,
            self._best_validation_loss, self._best_epoch,
        )

        # 9.8 达到趋势阈值时停止；本轮的 best/last 保存已经完成，不会跳过本轮保存。
        if stop_training:
            log.info(
                "检测到持续过拟合趋势，结束训练：epoch=%s，损失变化阈值=%s",
                epoch, self._config.minimum_loss_change,
            )
        return stop_training

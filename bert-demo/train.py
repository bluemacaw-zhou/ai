"""使用本地中文 BERT 和评论数据集训练、验证二分类模型。"""
import math
from pathlib import Path

import torch
from ai_starter import get_logger
from torch.optim import AdamW

from epoch_metrics import EpochMetrics
from overfitting_monitor import OverfittingMonitor
from training_config import TrainingConfig
from post_processing_model import PostProcessingModel
from review_dataset import ReviewDataset

log = get_logger(__name__)


def select_device() -> torch.device:
    """优先使用 NVIDIA CUDA；没有 CUDA 时使用 CPU。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("训练设备：%s", device)
    return device


def create_optimizer(model: PostProcessingModel, learning_rate: float) -> AdamW:
    """只将 requires_grad=True 的参数交给优化器，排除冻结的 BERT。"""
    # 显式保留旧 Transformers AdamW 的 eps 和权重衰减默认值。
    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate, eps=1e-6, weight_decay=0.0,
    )


def move_batch_to_device(batch, device: torch.device):
    """输入张量、标签和模型必须位于同一个设备才能参与计算。"""
    inputs, labels = batch
    return {name: tensor.to(device) for name, tensor in inputs.items()}, labels.to(device)


def train_one_epoch(model, data_loader, optimizer, loss_function, device, epoch, log_interval) -> EpochMetrics:
    """遍历训练集一次，完成前向计算、损失计算、反向传播和参数更新。"""

    # 上一轮验证会切换到 eval；每轮开始必须重新开启分类层的训练模式。
    # PostProcessingModel.train() 会让冻结的 BERT 继续保持评估模式。
    model.train()

    # 累计损失总和、预测正确数量及样本数，用于计算整轮指标。
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0
    for batch_index, batch in enumerate(data_loader):
        inputs, labels = move_batch_to_device(batch, device)
        # 清除上一批梯度，避免 PyTorch 默认的梯度累加影响本批更新。
        optimizer.zero_grad(set_to_none=True)

        # 输出形状为 [批大小, 2]，每条评论对应两个类别的原始分数。
        logits = model(**inputs)

        # 交叉熵内部处理 logits，不需要提前调用 softmax。
        loss = loss_function(logits, labels)

        # 反向传播计算分类层梯度，再由优化器更新参数。
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)

        # argmax 取分数最大的类别，与真实标签比较得到预测正确数量。
        batch_correct = (logits.argmax(dim=1) == labels).sum().item()

        # 默认损失是批内平均值，乘以实际批大小还原损失总和。
        # 最后一批可能更小，因此不能简单对各批损失平均。
        total_loss += loss.item() * batch_size
        correct_predictions += batch_correct
        sample_count += batch_size

        # 按批次输出日志，观察损失和准确率随训练的变化。
        if batch_index % log_interval == 0:
            log.info(
                "epoch=%s batch=%s/%s loss=%.4f accuracy=%.4f",
                epoch, batch_index + 1, len(data_loader), loss.item(), batch_correct / batch_size,
            )

    # 准确率必须除以样本总数，不能除以批次数。
    return EpochMetrics(total_loss / sample_count, correct_predictions / sample_count)


@torch.no_grad()
def validate(model, data_loader, loss_function, device) -> EpochMetrics:
    """在完整验证集上计算指标；装饰器关闭梯度记录以减少内存开销。"""

    # eval 控制 dropout 等层的行为，与 no_grad 的作用不同，两者都需要。
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0
    for batch in data_loader:
        # 验证只执行前向计算，不清梯度、不反向传播，也不更新参数。
        inputs, labels = move_batch_to_device(batch, device)
        logits = model(**inputs)
        batch_size = labels.size(0)

        # 与训练相同，按实际样本数加权累计损失和准确率。
        total_loss += loss_function(logits, labels).item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return EpochMetrics(total_loss / sample_count, correct_predictions / sample_count)


def save_checkpoint(model, directory: Path, filename: str) -> None:
    """BERT 固定不训练，只保存分类层，避免每轮重复写入整个 BERT。"""

    # 首次保存时自动创建目录；恢复时对 model.classifier 调用 load_state_dict。
    # 此文件只含分类层参数，不包含优化器状态，不能完整恢复中断时的训练状态。
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / filename
    torch.save(model.classifier.state_dict(), checkpoint_path)
    log.info("分类层参数已保存：%s", checkpoint_path)


def restore_best_checkpoint(model, directory: Path, device: torch.device) -> None:
    """结束训练后恢复验证损失最低的分类层，而非保留最后一轮参数。"""
    checkpoint_path = directory / "best_classifier.pth"
    parameters = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.classifier.load_state_dict(parameters)
    model.eval()
    log.info("已恢复验证损失最低的分类层：%s", checkpoint_path)


def run_training(config: TrainingConfig) -> None:
    """组织一次训练：准备资源 → 逐轮训练与验证 → 保存参数 → 判断停止 → 恢复最佳参数。

    config 提供本地模型路径、参数保存目录、最大轮数、批大小、学习率及停止阈值。
    本方法训练的是新建的二分类层，BERT 仅提取特征；不使用独立测试集。
    结果通过检查点文件保留，本方法不返回模型，也不从已有检查点自动续训。
    """

    # 1. 确定计算设备：有 NVIDIA CUDA 则使用 GPU，否则使用 CPU。
    # 后续模型参数、输入张量和真实标签必须在同一个设备上才能计算。
    device = select_device()

    # 2. 从 config.model_path 加载本地词表和分词器配置，不联网下载。
    # 分词器负责把评论文本变成词元编号、有效位置掩码和句段编号，不参与参数训练。
    tokenizer = PostProcessingModel.load_tokenizer(config.model_path)

    # 3. 准备两份可按批次遍历的数据：train 用于参数更新，validation 用于评估和选模型。
    # DataLoader 每次取 config.batch_size 条评论，并在取出时调用 ReviewDataset.encode_batch 静态方法编码。
    # 一批返回 (inputs, labels)：inputs 是模型输入张量字典，labels 是 0/1 真实类别。
    # 训练集每轮打乱，验证集不打乱；两者都保留最后不足一批的样本。
    # 此处不加载 test，避免让独立测试结果参与每轮调参和停止判断。
    training_loader = ReviewDataset.create_data_loader(
        "train", tokenizer, batch_size=config.batch_size, maximum_length=config.maximum_length
    )
    validation_loader = ReviewDataset.create_data_loader(
        "validation", tokenizer, batch_size=config.batch_size, maximum_length=config.maximum_length
    )

    # 4. 创建 PostProcessingModel 并移动到上述设备。
    # 它使用本地 BERT 提取每条评论的 [CLS] 特征，再通过 Linear(768, 2) 输出负面/正面分数。
    # BERT 参数冻结且保持评估模式；新建分类层的权重、偏置是本次要学习的参数。
    model = PostProcessingModel.create_model(config.model_path, device)

    # 5. 创建 AdamW，只接收 requires_grad=True 的分类层参数。
    # 优化器利用反向传播得到的梯度，按 learning_rate 控制的步长更新这些参数。
    optimizer = create_optimizer(model, config.learning_rate)

    # 6. 定义交叉熵损失：输入是每条评论的两个原始分数，以及它的真实类别编号。
    # 不需要先调用 softmax；损失函数内部完成数值稳定的计算，默认返回本批平均损失。
    # 给真实类别的概率越低，损失越大；它不等于错误率，准确率另行统计。
    loss_function = torch.nn.CrossEntropyLoss()

    # 7. 初始化本次训练的最佳记录，不读取历史 best 文件中的成绩。
    # 正无穷保证第一轮有限的验证损失一定更小，从而写入本次训练的首个最佳检查点。
    # best_epoch 记录最佳参数来自哪一轮；不是已训练轮数，也不是剩余轮数。
    # 用验证损失选择最优模型，比离散的准确率更容易观察概率预测的细微变化。
    best_validation_loss = math.inf
    best_epoch = 0

    # 8. 创建跨轮次保留状态的过拟合监控器，不能放进循环里每轮重新创建。
    # 它记录上一轮的训练/验证损失、是否曾出现验证损失下降，以及连续满足趋势的次数。
    # patience 默认 3；minimum_loss_change 默认 0，表示严格比较相邻轮次的升降方向。
    overfitting_monitor = OverfittingMonitor(config.patience, config.minimum_loss_change)

    # 9. 开始逐轮训练。epoch 从 1 开始计数，config.epochs 只是最大上限。
    # 一轮会完整遍历训练集一次，再完整评估验证集一次；满足停止条件可以提前退出。
    for epoch in range(1, config.epochs + 1):
        # 9.1 训练一轮：开启分类层训练模式，逐批清梯度、前向计算、求损失、反向传播和更新参数。
        # BERT 仍冻结。返回的 EpochMetrics 包含整轮样本平均损失和正确条数/样本数。
        # 训练指标是在参数边更新时累计的，不是用最终参数重新评估一次训练集的结果。
        training_metrics = train_one_epoch(
            model, training_loader, optimizer, loss_function, device, epoch, config.log_interval,
        )

        # 9.2 使用本轮更新后的同一份模型参数评估完整验证集。
        # validate 切换到 eval 并关闭梯度，不反向传播、不更新参数；下一轮训练会重新切换模式。
        # 返回的 loss 和 accuracy 均覆盖整份验证集，不能把它们理解为最后一个批次的指标。
        validation_metrics = validate(model, validation_loader, loss_function, device)

        # 9.3 检查计算是否有效。NaN/Inf 不能用于比较最佳损失或判断上升下降。
        # 出现异常立即报错，不用本轮参数覆盖此前可用的检查点。
        if not math.isfinite(training_metrics.loss) or not math.isfinite(validation_metrics.loss):
            raise RuntimeError("训练或验证损失出现 NaN/Inf，已终止训练；请检查数据和学习率")

        # 9.4 输出本轮汇总。accuracy=0.9 表示 90%，loss 则不是百分比。
        # 同时记录训练与验证指标，便于观察两者差距及过拟合迹象。
        log.info(
            "epoch=%s/%s train_loss=%.4f train_accuracy=%.4f validation_loss=%.4f validation_accuracy=%.4f",
            epoch, config.epochs, training_metrics.loss, training_metrics.accuracy,
            validation_metrics.loss, validation_metrics.accuracy,
        )

        # 9.5 若本轮验证损失创本次训练新低，更新最低值、对应轮数并覆盖 best_classifier.pth。
        # 始终保留验证损失最低时的参数，以便发现拐点后回退；这里不以准确率选最佳。
        # 检查点只包含分类层权重和偏置，BERT 继续使用原本地文件，不重复保存约 400 MB 权重。
        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            save_checkpoint(model, config.checkpoint_directory, "best_classifier.pth")

        # 9.6 无论本轮是否最佳，都覆盖 last_classifier.pth，保留最近完成的一轮分类层参数。
        # last 和 best 可能来自不同轮次；这两个文件都不包含优化器状态，不能完整恢复训练现场。
        save_checkpoint(model, config.checkpoint_directory, "last_classifier.pth")

        # 9.7 联合比较本轮和上一轮的两个损失，调用时同时更新监控器的历史状态。
        # 必须先观察到验证损失下降，再连续 patience 轮同时满足：训练损失下降、验证损失上升。
        # 变化幅度须超过 minimum_loss_change；一轮不满足就清零，第一轮只建立比较基准。
        # 停滞或单独的验证损失上升不会触发停止。这是趋势规则，不保证完全避免过拟合。
        stop_training = overfitting_monitor.should_stop(
            training_metrics.loss, validation_metrics.loss
        )
        # 例如 2/3 表示连续两轮符合条件，下一轮仍符合才停止；不是累计出现两次。
        log.info(
            "过拟合监控：连续训练损失下降且验证损失上升=%s/%s，最低验证损失=%.6f（第 %s 轮）",
            overfitting_monitor.consecutive_overfitting_epochs, config.patience,
            best_validation_loss, best_epoch,
        )

        # 9.8 达到趋势阈值时退出循环；本轮的 best/last 保存已经完成，不会跳过本轮保存。
        if stop_training:
            log.info("检测到持续过拟合趋势，结束训练：epoch=%s，损失变化阈值=%s", epoch, config.minimum_loss_change)
            break

    # 10. 早停或达到最大轮数后，都从 best_classifier.pth 恢复本次训练的最佳分类层并设为 eval。
    # 内存中的模型因此回到验证损失最低时的参数；磁盘 last 文件仍保留最后完成轮次的参数。
    # 这里只恢复和结束，不自动评估测试集；独立测试由 evaluate.py 加载 best 参数完成。
    # 本段仅在正常到达此处时执行；外部强制终止进程不会保证执行，但已保存文件仍可加载。
    restore_best_checkpoint(model, config.checkpoint_directory, device)


def main() -> None:
    """程序入口：解析启动参数，再按配置执行训练。"""
    run_training(TrainingConfig.parse_arguments())


# 直接运行脚本才启动训练；被其他文件导入时不会加载数据或模型。
if __name__ == "__main__":
    main()

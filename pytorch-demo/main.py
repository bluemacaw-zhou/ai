"""PyTorch 基础用例。"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import CustomDataset


def test_tensor_basics() -> None:
    """测试张量的创建、索引与切片。"""
    print("=" * 50)
    print("1. 张量基本操作示例")
    print("=" * 50)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("tensor[1, 2] =", tensor[1, 2].item())
    print("tensor[:, 1:] =\n", tensor[:, 1:])


def test_data_loading() -> None:
    """测试自定义数据集、图像转换与批量加载。"""
    print("\n" + "=" * 50)
    print("2. 数据预处理示例")
    print("=" * 50)

    num_samples = 100
    # data 对应监督学习里的 x：100 张原始 HWC 图像。
    # torchvision 对 NumPy 图像按 HWC 格式处理。
    data = np.random.randn(num_samples, 32, 32, 3).astype(np.float32)
    # targets 对应监督学习里的 y：每张图片的正确类别（0 到 9）。
    # data[i] 与 targets[i] 始终是一对输入 x 和正确答案 y。
    targets = np.random.randint(0, 10, num_samples)
    dataset = CustomDataset(data, targets)

    # Dataset 只定义“单条 (x, y) 如何按索引取出”；DataLoader 负责训练时的批量迭代。
    # batch_size=16：将多条样本自动拼为一个批次 (batch_x, batch_y)。
    # shuffle=True：每轮训练前随机打乱样本顺序，避免模型学习到原始排列规律。
    # num_workers=0：在当前主进程加载数据；数据量大、读取文件较慢时可增加工作进程。
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # DataLoader 会反复调用 dataset.__getitem__，并将 16 个 (x, y) 自动拼成一个 batch。
    # 完整训练时通常写为：for batch_x, batch_y in dataloader: ...
    batch_inputs, batch_targets = next(iter(dataloader))

    print(f"数据集大小: {len(dataset)}")
    print(f"批数据形状: 输入={batch_inputs.shape}, 标签={batch_targets.shape}")
    # 后续训练步骤的概念链路：
    # x = batch_inputs -> y_hat = model(x) -> loss = loss_function(y_hat, batch_targets)
    # 分类通常使用交叉熵损失；(y - y_hat) ** 2 的均方误差通常用于回归任务。


def main() -> None:
    """执行全部基础用例。"""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    test_tensor_basics()
    test_data_loading()


if __name__ == "__main__":
    main()

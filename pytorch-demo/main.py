"""PyTorch 基础用例。"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from datasets import CustomDataset, RegressionDataset
from models import LinearRegression


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


def test_matrix_operations() -> None:
    """测试矩阵加法、逐元素乘法和矩阵乘法。"""
    print("\n" + "=" * 50)
    print("3. 线性代数操作示例")
    print("=" * 50)

    matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

    print(f"矩阵加法:\n{matrix_a + matrix_b}")
    print(f"元素乘法:\n{matrix_a * matrix_b}")
    print(f"矩阵乘法:\n{torch.mm(matrix_a, matrix_b)}")


def test_linear_regression_training() -> None:
    """通过 Dataset 和 DataLoader 演示一个完整的线性回归训练过程。"""
    print("\n" + "=" * 50)
    print("4. 综合应用：线性回归训练")
    print("=" * 50)

    # 构造随机种子
    torch.manual_seed(42)

    # 构造监督学习数据：x 是一个特征，y 是带有随机噪声的正确目标值。
    # linspace(0, 10, 100) 在闭区间 [0, 10] 均匀产生 100 个数；初始形状是 (100,)。
    # reshape(-1, 1) 将它改为 (100, 1)：100 条样本，每条样本仅包含 1 个特征。
    # -1 表示由 PyTorch 自动推断这一维为 100，以保证元素总数不变。
    features = torch.linspace(0, 10, 100).reshape(-1, 1)
    # features: [x0, x1, x2, ..., xn]，每个 xi 是一条样本的输入特征。
    # randn_like(features) 生成同形状的噪声 [e0, e1, e2, ..., en]；每个 ei 独立服从
    # N(0, 1)，即理论平均数为 0、理论标准差为 1（一次实际生成的样本值只会接近它们）。
    # targets: [2.5*x0+1+1.5*e0, 2.5*x1+1+1.5*e1, ..., 2.5*xn+1+1.5*en]。
    # 因而最终噪声 1.5*ei 的平均数仍为 0、标准差为 1.5；数据点会围绕真实直线波动。
    targets = 2.5 * features + 1.0 + torch.randn_like(features) * 1.5

    # 三者的层级关系：
    # - 一条样本是 (features[i], targets[i])，即一对 (x, y)；
    # - features[i] 是该样本的输入特征，targets[i] 是该样本的正确目标；
    # - RegressionDataset 将全部 100 条样本组织成一个可按索引读取的数据集。
    # Dataset 按索引提供单条 (x, y)，DataLoader 负责打乱并组装为 batch。
    dataset = RegressionDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 模型接收 batch_x，输出预测值 y_hat；MSELoss 衡量 y_hat 与正确 y 的平均平方误差。
    model = LinearRegression()
    criterion = nn.MSELoss()
    # 创建优化器 更新模型的w和b
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 200
    for _ in range(epochs):
        for batch_features, batch_targets in dataloader:
            optimizer.zero_grad()  # 清除上一个 batch 保留的梯度。
            predictions = model(batch_features)  # y_hat = model(x)
            loss = criterion(predictions, batch_targets)  # loss(y_hat, y)
            loss.backward()  # 反向传播：计算参数对损失的影响。
            optimizer.step()  # 根据梯度更新 weight 和 bias。

    with torch.no_grad():
        predicted_targets = model(features)
        final_loss = criterion(predicted_targets, targets)
    print(f"最终损失: {final_loss.item():.4f}")
    print(f"学习到的权重: {model.linear.weight.item():.2f}")
    print(f"学习到的偏置: {model.linear.bias.item():.2f}")

    # 训练完成后，散点表示原始 (x, y)，直线表示模型给出的 y_hat。
    plt.scatter(features.numpy(), targets.numpy(), label="training data")
    plt.plot(features.numpy(), predicted_targets.numpy(), "r-", label="fitted line")
    plt.legend()
    plt.title("Linear regression trained with DataLoader")
    plt.show()


def main() -> None:
    """执行全部基础用例。"""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    test_tensor_basics()
    test_data_loading()
    test_matrix_operations()
    test_linear_regression_training()


if __name__ == "__main__":
    main()

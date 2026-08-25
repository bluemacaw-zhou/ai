"""自定义 PyTorch 数据集。"""

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """图像分类示例数据集，内置默认预处理管道。

    监督学习中，``data`` 通常记为 x（模型输入），``targets`` 通常记为
    y（正确标签）。Dataset 的职责只是根据索引将同一条样本的 x 和 y 配对返回。

    Dataset 的约定：
    - ``__len__`` 告诉 DataLoader 数据集有多少条样本；
    - ``__getitem__(index)`` 返回指定的一条 ``(样本, 标签)``。

    transform 不是 Dataset 的必需项，但可将原始样本转换为模型需要的
    Tensor，并在训练时实施随机翻转、旋转等数据增强。
    """

    # Compose 本身是一个可调用对象。执行 transform(sample) 时，它会按顺序
    # 执行下面的每个步骤：原始 HWC NumPy 图像 -> Tensor(CHW) -> 标准化 -> 增强图像。
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
    )

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data = data
        self.targets = targets
        # 调用方可传入一条自定义 transform 链；未传入时使用该数据集的默认链。
        # 因此 self.transform 保存的是一个 Compose 对象或其他可调用的转换对象。
        self.transform = transform or self.default_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        # DataLoader 取第 index 条数据时会调用此方法。
        # 这里等同于 self.transform.__call__(self.data[index])；Compose 会依次
        # 调用其内部的 ToTensor、Normalize、RandomHorizontalFlip、RandomRotation。
        sample = self.transform(self.data[index])
        # 返回 (x, y)：sample 是处理后的模型输入 x，targets[index] 是正确标签 y。
        # 模型随后才会根据 x 得到预测值 y_hat；Dataset 本身不负责模型预测或计算损失。
        # sample是预处理后的x的值 和实际的h还有距离
        return sample, self.targets[index]

"""线性回归示例数据集。"""

import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """保存监督学习中成对的输入 x 与目标 y。

    关系从小到大：一条样本是 ``(features[i], targets[i])``；其中 features[i]
    是该样本的一个或多个输入特征 x，targets[i] 是正确目标 y；本 Dataset 则是
    全部样本的集合，并负责保证相同索引处的 x 与 y 始终成对返回。
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if len(features) != len(targets):
            raise ValueError("features 与 targets 的样本数量必须一致")
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # DataLoader 将多次调用这里，再把多条 (x, y) 拼为一个 batch。
        return self.features[index], self.targets[index]

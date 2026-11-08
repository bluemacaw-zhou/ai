"""Epoch and loss 的独立绘图实现。"""

from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from tool.fitting_plot import FittingPlot
from tool.linear import Scalar


class EpochLossPlot(FittingPlot):
    def __init__(self, interval: float = 0.3) -> None:
        super().__init__('Epoch and loss', interval)
        self.epochs: list[int] = []  # 横轴历史值，每轮追加一个标量。
        self.losses: list[float] = []  # 与横轴一一对应的损失值。
        self.line: Line2D
        self.line, = self.axes.plot([], [], 'o-', markersize=4)
        self.axes.set(xlabel='Epoch (completed updates)', ylabel='MSE loss')

    def update(self, epoch: int, w: Scalar, b: Scalar = 0.0,
               prediction: NDArray[np.float64] | None = None,
               loss: Scalar | None = None) -> None:
        if not self.is_open():
            return
        if loss is None:
            raise ValueError('请传入当前训练状态的 loss')
        self.epochs.append(epoch)
        self.losses.append(float(loss))
        self.line.set_data(self.epochs, self.losses)
        self.axes.set_title(f'epoch={epoch}, W={w:.4f}, loss={loss:.6f}')
        self.refresh(autoscale=True)

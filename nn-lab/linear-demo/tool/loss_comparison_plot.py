"""在同一个窗口对比多个模型的损失曲线。"""

from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from tool.fitting_plot import FittingPlot
from tool.linear import Scalar


class LossComparisonPlot(FittingPlot):
    def __init__(self, labels: tuple[str, ...], interval: float = 0.0) -> None:
        super().__init__('Loss comparison', interval)
        self.epochs: dict[str, list[int]] = {label: [] for label in labels}
        self.losses: dict[str, list[float]] = {label: [] for label in labels}
        self.lines: dict[str, Line2D] = {}
        for label in labels:
            self.lines[label], = self.axes.plot([], [], label=label)
        self.axes.set(xlabel='Epoch (completed updates)', ylabel='MSE loss')
        self.axes.set_title('Same samples, different model capacity')
        self.axes.legend()

    def update(self, epoch: int, w: Scalar, b: Scalar = 0.0,
               prediction: NDArray[np.float64] | None = None,
               loss: Scalar | None = None, *, label: str = '') -> None:
        if not self.is_open():
            return
        if loss is None or label not in self.lines:
            raise ValueError('请传入当前模型的 loss 和已注册的 label')
        self.epochs[label].append(epoch)
        self.losses[label].append(float(loss))
        self.lines[label].set_data(self.epochs[label], self.losses[label])
        self.refresh(autoscale=True)

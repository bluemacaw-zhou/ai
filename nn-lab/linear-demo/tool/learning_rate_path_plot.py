"""在完整的 W-loss 曲线上对比不同学习率的参数移动路径。"""

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from numpy.typing import NDArray

from tool.fitting_plot import FittingPlot
from tool.linear import SampleArray, Scalar


class LearningRatePathPlot(FittingPlot):
    def __init__(self, x: SampleArray, y: SampleArray, labels: tuple[str, ...],
                 weight_limits: tuple[float, float]) -> None:
        super().__init__('Learning rates - W and loss', interval=0.0)
        self.weights: dict[str, list[float]] = {label: [] for label in labels}
        self.losses: dict[str, list[float]] = {label: [] for label in labels}
        self.lines: dict[str, Line2D] = {}
        self.points: dict[str, Line2D] = {}
        self.arrows: dict[str, FancyArrowPatch] = {}
        # 一维数组，shape=(400,)：仅用于画完整损失曲线，不参与参数更新。
        grid: NDArray[np.float64] = np.linspace(*weight_limits, 400)
        curve: NDArray[np.float64] = np.array([np.mean((w * x - y) ** 2) for w in grid])
        self.axes.plot(grid, curve, color='gray', alpha=0.6, label='MSE(W)')
        for label in labels:
            self.lines[label], = self.axes.plot([], [], 'o-', markersize=3, alpha=0.5,
                                                label=label)
            self.points[label], = self.axes.plot([], [], 'o', markersize=9,
                                                 color=self.lines[label].get_color())
        self.axes.set(xlabel='W', ylabel='MSE loss', xlim=weight_limits)
        self.axes.set_title('Large steps can cross the minimum repeatedly')
        self.axes.legend(fontsize=8)

    def update(self, epoch: int, w: Scalar, b: Scalar = 0.0,
               prediction: NDArray[np.float64] | None = None,
               loss: Scalar | None = None, *, label: str = '') -> None:
        if not self.is_open():
            return
        if loss is None or label not in self.lines:
            raise ValueError('请传入 loss 和已注册的 label')
        self.weights[label].append(float(w))
        self.losses[label].append(float(loss))
        self.lines[label].set_data(self.weights[label], self.losses[label])
        self.points[label].set_data([w], [loss])
        # 箭头只标记最新一步，避免历史箭头遮挡损失曲线。
        if label in self.arrows:
            self.arrows.pop(label).remove()
        if len(self.weights[label]) > 1:
            arrow: FancyArrowPatch = FancyArrowPatch(
                (self.weights[label][-2], self.losses[label][-2]), (float(w), float(loss)),
                arrowstyle='->', mutation_scale=14, color=self.lines[label].get_color(),
            )
            self.axes.add_patch(arrow)
            self.arrows[label] = arrow
        self.refresh()

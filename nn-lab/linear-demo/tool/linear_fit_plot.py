"""真实样本与模型预测直线。"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from tool.fitting_plot import FittingPlot
from tool.linear import SampleArray, Scalar


class LinearFitPlot(FittingPlot):
    """固定真实样本和坐标范围，只刷新模型预测直线。"""

    def __init__(self, x: SampleArray, y: SampleArray, w: Scalar,
                 b: Scalar = 0.0, interval: float = 0.3) -> None:
        super().__init__('Model fitting', interval)
        self.x: SampleArray = np.asarray(x)  # 一维输入数组，shape=(n,)。
        self.y: SampleArray = np.asarray(y)  # 一维标签数组，shape=(n,)。
        # 创建一次图形，避免每轮清空窗口造成闪烁。
        self.axes.scatter(self.x, self.y, label='Target', color='blue', zorder=3)
        self.line: Line2D
        self.line, = self.axes.plot(self.x, self.x * w + b, color='red', label='Prediction')
        self.axes.set(xlabel='X', ylabel='Y')
        self.axes.legend()
        # 固定坐标范围，便于直接观察斜率和截距的变化。
        self.axes.margins(0.15)
        self.axes.autoscale_view()
        self.axes.set_xlim(self.axes.get_xlim())
        self.axes.set_ylim(self.axes.get_ylim())
        if interval > 0:
            # 单窗口示例展示训练前的直线，留出时间观察初始状态。
            plt.show(block=False)
            self.update(0, w, b)

    def update(self, epoch: int, w: Scalar, b: Scalar = 0.0,
               prediction: NDArray[np.float64] | None = None,
               loss: Scalar | None = None) -> None:
        if not self.is_open():
            return
        # 多窗口直接复用训练结果；单窗口示例也可以只传入 w、b。
        if prediction is None:
            prediction = self.x * w + b
        if loss is None:
            loss = np.mean((prediction - self.y) ** 2)
        self.line.set_ydata(prediction)
        self.axes.set_title(f'epoch={epoch}, y={w:.4f}x + {b:.4f}, loss={loss:.6f}')
        self.refresh()

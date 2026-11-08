"""动态图窗基类：管理窗口、绘图后端和公共的刷新行为。"""

from abc import ABC, abstractmethod
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from tool.linear import Scalar


class FittingPlot(ABC):
    """每种图独立实现 update；基类负责窗口生命周期。"""

    def __init__(self, title: str, interval: float = 0.3) -> None:
        self.configure_backend()
        self.interval: float = interval  # 秒；为 0 时由训练循环统一处理动画事件。
        plt.ion()
        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=(6, 4))
        self.figure.canvas.manager.set_window_title(title)
        self.axes.grid(True)

    @staticmethod
    def configure_backend() -> None:
        backend: str = plt.get_backend().lower()
        in_pycharm: bool = os.environ.get("PYCHARM_HOSTED") == "1"
        captured_plot: bool = "interagg" in backend or "pycharm" in backend

        # 显式指定Agg时保留无界面模式，方便批量验证
        headless: bool = os.environ.get("MPLBACKEND", "").lower() == "agg"
        if sys.platform == "darwin" and (in_pycharm or captured_plot) and not headless:
            if backend != "macosx":
                plt.switch_backend("MacOSX")

    @staticmethod
    def arrange_side_by_side(plots: tuple["FittingPlot", ...]) -> None:
        """按创建顺序横向排列图窗，适配当前 macOS 屏幕的可用区域。"""
        if plt.get_backend().lower() != 'macosx':
            return
        from tool.window_layout import arrange_macos_windows
        # 显示窗口并处理创建事件后，再按窗口标题定位。
        plt.pause(0.01)
        titles: list[str] = [plot.figure.canvas.manager.get_window_title() for plot in plots]
        arrange_macos_windows(titles)

    def is_open(self) -> bool:
        return plt.fignum_exists(self.figure.number)

    @abstractmethod
    def update(self, epoch: int, w: Scalar, b: Scalar = 0.0,
               prediction: NDArray[np.float64] | None = None,
               loss: Scalar | None = None) -> None:
        """展示同一训练状态；prediction 和 loss 可直接使用训练循环的结果。"""
        raise NotImplementedError

    def refresh(self, autoscale: bool = False) -> None:
        if autoscale:
            self.axes.relim()
            self.axes.autoscale_view()
        self.figure.canvas.draw_idle()
        # 单窗口独立暂停；多窗口先全部更新，再由训练循环统一暂停一次。
        if self.interval > 0:
            plt.pause(self.interval)

    def finish(self) -> None:
        """保留最终图形，关闭窗口后继续或退出。"""
        plt.ioff()
        if self.is_open():
            plt.show(block=True)

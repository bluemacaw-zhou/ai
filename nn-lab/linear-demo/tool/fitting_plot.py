"""多个训练示例共用的实时拟合图。"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


class FittingPlot:
    """固定真实样本和坐标范围，只刷新模型预测直线。"""

    def __init__(self, x, y, w, b=0.0, interval=0.3):
        # PyCharm的Plots面板接管显示时，改用macOS独立窗口处理动画事件
        self.configure_backend()
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.interval = interval

        # 创建一次图形，避免每轮清空窗口造成闪烁
        plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(7, 5))
        self.axes.scatter(self.x, self.y, label="Target", color="blue", zorder=3)
        self.line, = self.axes.plot(self.x, self.x * w + b, color="red", label="Prediction")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.legend()
        self.axes.grid(True)

        # 固定坐标范围，便于直接观察斜率和截距的变化
        self.axes.margins(0.15)
        self.axes.autoscale_view()
        self.axes.set_xlim(self.axes.get_xlim())
        self.axes.set_ylim(self.axes.get_ylim())

        # 展示训练前的直线，留出时间观察初始状态
        plt.show(block=False)
        self.update(0, w, b)

    @staticmethod
    def configure_backend():
        backend = plt.get_backend().lower()
        in_pycharm = os.environ.get("PYCHARM_HOSTED") == "1"
        captured_plot = "interagg" in backend or "pycharm" in backend

        # 显式指定Agg时保留无界面模式，方便批量验证
        headless = os.environ.get("MPLBACKEND", "").lower() == "agg"
        if sys.platform == "darwin" and (in_pycharm or captured_plot) and not headless:
            if backend != "macosx":
                plt.switch_backend("MacOSX")

    def is_open(self):
        return plt.fignum_exists(self.figure.number)

    def update(self, epoch, w, b=0.0):
        """传入当前参数，更新预测线和对应的Loss。"""
        if not self.is_open():
            return

        prediction = self.x * w + b
        loss = np.mean((prediction - self.y) ** 2)
        self.line.set_ydata(prediction)
        self.axes.set_title(f"epoch={epoch}, y={w:.4f}x + {b:.4f}, loss={loss:.6f}")
        self.figure.canvas.draw_idle()

        # 暂停时处理窗口事件，让每一轮拟合变化都能被肉眼观察
        plt.pause(self.interval)

    def finish(self):
        """保留最终图形，关闭窗口后继续或退出。"""
        plt.ioff()
        if self.is_open():
            plt.show(block=True)

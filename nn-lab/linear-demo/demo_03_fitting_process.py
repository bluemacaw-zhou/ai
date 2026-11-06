"""用静态快照和动态图观察直线拟合过程。"""

import matplotlib.pyplot as plt
import numpy as np
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot

log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


# 固定的


class FittingProcessDemo:
    """用静态快照和动态图观察直线拟合过程。"""

    def run(self):
        """按阅读顺序执行本节示例。"""
        self.plot_snapshots()
        self.animate_training()

    def plot_snapshots(self):
        np.random.seed(0)

        # 输入值
        X = np.array([1, 2, 3, 4, 5])

        # 标签值
        Y = np.array([
            1 * 3,
            2 * 3,
            3 * 3,
            4 * 3,
            5 * 3,
        ])

        # 初始化随机权重
        W = np.random.random()

        # 学习步长
        lr = 0.01

        # 保存指定轮次的 W
        show_epochs = [0, 1, 2, 5, 10, 20, 50, 100]

        history = {}

        # 训练循环
        for epoch in range(1000):
            # 前向传播
            Y_pred = X * W

            # 误差
            error = Y_pred - Y

            # Loss计算 采用MSE算法
            loss = np.mean(error ** 2)

            # 反向传播，计算Loss对预测值的梯度
            dY_pred = 2 * error / len(X)

            dW = np.sum(dY_pred * X)

            # 保存当前模型
            if epoch in show_epochs:
                history[epoch] = W

            # 更新参数
            W = W - lr * dW

        # 画图

        # 真实结果
        plt.plot(
            X,
            Y,
            marker="o",
            linewidth=4,
            label="Real: y = 3x"
        )

        # 不同训练轮次
        for epoch, w in history.items():

            Y_pred = X * w

            plt.plot(
                X,
                Y_pred,
                marker="o",
                label=f"epoch={epoch}, W={w:.4f}"
            )

        plt.xlabel("X")
        plt.ylabel("Y")

        plt.title("Model Fitting Process")

        plt.legend()

        plt.grid()

        plt.show()

    def animate_training(self):
        # 实时变动的
        np.set_printoptions(suppress=True)

        # 训练目标：y = 3x
        X = np.array([1, 2, 3, 4, 5])

        # 标签值
        Y_true = np.array([
            1 * 3,
            2 * 3,
            3 * 3,
            4 * 3,
            5 * 3,
        ])

        # 初始化权重
        w = 0.1
        # 学习步长
        lr = 0.01

        # 创建公共拟合图
        plot = FittingPlot(X, Y_true, w)

        # 训练循环
        for epoch in range(20):
            # 关闭窗口时结束当前示例
            if not plot.is_open():
                break

            loss_total = 0
            dw_total = 0

            for x, y_true in zip(X, Y_true):
                # 前向传播
                y_pred = w * x

                # 误差
                error = y_pred - y_true

                # Loss计算 采用MSE算法
                loss = error ** 2
                loss_total += loss

                # 反向传播，计算权重梯度
                dw = 2 * error * x
                dw_total += dw

            loss_avg = loss_total / len(X)
            dw_avg = dw_total / len(X)

            # 更新参数
            w = w - lr * dw_avg

            # 当前模型预测直线
            Y_ret = X * w
            loss = np.mean((Y_ret - Y_true) ** 2)

            # 刷新当前参数对应的预测直线
            plot.update(epoch + 1, w)


            log.info(f'第 {epoch + 1} 次训练')
            log.info(f'loss: {loss}')
            log.info(f'dw_avg: {dw_avg}')
            log.info(f'w = {w}')
            log.info("-" * 50)

        # 训练结束后保留最终图形，关闭窗口后继续下一个示例或退出。
        plot.finish()


if __name__ == "__main__":
    FittingProcessDemo().run()

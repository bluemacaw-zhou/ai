"""观察训练轮次、权重与均方误差的关系。"""

import matplotlib.pyplot as plt
import numpy as np
from ai_starter import get_logger

log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


# 看每次训练，loss的变化


class GradientDescentDemo:
    """观察训练轮次、权重与均方误差的关系。"""

    def run(self):
        """按阅读顺序执行本节示例。"""
        self.plot_epoch_loss()
        self.plot_weight_loss()

    def plot_epoch_loss(self):
        # 数组输出时不使用科学计数法
        np.set_printoptions(suppress=True)

        # 中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 训练目标：让模型学会 y = 3x
        X = np.array([1, 3, 5, 9, 11], dtype=np.float32)
        # 标签值
        Y_true = np.array([
            1 * 3,
            3 * 3,
            5 * 3,
            9 * 3,
            11 * 3,
        ], dtype=np.float32)

        # 初始化权重
        w = 0.1
        # 学习步长
        lr = 0.01

        LOSS = []

        # 训练循环
        for epoch in range(10):
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
            LOSS.append(loss_avg)

            dw_avg = dw_total / len(X)

            # 更新参数
            w = w - lr * dw_avg

            log.info(f'epoch={epoch + 1}, loss_avg={loss_avg}, dw_avg={dw_avg}, w={w}')

        # 画 epoch 与 loss 的图
        epochs = np.arange(1, len(LOSS) + 1)

        plt.plot(epochs, LOSS)

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch 与 loss 的关系")

        plt.show()

    def plot_weight_loss(self):
        # 看每次训练，w与loss的变化关系
        # 数组输出时不使用科学计数法
        np.set_printoptions(suppress=True)

        # 中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 训练目标：让模型学会 y = 3x
        X = np.array([1, 3,], dtype=np.float32)
        # 标签值
        Y_true = np.array([
            1 * 3,
            3 * 3,
        ], dtype=np.float32)

        # 初始化权重
        w = 0.1
        # 学习步长
        lr = 0.01

        W = []
        LOSS = []

        # 训练循环
        for epoch in range(100):
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
            LOSS.append(loss_avg)

            dw_avg = dw_total / len(X)

            # 更新参数
            w = w - lr * dw_avg
            W.append(w)

            log.info(f'epoch={epoch + 1}, loss_avg={loss_avg}, dw_avg={dw_avg}, w={w}')

        plt.plot(W, LOSS)
        plt.xlabel("w")
        plt.ylabel("loss")
        plt.title("w 与 loss 的关系(f(w)，loss随w变化)")
        plt.show()


if __name__ == "__main__":
    GradientDescentDemo().run()

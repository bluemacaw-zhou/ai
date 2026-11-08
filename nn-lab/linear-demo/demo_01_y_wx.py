"""训练单参数模型并实时观察直线拟合过程。"""

import numpy as np
from numpy.typing import NDArray
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot

# Scalar 表示一个数：可能是 Python 数值，也可能是 NumPy 数值标量；不是数组。
Scalar = int | float | np.int64 | np.float32 | np.float64
# NDArray[...] 表示 NumPy 数组，方括号中是元素类型；形状见变量旁的注释。

# 日志记录器对象，用于输出训练信息。
log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


# 让线性模型学会 y=3x
class LinearWeightDemo:
    """训练单参数模型并实时观察直线拟合过程。"""

    def run(self) -> None:
        """按阅读顺序执行本节示例。"""
        self.train()

    def train(self) -> None:
        # 设置随机种子
        np.random.seed(0)

        # 输入值
        # 一维数组，shape=(n,)，n 为样本数；每个元素对应一个样本。
        X: NDArray[np.int64] = np.array([1, 2, 3, 4, 5])

        # 标签值
        # 一维数组，shape=(n,)，n 为样本数；每个元素对应一个样本。
        Y: NDArray[np.int64] = np.array([
            1 * 3,
            2 * 3,
            3 * 3,
            4 * 3,
            5 * 3,
        ])

        # 初始化随机权重
        # 数值标量，一个数，不是数组。
        W: Scalar = np.random.random()
        log.info("W的值是: %s", W)

        # 学习步长
        # Python 浮点标量，一个数。
        lr: float = 0.01

        # 创建公共拟合图，每轮停留0.5秒，观察直线如何靠近真实点
        plot: FittingPlot = FittingPlot(X, Y, W, interval=0.5)

        # 训练循环
        # epoch: int，当前训练轮次（从 0 开始）。
        for epoch in range(20):
            # 关闭窗口时结束训练
            if not plot.is_open():
                plot.finish()
                return

            # 前向传播
            # 浮点一维数组，shape=(n,)，每个样本对应一个值。
            Y_pred: NDArray[np.float64] = X * W

            # 误差
            # 浮点一维数组，shape=(n,)，每个样本对应一个值。
            error: NDArray[np.float64] = Y_pred - Y

            # Loss计算 采用MSE算法
            # 计算示例：假设 error = np.array([1, -2, 3])
            # 第一步：error ** 2 对每个误差分别平方，得到 [1, 4, 9]
            # 第二步：np.mean 求平均值，即 (1 + 4 + 9) / 3 = 14 / 3 ≈ 4.6667
            # 因此 loss 是一个数，表示所有样本的平均平方误差，越小表示预测越准确。
            # 这里的 loss 用于展示当前损失；下面直接用手写的导数公式计算梯度。
            # 数值标量，一个数，不是数组。
            loss: Scalar = np.mean(error ** 2)

            # 展示当前参数的结果；epoch 表示已经完成的参数更新次数。
            self.plot_training(plot, epoch, W)
            log.info(f'epoch={epoch}, loss={loss:.6f}, W={W:.6f}')

            # 反向传播
            # 幂函数求导规则：指数移到前面作为系数，同时指数减 1。
            # 即 w ** n 的导数是 n * w ** (n - 1)。
            # 例如 w ** 2 -> 2 * w；w ** 1 -> 1（一次函数的变化速度恒定）。
            # loss = sum((Y_pred - Y) ** 2) / len(X)，对每个预测值求导：
            # 平方求导后得到 2 * (Y_pred - Y)，再除以样本数；这里用 error，不是 loss。
            # 浮点一维数组，shape=(n,)，每个样本对应一个值。
            d_loss_d_y_pred: NDArray[np.float64] = 2 * error / len(X)
            # Y_pred_i = W * X_i：对 W 求导时 X_i 固定，W 的导数是 1，结果为 X_i。
            # 按链式法则，将上一步的导数乘以 X_i，再将所有样本对 W 的贡献相加。
            # 数值标量，一个数，不是数组。
            d_loss_d_w: Scalar = np.sum(d_loss_d_y_pred * X)

            # 更新参数
            # 其实这个就是优化器
            W = W - lr * d_loss_d_w


        log.info("=" * 50)
        log.info(f'训练后的 W:\n{W}')

        # 预测值
        Y_pred = X * W

        # 循环结束后统一展示最后一次更新的结果，避免在每轮中重复预测。
        loss = np.mean((Y_pred - Y) ** 2)
        self.plot_training(plot, epoch + 1, W)
        log.info(f'最终 epoch={epoch + 1}, loss={loss:.6f}, W={W:.6f}')

        # 肉眼就可以看出来，是线性的。线性模型学习过程中，找到一根线，把这几个点串起来，就搞定
        log.info(f'最终预测结果:\n{Y_pred}')
        log.info(f'正确结果:\n{Y}')

        # 训练结束后保留最终图形，关闭窗口后退出
        plot.finish()

    def plot_training(self, plot: FittingPlot, epoch: int, w: Scalar) -> None:
        # 绘图交给公共工具，训练循环只传入当前轮次和权重
        plot.update(epoch, w)


if __name__ == "__main__":
    LinearWeightDemo().run()

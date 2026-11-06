"""对比不含偏置和含偏置的线性模型。"""

import numpy as np
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot

log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


# 如果数据不经过原点，不加偏置b，线性模型永远学不会


class LinearBiasDemo:
    """对比不含偏置和含偏置的线性模型。"""

    def run(self):
        """按阅读顺序执行本节示例。"""
        self.train_without_bias()
        self.train_with_bias()

    def train_without_bias(self):
        np.set_printoptions(suppress=True)

        # 训练目标：y = 3x + 5；本节先演示不含偏置的模型
        X = np.array([1, 2, 3, 4, 5])

        # 标签值
        Y_true = np.array([
            1 * 3 + 5,
            2 * 3 + 5,
            3 * 3 + 5,
            4 * 3 + 5,
            5 * 3 + 5,
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

    def train_with_bias(self):
        # 加上偏置b才能学会
        np.set_printoptions(suppress=True)

        # 训练目标：y = 3x + 5
        X = np.array([1, 2, 3, 4, 5])

        # 标签值
        Y_true = np.array([
            1 * 3 + 5,
            2 * 3 + 5,
            3 * 3 + 5,
            4 * 3 + 5,
            5 * 3 + 5,
        ])

        # 初始化权重和偏置
        w = 0.1
        b = 0

        # 学习步长
        lr = 0.01

        # 创建公共拟合图
        plot = FittingPlot(X, Y_true, w, b)

        # 训练循环
        for epoch in range(100):
            # 关闭窗口时结束当前示例
            if not plot.is_open():
                break

            # 前向传播
            Y_pred = X * w + b

            # 误差
            error = Y_pred - Y_true

            # Loss计算 采用MSE算法
            loss = np.mean(error ** 2)

            # 反向传播，计算Loss对预测值的梯度
            dY_pred = 2 * error / len(X)

            # 计算权重梯度，dY_pred / dw = X
            dw = np.sum(dY_pred * X)

            # 计算偏置梯度，dY_pred / db = 1
            db = np.sum(dY_pred)

            # 梯度下降
            w = w - lr * dw

            b = b - lr * db

            # 使用更新后的 w、b 重新预测
            Y_ret = X * w + b
            loss = np.mean((Y_ret - Y_true) ** 2)

            # 刷新当前参数对应的预测直线
            plot.update(epoch + 1, w, b)


            log.info(f'第 {epoch + 1} 次训练')

            log.info(f'loss = {loss}')

            log.info(f'dw = {dw}')
            log.info(f'db = {db}')

            log.info(f'w = {w}')
            log.info(f'b = {b}')

            log.info("-" * 50)

        # 训练结束后保留最终图形，关闭窗口后继续下一个示例或退出。
        plot.finish()

        # 上面的代码，训练了100次，线性模型好像没有完全学会，那要怎么办呢？
        # 1、增加训练轮次，观察参数是否继续收敛；重复相同数据不会改变最优解
        # 2、w、b不共享lr，可以定义一个b独享的lr_b=0.1，可以加快线性模型的学习速度


if __name__ == "__main__":
    LinearBiasDemo().run()

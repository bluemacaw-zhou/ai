"""使用不同学习率拟合非共线数据。"""

import numpy as np
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot

log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


class ApproximateFitDemo:
    """使用不同学习率拟合非共线数据。"""

    def run(self):
        """按阅读顺序执行本节示例。"""
        self.train()

    def train(self):
        np.set_printoptions(suppress=True)

        # 用直线近似拟合非共线数据；这些点无法由 y = wx + b 完全拟合
        X = np.array([1, 2, 3, 4, 5])

        # 标签值
        Y_true = np.array([
            1 * 3 + 5,
            2 * 3 + 7,
            3 * 3 + 10,
            4 * 3 + 14,
            5 * 3 + 19,
        ])

        # 初始化权重和偏置
        w = 0.1
        b = 0

        # 学习步长
        lr = 0.01
        lr_b = 0.1

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

            b = b - lr_b * db

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


if __name__ == "__main__":
    ApproximateFitDemo().run()

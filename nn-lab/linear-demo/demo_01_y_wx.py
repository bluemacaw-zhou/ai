"""训练单参数模型并实时观察直线拟合过程。"""

import numpy as np
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot

log = get_logger(__name__)


# 阅读导引：01：单变量线性回归与梯度下降
# 从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。
# 本目录说明和推荐顺序见 `README.md`。本章使用 NumPy 手写训练过程，封装示例使用本目录 `tool/` 下的 `linear.py` 和 `sgd.py`。


# 让线性模型学会 y=3x
class LinearWeightDemo:
    """训练单参数模型并实时观察直线拟合过程。"""

    def run(self):
        """按阅读顺序执行本节示例。"""
        self.train()

    def train(self):
        # 设置随机种子
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

        # 创建公共拟合图，每轮停留0.5秒，观察直线如何靠近真实点
        plot = FittingPlot(X, Y, W, interval=0.5)

        # 训练循环
        for epoch in range(20):
            # 关闭窗口时结束训练
            if not plot.is_open():
                break

            # 前向传播
            Y_pred = X * W

            # 误差
            error = Y_pred - Y

            # Loss计算 采用MSE算法
            loss = np.mean(error ** 2)

            # 反向传播
            dY_pred = 2 * error / len(X)
            dW = np.sum(dY_pred * X)

            # 更新参数
            W = W - lr * dW

            # 使用新权重重新预测，让直线、标题和日志对应同一组参数
            Y_pred = X * W
            loss = np.mean((Y_pred - Y) ** 2)
            self.plot_training(plot, epoch + 1, W)

            log.info(f'epoch={epoch + 1}, loss={loss:.6f}, W={W:.6f}')

        log.info("=" * 50)
        log.info(f'训练后的 W:\n{W}')

        # 预测值
        Y_pred = X * W

        # 肉眼就可以看出来，是线性的。线性模型学习过程中，找到一根线，把这几个点串起来，就搞定
        log.info(f'最终预测结果:\n{Y_pred}')
        log.info(f'正确结果:\n{Y}')

        # 训练结束后保留最终图形，关闭窗口后退出
        plot.finish()

    def plot_training(self, plot, epoch, w):
        # 绘图交给公共工具，训练循环只传入当前轮次和权重
        plot.update(epoch, w)


if __name__ == "__main__":
    LinearWeightDemo().run()

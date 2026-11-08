"""同一训练循环对比 y = wx 和 y = wx + b，观察偏置的作用。"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import NDArray
from ai_starter import get_logger

from tool.fitting_plot import FittingPlot
from tool.linear import Linear, Scalar
from tool.linear_fit_plot import LinearFitPlot
from tool.loss_comparison_plot import LossComparisonPlot
from tool.sgd import SGD

log = get_logger(__name__)  # 日志记录器对象。


class LinearBiasDemo:
    """两个模型共用样本和训练逻辑，各自维护参数、优化器和拟合图。"""

    def run(self) -> None:
        # 实验参数：改为 target_b=0，观察两个模型都可以拟合。
        target_b: float = 5.0
        # 使用奇数个、关于 0 对称的样本，包含 x=0；可改为 101 对比样本数量。
        # 保持区间不变，避免把样本分布变化误认为样本数量的影响。
        sample_count: int = 5
        epochs: int = 100
        lr: float = 0.05  # 学习率，两个模型使用相同的步长。

        # 一维浮点数组，shape=(sample_count,)，每个元素对应一个样本。
        X: NDArray[np.float64] = np.linspace(-2.0, 2.0, sample_count)
        Y_true: NDArray[np.float64] = 3 * X + target_b

        labels: tuple[str, ...] = ('Without bias: y = wx', 'With bias: y = wx + b')
        models: tuple[Linear, ...] = (Linear(use_bias=False), Linear(use_bias=True))
        optimizers: tuple[SGD, ...] = tuple(SGD(model, lr=lr) for model in models)
        fit_plots: tuple[LinearFitPlot, ...] = tuple(
            LinearFitPlot(X, Y_true, model.w, model.b, interval=0.0) for model in models
        )
        loss_plot: LossComparisonPlot = LossComparisonPlot(labels)
        plots: tuple[FittingPlot, ...] = (*fit_plots, loss_plot)
        # 每个拟合窗口的虚线段表示预测值与标签值之间的误差。
        residuals: list[LineCollection] = []
        for label, plot in zip(labels, fit_plots):
            plot.figure.canvas.manager.set_window_title(label)
            plot.axes.plot(X, Y_true, '--', color='blue', alpha=0.5, label='True relationship')
            plot.axes.scatter([0], [0], color='black', marker='x', label='Origin (0, 0)')
            plot.axes.scatter([0], [target_b], facecolors='none', edgecolors='green',
                              s=100, label=f'Target at x=0: {target_b:g}')
            residuals.append(plot.axes.vlines(X, Y_true, Y_true, color='orange',
                                             linestyles='dashed', label='Prediction error'))
            plot.axes.legend(fontsize=8)
            # 两张图使用相同范围，包含原点和所有样本，便于公平比较。
            plot.axes.set_ylim(min(float(Y_true.min()), float((3 * X).min()), 0.0) - 1,
                               max(float(Y_true.max()), float((3 * X).max()), 0.0) + 1)
        plt.show(block=False)
        FittingPlot.arrange_side_by_side(plots)

        # epoch 表示已完成的更新次数；最后一轮只展示最终结果，不再更新参数。
        # 预测、损失、展示、求梯度和更新的逻辑只写一份，供两个模型共用。
        for epoch in range(epochs + 1):
            if not all(plot.is_open() for plot in plots):
                break
            for label, model, optimizer, plot, residual in zip(
                labels, models, optimizers, fit_plots, residuals,
            ):
                # 前向传播：浮点一维数组，shape=(sample_count,)。
                Y_pred: NDArray[np.float64] = model.forward(X)
                error: NDArray[np.float64] = Y_pred - Y_true
                # MSE：先逐项平方，再取平均；loss 是一个标量，不是数组。
                loss: Scalar = np.mean(error ** 2)

                # 当前权重、偏置、预测值和损失属于同一个训练状态。
                residual.set_segments([[(x, actual), (x, predicted)]
                                       for x, actual, predicted in zip(X, Y_true, Y_pred)])
                plot.update(epoch, model.w, model.b, prediction=Y_pred, loss=loss)
                loss_plot.update(epoch, model.w, model.b, loss=loss, label=label)
                log.info('%s | epoch=%s, w=%.4f, b=%.4f, loss=%.6f',
                         label, epoch, model.w, model.b, loss)

                if epoch < epochs:
                    optimizer.zero_grad()
                    # 平方求导：指数移到前面，同时指数减 1。
                    # loss 对每个预测值的导数为 2 * error / 样本数，结果仍是一维数组。
                    d_loss_d_y_pred: NDArray[np.float64] = 2 * error / len(X)
                    # 链式法则：预测值对 w 的导数是 X，对 b 的导数是 1。
                    # d_loss_d_w = sum(d_loss_d_y_pred * X)，是标量。
                    # d_loss_d_b = sum(d_loss_d_y_pred)，也是标量。
                    model.backward(d_loss_d_y_pred)
                    # 优化器执行 w -= lr * d_loss_d_w；使用偏置时也更新 b。
                    optimizer.step()
            # 三个窗口更新后统一暂停一次，同时处理窗口事件。
            plt.pause(0.1)

        # 对称样本使不含偏置的最优斜率也是 3，但缺少的截距会留下误差。
        # x=0 时，y=wx 永远预测 0；target_b=5 时，该样本误差永远为 -5。
        # 增加训练轮次或重复样本不能补上缺失的偏置；增加样本数也不等于误差更大。
        # 训练结束后保留最终图形；关闭任意窗口停止训练，关闭其余窗口后退出。
        plt.ioff()
        plt.show(block=True)


if __name__ == '__main__':
    LinearBiasDemo().run()

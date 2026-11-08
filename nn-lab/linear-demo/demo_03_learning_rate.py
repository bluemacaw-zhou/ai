"""同一训练流程对比不同学习率：更大的步长不一定收敛得更快。"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ai_starter import get_logger

from tool.epoch_loss_plot import EpochLossPlot
from tool.fitting_plot import FittingPlot
from tool.learning_rate_path_plot import LearningRatePathPlot
from tool.linear import Linear, Scalar
from tool.sgd import SGD

log = get_logger(__name__)  # 日志记录器对象。


class LearningRateDemo:
    """相同数据、初始权重和训练次数，只改变学习率。"""

    def run(self) -> None:
        # 一维浮点数组，shape=(5,)，目标 y=3x；只训练一个权重，不使用偏置。
        X: NDArray[np.float64] = np.array([1., 2., 3., 4., 5.])
        Y_true: NDArray[np.float64] = 3 * X
        initial_w: float = 0.1
        epochs: int = 40
        learning_rates: tuple[float, ...] = (0.04, 0.089)
        labels: tuple[str, ...] = tuple(f'lr={lr:g}' for lr in learning_rates)

        # 本例 loss=11*(W-3)^2，导数为 22*(W-3)。
        # 更新后 W_new-3=(1-22*lr)*(W-3)，由这个倍数判断参数如何移动：
        # lr=0.04  -> 倍数 0.12：从同一侧快速靠近最低点。
        # lr=0.089 -> 倍数 -0.958：每次跨过最低点，但距离缩小得很慢。
        # 注意：参数来回跳跃时，loss 仍可能一直下降，不一定上下震荡。
        # 此数据下 lr > 1/11 ≈ 0.09091 会发散；这个阈值不适用于所有数据。
        models: tuple[Linear, ...] = tuple(Linear(use_bias=False) for _ in learning_rates)
        for model in models:
            model.w = initial_w
        optimizers: tuple[SGD, ...] = tuple(
            SGD(model, lr=lr) for model, lr in zip(models, learning_rates)
        )
        epoch_plots: tuple[EpochLossPlot, ...] = tuple(
            EpochLossPlot(interval=0.0) for _ in learning_rates
        )
        # 使用相同的横纵轴范围，防止自动缩放掩盖两种学习率的差别。
        initial_loss: float = float(np.mean((initial_w * X - Y_true) ** 2))
        for label, plot in zip(labels, epoch_plots):
            plot.figure.canvas.manager.set_window_title(f'Epoch and loss - {label}')
            plot.axes.set_xlim(0, epochs)
            plot.axes.set_ylim(-initial_loss * 0.03, initial_loss * 1.05)
            plot.axes.set_autoscale_on(False)
        path_plot: LearningRatePathPlot = LearningRatePathPlot(
            X, Y_true, labels, weight_limits=(-0.2, 6.2),
        )
        path_plot.axes.axvline(3, color='green', linestyle='--', alpha=0.5,
                              label='Minimum W=3')
        path_plot.axes.legend(fontsize=8)
        plots: tuple[FittingPlot, ...] = (*epoch_plots, path_plot)
        plt.show(block=False)
        FittingPlot.arrange_side_by_side(plots)

        # 同一个循环推进两个模型；epoch 是已完成的更新次数。
        # 最后一个状态只展示，不再更新，避免循环内部重复预测和计算 loss。
        for epoch in range(epochs + 1):
            if not all(plot.is_open() for plot in plots):
                break
            for label, model, optimizer, plot in zip(labels, models, optimizers, epoch_plots):
                # 预测值和误差都是一维数组，shape=(5,)。
                Y_pred: NDArray[np.float64] = model.forward(X)
                error: NDArray[np.float64] = Y_pred - Y_true
                # 先逐项平方，再取平均；loss 是标量，不是数组。
                loss: Scalar = np.mean(error ** 2)
                plot.update(epoch, model.w, prediction=Y_pred, loss=loss)
                path_plot.update(epoch, model.w, loss=loss, label=label)
                log.info('%s | epoch=%s, W=%.6f, loss=%.8f', label, epoch, model.w, loss)

                if epoch < epochs:
                    optimizer.zero_grad()
                    # 平方求导后得到 2*error，再除以样本数；结果为一维数组。
                    d_loss_d_y_pred: NDArray[np.float64] = 2 * error / len(X)
                    # 内部计算 d_loss_d_w=sum(d_loss_d_y_pred*X)，得到权重的标量梯度。
                    model.backward(d_loss_d_y_pred)
                    # W = W - lr*d_loss_d_w；两个模型只有这里的 lr 不同。
                    optimizer.step()
            # 三个窗口每轮一起刷新，第三个窗口的大圆点和箭头表示当前状态与最新移动。
            plt.pause(0.3)

        plt.ioff()
        plt.show(block=True)


if __name__ == '__main__':
    LearningRateDemo().run()

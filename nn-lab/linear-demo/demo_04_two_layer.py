"""两层一元线性模型：观察梯度如何从第二层传回第一层。

每个样本只有一个输入值，两层也各自只有一个线性神经元。
这里不加激活函数，目的是单独看清链式法则；两层合起来仍等价于一条直线。
"""

import numpy as np
from numpy.typing import NDArray
from ai_starter import get_logger

from tool.linear import Linear, Scalar
from tool.sgd import SGD

log = get_logger(__name__)  # 日志记录器对象。


class TwoLayerDemo:
    """复用两个 Linear 实例，明确写出前向和反向的调用顺序。"""

    def run(self) -> None:
        # 一维浮点数组，shape=(5,)：5 个样本，每个样本只有 1 个输入特征。
        X: NDArray[np.float64] = np.array([-2., -1., 0., 1., 2.])
        Y_true: NDArray[np.float64] = 3 * X + 5

        # 每层各有一个权重和偏置；它们都是标量，不是数组。
        layer1: Linear = Linear()
        layer2: Linear = Linear()
        # 显式设置初值，便于手算第一轮；不要把两层权重都初始化为 0。
        layer1.w = 0.5
        layer2.w = 0.5
        optimizer1: SGD = SGD(layer1, lr=0.01)
        optimizer2: SGD = SGD(layer2, lr=0.01)
        epochs: int = 200

        # 计算关系：X -> 第一层 -> h -> 第二层 -> Y_pred -> loss。
        # h = w1 * X + b1；Y_pred = w2 * h + b2。
        # epoch 表示已完成的更新次数，最后一次只展示结果，不再更新参数。
        for epoch in range(epochs + 1):
            # 1. 前向传播：先第一层，再第二层。
            # h 是中间结果，既是第一层的输出，也是第二层的输入。
            # h、Y_pred、error 都是一维数组，shape=(5,)。
            h: NDArray[np.float64] = layer1.forward(X)
            Y_pred: NDArray[np.float64] = layer2.forward(h)
            error: NDArray[np.float64] = Y_pred - Y_true
            # MSE：逐项平方再求平均，得到一个损失标量。
            loss: Scalar = np.mean(error ** 2)

            if epoch % 20 == 0:
                log.info('epoch=%s, loss=%.8f | w1=%.4f, b1=%.4f | w2=%.4f, b2=%.4f',
                         epoch, loss, layer1.w, layer1.b, layer2.w, layer2.b)
            if epoch == epochs:
                break

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # 2. 从损失开始求导。平方的导数是 2 * error，再除以样本数。
            # loss 对每个预测值的导数：一维数组，shape=(5,)。
            d_loss_d_y_pred: NDArray[np.float64] = 2 * error / len(X)

            # 3. 先反向传播第二层：Y_pred = w2 * h + b2。
            # 内部保存两个标量，供 optimizer2 使用：
            # d_loss_d_w2 = sum(d_loss_d_y_pred * h)
            # d_loss_d_b2 = sum(d_loss_d_y_pred)
            # 返回 loss 对第二层输入 h 的导数：d_loss_d_h = d_loss_d_y_pred * w2。
            # d_loss_d_h 是一维数组，shape=(5,)，正是要传给第一层的梯度。
            d_loss_d_h: NDArray[np.float64] = layer2.backward(d_loss_d_y_pred)

            # 4. 再反向传播第一层：h = w1 * X + b1。
            # 第一层接收的是 loss 对自身输出 h 的导数，而不是重新计算 loss。
            # 内部保存两个标量，供 optimizer1 使用：
            # d_loss_d_w1 = sum(d_loss_d_h * X)
            # d_loss_d_b1 = sum(d_loss_d_h)
            # 返回 d_loss_d_x = d_loss_d_h * w1，一维数组，shape=(5,)。
            # X 是固定的原始数据，前面没有待训练的层，因此无需继续使用这个返回值。
            d_loss_d_x: NDArray[np.float64] = layer1.backward(d_loss_d_h)

            # 第一次打印完整数值，沿着梯度从输出走回输入。
            if epoch == 0:
                log.info('第一轮 h = %s', h)
                log.info('第一轮 Y_pred = %s', Y_pred)
                log.info('d_loss_d_y_pred = %s', d_loss_d_y_pred)
                log.info('第二层返回 d_loss_d_h = %s', d_loss_d_h)
                log.info('第二层梯度：d_loss_d_w2=%s, d_loss_d_b2=%s',
                         layer2.d_loss_d_w, layer2.d_loss_d_b)
                log.info('第一层梯度：d_loss_d_w1=%s, d_loss_d_b1=%s',
                         layer1.d_loss_d_w, layer1.d_loss_d_b)
                log.info('第一层返回 d_loss_d_x = %s（原始输入无需更新）', d_loss_d_x)

            # 5. 两层的梯度全部算完，再更新两层的权重与偏置。
            # 每个优化器执行 w -= lr * d_loss_d_w，b -= lr * d_loss_d_b。
            # 不要在 layer1.backward 之前更新 layer1 或 layer2 的参数。
            optimizer1.step()
            optimizer2.step()

        # 两层没有激活函数，展开后仍然是 y = effective_w * x + effective_b：
        # w2 * (w1 * x + b1) + b2 = (w2 * w1) * x + (w2 * b1 + b2)。
        effective_w: Scalar = layer2.w * layer1.w
        effective_b: Scalar = layer2.w * layer1.b + layer2.b
        # 各层参数不必分别等于 3 或 5，只需组合后的斜率、截距接近 3 和 5。
        log.info('组合后的模型：y = %.6fx + %.6f', effective_w, effective_b)
        log.info('最终预测：%s', Y_pred)
        log.info('真实标签：%s', Y_true)
        log.info('最终 loss：%.10f', loss)


if __name__ == '__main__':
    TwoLayerDemo().run()

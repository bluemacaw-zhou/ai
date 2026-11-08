import numpy as np
from numpy.typing import NDArray

Scalar = int | float | np.int64 | np.float32 | np.float64
SampleArray = NDArray[np.int64] | NDArray[np.float32] | NDArray[np.float64]

class Linear:
    """拟合 y = wx + b，梯度由调用方手动传入。"""

    def __init__(self, use_bias: bool = True) -> None:
        self.use_bias: bool = use_bias  # False 时模型为 y = wx，偏置固定为 0。
        self.w: Scalar = 0.1
        self.b: Scalar = 0.0
        self.d_loss_d_w: Scalar = 0.0
        self.d_loss_d_b: Scalar = 0.0
        # 缓存输入：调用 forward 前为 None，之后为 shape=(n,) 的样本数组。
        self.x: SampleArray | None = None

    def forward(self, x: SampleArray) -> NDArray[np.float64]:
        self.x = np.asarray(x)
        return self.x * self.w + (self.b if self.use_bias else 0.0)

    def backward(self, d_loss_d_y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        """输入 loss 对预测值的导数数组，返回 loss 对输入 x 的导数数组；shape=(n,)。"""
        if self.x is None:
            raise RuntimeError("请先调用 forward()")
        # 链式法则：预测值对 w 的导数是 x，对 b 的导数是 1。
        self.d_loss_d_w = float(np.sum(d_loss_d_y_pred * self.x))
        self.d_loss_d_b = float(np.sum(d_loss_d_y_pred)) if self.use_bias else 0.0
        return d_loss_d_y_pred * self.w

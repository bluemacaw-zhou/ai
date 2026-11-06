import numpy as np

class Linear:
    """拟合 y = wx + b，梯度由调用方手动传入。"""

    def __init__(self):
        self.w = 0.1
        self.b = 0.0
        self.dw = 0.0
        self.db = 0.0
        self.x = None

    def forward(self, x):
        self.x = np.asarray(x)
        return self.x * self.w + self.b

    def backward(self, dout):
        if self.x is None:
            raise RuntimeError("请先调用 forward()")
        self.dw = float(np.sum(dout * self.x))
        self.db = float(np.sum(dout))
        return dout * self.w
from tool.linear import Linear


class SGD:
    """根据 Linear 保存的梯度更新 w、b。"""

    def __init__(self, model: Linear, lr: float = 0.01) -> None:
        self.model: Linear = model
        self.lr: float = lr

    def zero_grad(self) -> None:
        self.model.d_loss_d_w = 0.0
        self.model.d_loss_d_b = 0.0

    def step(self) -> None:
        self.model.w -= self.lr * self.model.d_loss_d_w
        if self.model.use_bias:
            self.model.b -= self.lr * self.model.d_loss_d_b

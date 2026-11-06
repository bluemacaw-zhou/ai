class SGD:
    """根据 Linear 保存的梯度更新 w、b。"""

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        self.model.dw = 0.0
        self.model.db = 0.0

    def step(self):
        self.model.w -= self.lr * self.model.dw
        self.model.b -= self.lr * self.model.db
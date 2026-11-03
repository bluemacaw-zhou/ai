"""通过相邻轮次的损失变化监控过拟合趋势。"""

from dataclasses import dataclass


@dataclass
class OverfittingMonitor:
    """识别验证损失先降后升，同时训练损失继续下降的过拟合趋势。"""

    patience: int
    minimum_loss_change: float
    previous_training_loss: float | None = None
    previous_validation_loss: float | None = None
    validation_has_decreased: bool = False
    consecutive_overfitting_epochs: int = 0

    def should_stop(self, training_loss: float, validation_loss: float) -> bool:
        """比较相邻两轮损失；单轮波动不停止，平台期不计入过拟合。"""
        if self.previous_validation_loss is not None:
            # 先观察到验证损失下降，之后的上升才符合“由下降转为上升”。
            if validation_loss < self.previous_validation_loss - self.minimum_loss_change:
                self.validation_has_decreased = True
            training_is_decreasing = (
                training_loss < self.previous_training_loss - self.minimum_loss_change
            )
            validation_is_increasing = (
                validation_loss > self.previous_validation_loss + self.minimum_loss_change
            )
            # 两个趋势必须在同一轮同时成立；中间中断则重新累计连续轮数。
            if self.validation_has_decreased and training_is_decreasing and validation_is_increasing:
                self.consecutive_overfitting_epochs += 1
            else:
                self.consecutive_overfitting_epochs = 0
        # 第一轮只建立基准，不凭一个损失数值判断趋势。
        self.previous_training_loss = training_loss
        self.previous_validation_loss = validation_loss
        return self.consecutive_overfitting_epochs >= self.patience

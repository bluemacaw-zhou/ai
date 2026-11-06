"""入口：使用本地中文 BERT 和评论数据集训练、验证二分类模型。"""
from trainer import Trainer
from training_config import TrainingConfig


def main() -> None:
    """解析启动参数，再按配置执行训练。"""
    Trainer(TrainingConfig.parse_arguments()).run()


# 直接运行脚本才启动训练；被其他文件导入时不会加载数据或模型。
if __name__ == "__main__":
    main()

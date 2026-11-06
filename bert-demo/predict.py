"""入口：使用训练好的分类层判断单条语句的情感倾向。

用法：
    python predict.py --text "这家酒店服务很好"   # 判断一条后退出
    python predict.py                             # 交互模式，输入 q 退出
"""
import argparse
from pathlib import Path

from ai_starter import get_logger

from model_loader import TrainedModelLoader
from training_config import TrainingConfig

log = get_logger(__name__)


def predict_sentence(loader: TrainedModelLoader, text: str) -> None:
    """预测一条语句并记录结果。"""
    prediction = loader.predict([text])[0]
    log.info("模型判定：%s（置信度 %.4f）文本：%s", prediction.label_name, prediction.confidence, prediction.text)


def run_interactive(loader: TrainedModelLoader) -> None:
    """循环读取输入；空行跳过，输入 q 退出。"""
    while True:
        text = input("请输入测试语句（输入 q 退出）：").strip()
        if text.lower() == "q":
            log.info("测试结束")
            return
        if text:
            predict_sentence(loader, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用训练好的模型判断单条语句的情感倾向。")
    parser.add_argument(
        "--checkpoint", type=Path, default=TrainingConfig().checkpoint_directory / "best_classifier.pth"
    )
    parser.add_argument("--text", type=str, default=None, help="待判断的语句；省略则进入交互模式")
    arguments = parser.parse_args()

    loader = TrainedModelLoader(arguments.checkpoint)
    if arguments.text:
        predict_sentence(loader, arguments.text)
    else:
        run_interactive(loader)


if __name__ == "__main__":
    main()

"""下载 ChnSentiCorp 数据集并保存到本地。"""

from pathlib import Path

from ai_starter import get_logger
from datasets import load_dataset

log = get_logger(__name__)


def main():
    dataset_id = "lansinuote/ChnSentiCorp"
    save_path = Path(__file__).resolve().parent / "data" / "ChnSentiCorp"

    log.info(f"开始下载数据集：{dataset_id}")
    dataset = load_dataset(dataset_id)
    dataset.save_to_disk(str(save_path))

    log.info(f"数据集已保存至：{save_path}")
    for split, data in dataset.items():
        log.info(f"{split}：{len(data)} 条数据")


if __name__ == "__main__":
    main()

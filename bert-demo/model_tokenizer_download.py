"""下载中文 BERT 模型和分词器并保存到本地。"""

# 如果 Python 下载不稳定，也可以在浏览器中下载以下两个文件：
# 模型权重（约 412 MB）：
# https://huggingface.co/google-bert/bert-base-chinese/resolve/main/model.safetensors?download=true
# 模型结构配置：
# https://huggingface.co/google-bert/bert-base-chinese/resolve/main/config.json?download=true
# 将 model.safetensors 和 config.json 放到项目的 model/bert-base-chinese/ 目录，
# 与已下载的 vocab.txt、tokenizer_config.json、special_tokens_map.json 放在一起。
# 不需要再下载 pytorch_model.bin、TensorFlow 或 Flax 格式的权重。
# 浏览器下载不会续传 Python 已下载的缓存；使用浏览器下载前可先在 PyCharm 停止下载进程。

from pathlib import Path

from ai_starter import get_logger
from transformers import BertModel, BertTokenizer

log = get_logger(__name__)


def main():
    model_id = "google-bert/bert-base-chinese"
    save_path = Path(__file__).resolve().parent / "model" / "bert-base-chinese"

    try:
        BertTokenizer.from_pretrained(str(save_path), local_files_only=True)
    except (OSError, ValueError, TypeError):
        log.info(f"本地分词器缺失或无法加载，开始下载：{model_id}")
        tokenizer = BertTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(str(save_path))
        log.info(f"分词器已保存至：{save_path}")
    else:
        log.info(f"本地分词器验证通过，跳过下载：{save_path}")

    log.info(f"开始下载模型：{model_id}")
    model = BertModel.from_pretrained(model_id)
    model.save_pretrained(str(save_path))
    log.info(f"模型已保存至：{save_path}")


if __name__ == "__main__":
    main()

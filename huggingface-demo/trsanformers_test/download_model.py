"""下载 GPT-2 模型到项目 model 目录，并验证本地加载。"""

import os
import sys
from pathlib import Path

# 本机代理下 Xet 下载出现 TLS 错误，使用标准 HTTP 下载。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ai_starter import Config, get_logger
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
MODEL_DIR = PROJECT_ROOT / "model" / MODEL_NAME

Config().load_file(str(PROJECT_ROOT / "config.yaml"))
log = get_logger(__name__)


def main() -> int:
    log.info("开始下载模型：%s", MODEL_NAME)
    log.info("本地保存目录：%s", MODEL_DIR)
    try:
        # 直接保存模型文件，避免 cache_dir 产生多层 snapshots 缓存目录。
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=MODEL_DIR,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model*.bin"],
        )
        log.info("下载完成，验证模型和分词器能否从本地加载。")
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        log.info("本地加载成功：模型=%s，分词器=%s", type(model).__name__, type(tokenizer).__name__)
    except Exception as exc:
        log.error("模型下载或加载失败（%s）：%s", type(exc).__name__, exc)
        return 1
    log.info("模型和分词器已保存到：%s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

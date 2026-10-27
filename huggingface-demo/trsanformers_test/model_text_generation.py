"""使用已下载的本地模型离线生成文本。"""

import sys
from pathlib import Path

import torch
from ai_starter import Config, get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "uer" / "gpt2-chinese-cluecorpussmall"
PROMPT = "你好，我是一款语言模型，"

Config().load_file(str(PROJECT_ROOT / "config.yaml"))
log = get_logger(__name__)


def main() -> int:
    # 当前 Mac 没有 CUDA 时使用 CPU，也兼容带 NVIDIA GPU 的机器。
    device = 0 if torch.cuda.is_available() else -1
    log.info("本地模型目录：%s", MODEL_DIR)
    log.info("推理设备：%s", "cuda:0" if device == 0 else "cpu")
    try:
        # 禁止自动联网下载，以验证本地文件是否完整可用。
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        log.info("输入：%s", PROMPT)
        output = generator(
            PROMPT,
            max_new_tokens=50,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
            clean_up_tokenization_spaces=True,
        )
        log.info("生成结果：%s", output[0]["generated_text"])
    except Exception as exc:
        log.error("本地推理失败（%s）：%s", type(exc).__name__, exc)
        return 1
    log.info("本地离线推理成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

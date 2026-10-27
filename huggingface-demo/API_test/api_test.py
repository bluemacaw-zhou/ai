"""使用环境变量中的 HF Token 测试在线模型推理。"""

import os
import sys
from pathlib import Path

import requests
from ai_starter import Config, get_logger

# 使用脚本路径定位配置，兼容 PyCharm 不同的工作目录设置。
Config().load_file(str(Path(__file__).resolve().parents[1] / "config.yaml"))
log = get_logger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"
API_URL = "https://router.huggingface.co/v1/chat/completions"
QUESTION = "What is the capital of France? Answer in one short sentence."


def main() -> int:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("hf_token") or "").strip()
    if not token:
        log.error("未找到 HF_TOKEN 或 hf_token 环境变量。")
        return 1

    log.info("模型：%s", MODEL_ID)
    log.info("接口：%s", API_URL)
    log.info("问题：%s", QUESTION)
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": QUESTION}],
                "max_tokens": 64,
                "stream": False,
            },
            timeout=(10, 60),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        log.error("请求失败：%s", str(exc).replace(token, "[REDACTED]"))
        return 1

    log.info("HTTP 状态码：%s", response.status_code)
    if not response.ok or response.is_redirect:
        log.error("接口响应：%s", response.text.replace(token, "[REDACTED]"))
        return 1
    try:
        result = response.json()
    except ValueError:
        log.error("接口未返回 JSON 推理结果。")
        return 1
    try:
        answer = result["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty answer")
    except (KeyError, IndexError, TypeError, ValueError):
        log.error("接口未返回有效的对话回答。")
        return 1
    log.info("回答：%s", answer.replace(token, "[REDACTED]"))
    log.info("模型推理成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

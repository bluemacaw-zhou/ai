"""HTTP client adapter for Hugging Face Hub downloads."""

import os
from pathlib import Path
from typing import Any

import httpx
from ai_starter import Config, HttpClientFactory


class HuggingFaceHttpClientAdapter:
    """把 ai-starter 的 HTTP client 配置适配给 huggingface_hub。"""

    def __init__(self, *, project_root: Path, cache_dir: Path) -> None:
        self._project_root = project_root
        self._cache_dir = cache_dir

    def configure(self) -> Any:
        # Hugging Face 的默认缓存目录在用户 home 下。
        # demo 固定写到项目 .cache/huggingface，方便知道数据集是否已经下载，也方便清理。
        os.environ.setdefault("HF_HOME", str(self._cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(self._cache_dir / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(self._cache_dir / "datasets"))

        # Windows 普通权限下 symlink 经常不可用；关闭后 HF 会退化为普通文件缓存。
        # 这会多占一点空间，但对 demo 更稳定。
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        hf_token = self._normalize_token()

        # 这两个包只在真正访问 Hugging Face 时需要。
        # 延迟导入可以避免纯加载模块时就触发 HF SDK 的全局状态初始化。
        import datasets
        from huggingface_hub import set_client_factory

        # Config 显式加载 function-call-demo/config.yaml，确保代理和 SSL 配置来自当前 demo 项目。
        Config().load_file(str(self._project_root / "config.yaml"))

        # huggingface_hub 1.x 使用全局共享 httpx.Client。
        # set_client_factory 是它提供的扩展点，用来接入公司代理、证书策略和超时配置。
        set_client_factory(self._create_client)
        return datasets.DownloadConfig(cache_dir=str(self._cache_dir), max_retries=5, token=hf_token)

    @staticmethod
    def close() -> None:
        from huggingface_hub import close_session

        close_session()

    @staticmethod
    def _create_client() -> httpx.Client:
        from huggingface_hub.utils._http import hf_request_event_hook

        # 复用 ai-starter 里的代理、超时和 SSL 配置，再补上 Hugging Face 自己的 request hook。
        client = HttpClientFactory.create()
        client.event_hooks["request"].append(hf_request_event_hook)
        client.follow_redirects = True
        return client

    @staticmethod
    def _normalize_token() -> str | None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            return None

        # PyCharm 环境变量里容易误填引号；这里做一次容错清理。
        # 同时写回两个变量名，兼容 huggingface_hub 和其它生态库的读取习惯。
        token = token.strip().strip('"').strip("'")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

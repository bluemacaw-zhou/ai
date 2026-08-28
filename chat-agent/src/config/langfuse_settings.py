"""Langfuse 可观测性配置。

配置统一来自项目 ``config.yaml`` 的 ``observability.langfuse`` 节点；敏感值可用
``${ENV_NAME}`` 引用环境变量，不额外读取或生成其他配置文件。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .app_config import Config

_ENV_PLACEHOLDER = re.compile(r"^\$\{(\w+)\}$")
_CONFIG_ROOT = "observability.langfuse"


@dataclass(frozen=True)
class LangfuseSettings:
    """Langfuse 客户端及 LangChain callback 的进程级配置。"""

    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None
    environment: str = "development"
    release: str | None = None
    sample_rate: float = 1.0
    timeout: int = 5
    flush_at: int = 50
    flush_interval: float = 5.0
    debug: bool = False
    bypass_proxy: bool = True
    tags: tuple[str, ...] = ()
    instance_ip: str | None = None

    def validate(self) -> None:
        """校验显式启用时的必需项和客户端参数。"""
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("public_key", self.public_key),
                    ("secret_key", self.secret_key),
                    ("base_url", self.base_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Langfuse 已启用，但缺少配置: " + ", ".join(missing)
                )
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("observability.langfuse.sample_rate 必须在 0 到 1 之间")
        if self.timeout <= 0:
            raise ValueError("observability.langfuse.timeout 必须大于 0")
        if self.flush_at <= 0:
            raise ValueError("observability.langfuse.flush_at 必须大于 0")
        if self.flush_interval <= 0:
            raise ValueError("observability.langfuse.flush_interval 必须大于 0")


    @classmethod
    def load(cls) -> "LangfuseSettings":
        cfg: dict[str, Any] = Config().get(_CONFIG_ROOT, {}) or {}
        settings = cls(
            enabled=bool(cfg.get("enabled", False)),
            public_key=_resolve_value(
                cfg.get("public_key"), fallback_env=("LANGFUSE_PUBLIC_KEY",)
            ),
            secret_key=_resolve_value(
                cfg.get("secret_key"), fallback_env=("LANGFUSE_SECRET_KEY",)
            ),
            base_url=_resolve_value(
                cfg.get("base_url") or cfg.get("host"),
                fallback_env=("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
            ),
            environment=str(cfg.get("environment", "development")),
            release=_resolve_value(cfg.get("release"), fallback_env=("LANGFUSE_RELEASE",)),
            sample_rate=float(cfg.get("sample_rate", 1.0)),
            timeout=int(cfg.get("timeout", 5)),
            flush_at=int(cfg.get("flush_at", 50)),
            flush_interval=float(cfg.get("flush_interval", 5.0)),
            debug=bool(cfg.get("debug", False)),
            bypass_proxy=bool(cfg.get("bypass_proxy", True)),
            tags=tuple(str(tag) for tag in (cfg.get("tags", []) or [])),
            instance_ip=_resolve_value(
                cfg.get("instance_ip"), fallback_env=("A2UI_INSTANCE_IP",)
            ),
        )
        settings.validate()
        return settings


def _resolve_value(value: Any, *, fallback_env: tuple[str, ...]) -> str | None:
    """解析配置值中的环境变量占位符，并支持标准 Langfuse 环境变量兜底。"""
    if value is not None:
        text = str(value).strip()
        match = _ENV_PLACEHOLDER.fullmatch(text)
        if not match:
            return text or None
        resolved = os.getenv(match.group(1))
        if resolved:
            return resolved
    for env_name in fallback_env:
        resolved = os.getenv(env_name)
        if resolved:
            return resolved
    return None


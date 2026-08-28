"""LangChain chat model 的 LLM 配置（支持多个命名 agent，扁平自包含）。

从 ag-ui-python-backend 的 config/llm_settings.py 移植并改造：
源项目只有单个 `ag_ui_backend.llm` 配置；这里改为支持多个命名 agent，
每个 agent 在 config.yaml 里自包含（不共享 defaults，一目了然）。

config.yaml 结构::

    llm:
      agents:
        default:
          api_base: "https://.../v1"
          model: "qwen-plus"
          api_key: "${DASHSCOPE_API_KEY}"
          temperature: 0
          max_tokens: null
          http:
            proxy: {enabled: false}
            timeout: 60
            verify_ssl: true
        planner:  {api_base: ..., model: "qwen-max",  ...}
        reviewer: {api_base: ..., model: "qwen-plus", ...}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Dict

from .app_config import Config

_ENV_PLACEHOLDER = re.compile(r"^\$\{(\w+)\}$")

# 配置根路径。改这里即可整体迁移配置节点。
_AGENTS_ROOT = "llm.agents"


@dataclass(frozen=True)
class LlmHttpSettings:
    """LLM 客户端使用的 HTTP 选项。"""

    proxy_enabled: bool = False
    proxy_http: str | None = None
    proxy_https: str | None = None
    timeout: float = 60.0
    verify_ssl: bool = True


@dataclass(frozen=True)
class LlmSettings:
    """单个 agent 的 LLM 配置。"""

    name: str
    api_base: str
    model: str
    provider: str = "openai_compatible"
    api_key: str | None = None
    pkey: str | None = None
    source: str | None = None
    thinking: bool | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    http: LlmHttpSettings = field(default_factory=LlmHttpSettings)


def get_agent_names() -> list[str]:
    """返回配置中定义的全部 agent 名称。"""
    agents = Config().get(_AGENTS_ROOT, {}) or {}
    return list(agents.keys())


def get_llm_settings(name: str = "default") -> LlmSettings:
    """读取指定命名 agent 的 LLM 配置（agent 自包含）。

    Args:
        name: agent 名称，需与 config.yaml 中 ``llm.agents`` 下的 key 对应。

    Raises:
        KeyError: 指定的 agent 未在配置中定义。
    """
    agents: Dict[str, Any] = Config().get(_AGENTS_ROOT, {}) or {}

    if name not in agents:
        available = ", ".join(agents.keys()) or "(无)"
        raise KeyError(
            f"未找到名为 '{name}' 的 agent 配置。可用 agent: {available}"
        )

    cfg: Dict[str, Any] = agents[name] or {}
    http_cfg: Dict[str, Any] = cfg.get("http", {}) or {}
    proxy_cfg: Dict[str, Any] = http_cfg.get("proxy", {}) or {}

    return LlmSettings(
        name=name,
        api_base=cfg.get("api_base", ""),
        model=cfg.get("model", "qwen-plus"),
        provider=str(cfg.get("provider", "openai_compatible")),
        api_key=_resolve_env_placeholder(cfg.get("api_key")),
        pkey=_resolve_env_placeholder(cfg.get("pkey")),
        source=cfg.get("source"),
        thinking=_optional_bool(cfg.get("thinking")),
        temperature=float(cfg.get("temperature", 0)),
        max_tokens=_optional_int(cfg.get("max_tokens")),
        http=LlmHttpSettings(
            proxy_enabled=bool(proxy_cfg.get("enabled", False)),
            proxy_http=proxy_cfg.get("http"),
            proxy_https=proxy_cfg.get("https"),
            timeout=float(http_cfg.get("timeout", 60)),
            verify_ssl=bool(http_cfg.get("verify_ssl", True)),
        ),
    )


def get_all_llm_settings() -> Dict[str, LlmSettings]:
    """读取全部命名 agent 的 LLM 配置，返回 {name: LlmSettings}。"""
    return {name: get_llm_settings(name) for name in get_agent_names()}


def _optional_int(value: Any) -> int | None:
    """把配置值转成 int，None/空 返回 None。"""
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    """Return None for missing values, otherwise parse common bool forms."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _resolve_env_placeholder(value: str | None) -> str | None:
    """解析 ${ENV_NAME} 占位符，否则原样返回配置值。"""
    if not value:
        return value
    match = _ENV_PLACEHOLDER.match(value.strip())
    if not match:
        return value
    return getenv(match.group(1))

"""配置化的 LangChain chat model（支持多个命名 agent）。

从 ag-ui-python-backend 的 services/chat_model.py 移植并改造：
源项目只构建单个 ChatModel；这里提供 ChatModelRegistry，按 agent 名
构建并缓存多个 ChatOpenAI，供 LangGraph 直接使用（bind_tools / ainvoke）。
"""

from __future__ import annotations

import os

os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from .http_client_factory import HttpClientFactory
from .llm_settings import (
    LlmSettings,
    get_all_llm_settings,
    get_llm_settings,
)

# 代理由 HttpClientFactory 按配置显式控制 (proxy_enabled)，从不依赖 httpx 的系统代理
# 自动探测。langchain-openai 默认注入自定义 transport 加 TCP keepalive，会关闭那个探测
# 并打出无关警告。这里关掉它的 keepalive 注入，消除噪音警告，且不改变现有代理行为。
os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")


class ChatModel:
    """构建并持有单个命名 agent 的 OpenAI 兼容 LangChain chat model。"""

    def __init__(self, settings: LlmSettings):
        self._settings = settings
        self._model = self._build_model(settings)

    @property
    def name(self) -> str:
        return self._settings.name

    @property
    def settings(self) -> LlmSettings:
        return self._settings

    @property
    def model(self) -> ChatOpenAI:
        return self._model

    def bind_tools(self, tools: list[BaseTool]) -> LanguageModelLike:
        return self._model.bind_tools(tools)

    @staticmethod
    def _build_model(settings: LlmSettings) -> ChatOpenAI:
        if settings.provider != "openai_compatible":
            raise ValueError(f"Unsupported LLM provider: {settings.provider}")
        client = HttpClientFactory.create_async_client(settings)

        kwargs: dict = {
            "model": settings.model,
            "base_url": settings.api_base,
            "api_key": settings.api_key,
            "temperature": settings.temperature,
            "http_async_client": client,
        }
        if settings.max_tokens is not None:
            kwargs["max_tokens"] = settings.max_tokens
        return ChatOpenAI(**kwargs)


class ChatModelRegistry:
    """按 agent 名构建并缓存多个 ChatModel。

    进程级单例：通过 :meth:`instance` 获取全局唯一实例，任意模块（尤其是
    LangGraph 节点）可以直接按需加载所需的命名模型，而不必让模型实例沿着
    构造函数逐层往下传递。``AiPadBootstrap.build_runtime()`` 负责在进程启动时
    首次创建该单例；此后 :meth:`instance` 返回同一个对象。

    Examples:
        >>> registry = ChatModelRegistry.instance()
        >>> planner = registry.model("planner")        # ChatOpenAI，LangGraph 可用
        >>> reviewer = registry.model("reviewer")
        >>> registry.names()                            # 已配置的全部 agent 名
    """

    _instance: "ChatModelRegistry | None" = None

    def __init__(self, *, eager: bool = False):
        """初始化注册表。

        Args:
            eager: True 则立即构建全部 agent 模型；默认懒加载（首次访问才构建）。
        """
        self._cache: dict[str, ChatModel] = {}
        if eager:
            self.build_all()

    @classmethod
    def instance(cls, *, eager: bool = False) -> "ChatModelRegistry":
        """返回进程级单例；首次调用时按 ``eager`` 构建。

        后续调用忽略 ``eager`` 参数（单例已存在则直接复用），仅在单例尚未创建时
        生效。
        """
        if cls._instance is None:
            cls._instance = cls(eager=eager)
        return cls._instance

    def get(self, name: str = "default") -> ChatModel:
        """返回指定 agent 的 ChatModel（懒加载 + 缓存）。"""
        if name not in self._cache:
            self._cache[name] = ChatModel(get_llm_settings(name))
        return self._cache[name]

    def model(self, name: str = "default") -> ChatOpenAI:
        """返回指定 agent 的底层 ChatOpenAI（LangGraph 直接可用）。"""
        return self.get(name).model

    def bind_tools(self, name: str, tools: list[BaseTool]) -> LanguageModelLike:
        """返回指定 agent 绑定工具后的模型。"""
        return self.get(name).bind_tools(tools)

    def build_all(self) -> dict[str, ChatModel]:
        """构建全部已配置 agent 的模型并缓存，返回 {name: ChatModel}。"""
        for name, settings in get_all_llm_settings().items():
            if name not in self._cache:
                self._cache[name] = ChatModel(settings)
        return dict(self._cache)

    def names(self) -> list[str]:
        """返回全部已缓存的 agent 名。"""
        return list(self._cache.keys())

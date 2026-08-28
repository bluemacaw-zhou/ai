"""LLM / MCP 集成共用的 HTTP 客户端构建工厂。

从 ag-ui-python-backend 的 common/http_client_factory.py 移植。
调用方既可以传入 LlmSettings / LlmHttpSettings 对象，也可以直接传 HTTP 选项；
直接传入的关键字参数优先级高于 settings 里的值。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import httpx

if TYPE_CHECKING:
    from .llm_settings import LlmHttpSettings, LlmSettings


class HttpClientFactory:
    """构建配置化的同步 / 异步 HTTP 客户端。"""

    @overload
    @classmethod
    def create_sync_client(
        cls,
        settings: "LlmSettings | LlmHttpSettings",
        *,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.Client:
        ...

    @overload
    @classmethod
    def create_sync_client(
        cls,
        settings: None = None,
        *,
        proxy_enabled: bool = False,
        proxy_http: str | None = None,
        proxy_https: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.Client:
        ...

    @classmethod
    def create_sync_client(
        cls,
        settings: Any = None,
        *,
        proxy_enabled: bool | None = None,
        proxy_http: str | None = None,
        proxy_https: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        verify_ssl: bool | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.Client:
        options = cls._resolve_options(
            settings=settings,
            proxy_enabled=proxy_enabled,
            proxy_http=proxy_http,
            proxy_https=proxy_https,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        return httpx.Client(
            headers=headers,
            auth=auth,
            proxy=options["proxy"],
            verify=options["verify_ssl"],
            timeout=options["timeout"],
            trust_env=trust_env,
        )

    @overload
    @classmethod
    def create_async_client(
        cls,
        settings: "LlmSettings | LlmHttpSettings",
        *,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.AsyncClient:
        ...

    @overload
    @classmethod
    def create_async_client(
        cls,
        settings: None = None,
        *,
        proxy_enabled: bool = False,
        proxy_http: str | None = None,
        proxy_https: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.AsyncClient:
        ...

    @classmethod
    def create_async_client(
        cls,
        settings: Any = None,
        *,
        proxy_enabled: bool | None = None,
        proxy_http: str | None = None,
        proxy_https: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        verify_ssl: bool | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        trust_env: bool = True,
    ) -> httpx.AsyncClient:
        options = cls._resolve_options(
            settings=settings,
            proxy_enabled=proxy_enabled,
            proxy_http=proxy_http,
            proxy_https=proxy_https,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            proxy=options["proxy"],
            verify=options["verify_ssl"],
            timeout=options["timeout"],
            trust_env=trust_env,
        )

    @classmethod
    def _resolve_options(
        cls,
        *,
        settings: Any,
        proxy_enabled: bool | None,
        proxy_http: str | None,
        proxy_https: str | None,
        timeout: float | httpx.Timeout | None,
        verify_ssl: bool | None,
    ) -> dict[str, Any]:
        http_settings = cls._http_settings(settings)
        resolved_proxy_enabled = cls._value(
            proxy_enabled, http_settings, "proxy_enabled", False
        )
        resolved_proxy_http = cls._value(proxy_http, http_settings, "proxy_http", None)
        resolved_proxy_https = cls._value(
            proxy_https, http_settings, "proxy_https", None
        )
        return {
            "proxy": (
                resolved_proxy_https or resolved_proxy_http
                if resolved_proxy_enabled
                else None
            ),
            "timeout": cls._value(timeout, http_settings, "timeout", None),
            "verify_ssl": cls._value(verify_ssl, http_settings, "verify_ssl", True),
        }

    @staticmethod
    def _http_settings(settings: Any) -> Any:
        if settings is None:
            return None
        return getattr(settings, "http", settings)

    @staticmethod
    def _value(explicit: Any, settings: Any, name: str, default: Any) -> Any:
        if explicit is not None:
            return explicit
        if settings is None:
            return default
        return getattr(settings, name, default)

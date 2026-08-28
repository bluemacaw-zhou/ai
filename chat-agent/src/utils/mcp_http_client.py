"""MCP 传输层 http client。"""

from __future__ import annotations

import httpx

from config import HttpClientFactory


def create_mcp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """构建调用 MCP server 用的 http client (不走环境代理，不校验证书)。

    直接作为 langchain_mcp_adapters 的 httpx_client_factory 传入。
    """
    return HttpClientFactory.create_async_client(
        headers=headers,
        timeout=timeout,
        auth=auth,
        verify_ssl=False,
        trust_env=False,
    )

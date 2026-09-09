"""OpenViking Server HTTP integration configured through ``config.yaml``."""

from __future__ import annotations

from typing import Any

import httpx

from config.app_config import Config
from config.http_client_factory import HttpClientFactory


class OpenVikingClient:
    """OpenViking 检索和内容读取客户端，不包含任何 RAG 编排逻辑。"""

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 account: str | None = None, user: str | None = None,
                 timeout_seconds: float | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        config = Config()
        self._base_url = (base_url or config.get("rag.openviking.base_url", "")).rstrip("/")
        self._api_key = api_key or config.get("rag.openviking.api_key")
        self._account = account or config.get("rag.openviking.account")
        self._user = user or config.get("rag.openviking.user")
        self._timeout_seconds = float(timeout_seconds or config.get("rag.openviking.timeout_seconds", 45))
        self._client = client
        if not self._base_url:
            raise ValueError("缺少 rag.openviking.base_url 配置")

    async def search(self, query: str, *, session_id: str | None = None,
                     target_uri: str | list[str] | None = None,
                     tags: list[str] | None = None, limit: int | None = None,
                     context_types: list[str] | None = None) -> dict[str, Any]:
        config = Config()
        payload: dict[str, Any] = {
            "query": query,
            "target_uri": target_uri if target_uri is not None else config.get("rag.openviking.target_uri"),
            "limit": limit or int(config.get("rag.openviking.limit", 6)),
            "context_type": context_types or config.get("rag.openviking.context_types", ["resource"]),
        }
        if session_id:
            payload["session_id"] = session_id
        if tags:
            payload["tags"] = tags
        return await self._request("POST", "/api/v1/search/search", json=payload)

    async def load_context(self, context: dict[str, Any]) -> dict[str, Any]:
        uri = str(context.get("uri") or "")
        if not uri:
            return {**context, "content": ""}
        endpoint = "/api/v1/content/read" if int(context.get("level") or 0) >= 2 else "/api/v1/content/overview"
        payload = await self._request("GET", endpoint, params={"uri": uri})
        return {**context, "content": self._unwrap_content(payload)}

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        owns_client = self._client is None
        client = self._client or HttpClientFactory.create_async_client(timeout=self._timeout_seconds, headers=self._headers())
        try:
            response = await client.request(method, f"{self._base_url}{path}", **kwargs)
            response.raise_for_status()
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        finally:
            if owns_client:
                await client.aclose()

    def _headers(self) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._account:
            headers["X-OpenViking-Account"] = self._account
        if self._user:
            headers["X-OpenViking-User"] = self._user
        return headers or None

    @staticmethod
    def _unwrap_content(payload: dict[str, Any]) -> str:
        value = payload.get("result", payload)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("content", "overview", "text"):
                if value.get(key) is not None:
                    return str(value[key])
        return str(value or "")

"""Judge0 CE HTTP integration configured through ``config.yaml``."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config.app_config import Config
from config.http_client_factory import HttpClientFactory

_PENDING_STATUS_IDS = {1, 2}
_LANGUAGE_IDS = {"python": 71, "python3": 71, "java": 62}


class Judge0Client:
    """Judge0 CE 的基础设施客户端，不包含任何图编排逻辑。"""

    def __init__(self, *, base_url: str | None = None, auth_token: str | None = None,
                 poll_interval_seconds: float | None = None,
                 poll_timeout_seconds: float | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        config = Config()
        self._base_url = (base_url or config.get("sandbox.judge0.base_url", "")).rstrip("/")
        self._auth_token = auth_token or config.get("sandbox.judge0.auth_token")
        self._poll_interval = float(poll_interval_seconds or config.get("sandbox.judge0.poll_interval_seconds", 0.5))
        self._poll_timeout = float(poll_timeout_seconds or config.get("sandbox.judge0.poll_timeout_seconds", 30))
        self._client = client
        if not self._base_url:
            raise ValueError("缺少 sandbox.judge0.base_url 配置")

    async def execute(self, *, code: str, language: str = "python", stdin: str = "") -> dict[str, Any]:
        language_id = _LANGUAGE_IDS.get(language.lower())
        if language_id is None:
            raise ValueError(f"暂不支持的 Judge0 语言: {language}")
        owns_client = self._client is None
        client = self._client or HttpClientFactory.create_async_client(
            timeout=self._poll_timeout + 10, headers=self._headers()
        )
        try:
            submitted = await client.post(
                f"{self._base_url}/submissions",
                params={"base64_encoded": "false", "wait": "false"},
                json={"source_code": code, "language_id": language_id, "stdin": stdin},
            )
            submitted.raise_for_status()
            token = str(submitted.json().get("token") or "")
            if not token:
                raise RuntimeError("Judge0 未返回 submission token")
            return await self._wait_for_result(client, token)
        finally:
            if owns_client:
                await client.aclose()

    async def _wait_for_result(self, client: httpx.AsyncClient, token: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self._poll_timeout
        while True:
            response = await client.get(
                f"{self._base_url}/submissions/{token}",
                params={"base64_encoded": "false", "fields": "*"},
            )
            response.raise_for_status()
            result = response.json()
            if int((result.get("status") or {}).get("id") or 0) not in _PENDING_STATUS_IDS:
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Judge0 执行超时，submission={token}")
            await asyncio.sleep(self._poll_interval)

    def _headers(self) -> dict[str, str] | None:
        return {"X-Auth-Token": self._auth_token} if self._auth_token else None

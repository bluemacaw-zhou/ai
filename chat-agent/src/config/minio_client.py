"""MinIO operations owned by SandboxAgent; credentials never enter Judge0 or prompts."""

from __future__ import annotations

import io
import json
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

try:
    from minio import Minio
except ImportError:  # pragma: no cover - depends on deployment extras
    Minio = None  # type: ignore[assignment,misc]

from config.app_config import Config


class MinioClient:
    """Reads JSON metadata, prepares temporary downloads, and publishes JSON results."""

    def __init__(self) -> None:
        if Minio is None:
            raise RuntimeError("未安装 minio 依赖；请在项目目录执行 uv sync 后重试")
        config = Config()
        endpoint = str(config.get("sandbox.minio.endpoint", ""))
        if not endpoint:
            raise ValueError("缺少 sandbox.minio.endpoint 配置")
        self._client = Minio(
            endpoint,
            access_key=config.get("sandbox.minio.access_key"),
            secret_key=config.get("sandbox.minio.secret_key"),
            secure=bool(config.get("sandbox.minio.secure", False)),
        )
        self._default_bucket = config.get("sandbox.minio.bucket")
        self._result_bucket = config.get("sandbox.minio.result_bucket") or self._default_bucket
        self._result_prefix = str(config.get("sandbox.minio.result_prefix", "sandbox-results")).strip("/")
        self._expiry_seconds = int(config.get("sandbox.minio.presigned_url_expiry_seconds", 600))

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("input_uris 仅支持 s3://bucket/object 格式的 MinIO URI")
        return parsed.netloc, parsed.path.lstrip("/")

    def read_json_metadata(self, uri: str) -> dict[str, Any]:
        """读取统一 JSON 信封，只返回其 metadata 部分给模型。"""
        bucket, object_name = self._parse_uri(uri)
        response = self._client.get_object(bucket, object_name)
        try:
            document = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
        metadata = document.get("metadata") if isinstance(document, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError(f"输入对象 {uri} 不是包含 metadata 的统一 JSON 信封")
        return metadata

    def execution_assets(self, uris: list[str]) -> list[dict[str, str]]:
        """仅在提交 Judge0 前创建预签名 URL；结果不会写入 LangGraph state。"""
        assets: list[dict[str, str]] = []
        for index, uri in enumerate(uris, start=1):
            bucket, object_name = self._parse_uri(uri)
            filename = PurePosixPath(object_name).name or f"asset_{index}.json"
            assets.append(
                {
                    "asset_id": f"asset_{index}",
                    "url": self._client.presigned_get_object(
                        bucket, object_name, expires=timedelta(seconds=self._expiry_seconds)
                    ),
                    "sandbox_path": f"/tmp/sandbox_inputs/asset_{index}_{filename}",
                }
            )
        return assets

    def put_result_json(self, document: dict[str, Any]) -> str:
        """将结果信封上传 MinIO，A2A 最终只返回该不可变 URI。"""
        if not self._result_bucket:
            raise ValueError("缺少 sandbox.minio.result_bucket 或 sandbox.minio.bucket 配置")
        object_name = f"{self._result_prefix}/{uuid4().hex}.json"
        payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
        self._client.put_object(
            str(self._result_bucket),
            object_name,
            io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )
        return f"s3://{self._result_bucket}/{object_name}"

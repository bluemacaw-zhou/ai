"""节点 1：从统一 JSON 信封中提取 metadata。"""

from __future__ import annotations

from typing import Any

from config.minio_client import MinioClient
from graph.sandbox_harness_graph.sandbox_harness_state import SandboxHarnessState


class LoadDataMetadata:
    """完整 data 留在 MinIO；只有 metadata 进入模型上下文和 LangGraph state。"""

    def __init__(self, minio: MinioClient | None = None) -> None:
        self._minio = minio

    async def load(self, state: SandboxHarnessState) -> dict[str, Any]:
        if self._minio is None:
            self._minio = MinioClient()
        metadata = [
            {"asset_id": f"asset_{index}", "metadata": self._minio.read_json_metadata(uri)}
            for index, uri in enumerate(state.get("input_uris") or [], start=1)
        ]
        return {"data_metadata": metadata}

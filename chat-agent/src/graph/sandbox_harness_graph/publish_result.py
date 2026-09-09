"""节点 5：由 SandboxAgent 服务端保存统一结果 JSON，而非由 Judge0 持有 MinIO 凭证。"""

from __future__ import annotations

from typing import Any

from config.minio_client import MinioClient
from graph.sandbox_harness_graph.sandbox_harness_state import SandboxHarnessState


class PublishResult:
    def __init__(self, minio: MinioClient | None = None) -> None:
        self._minio = minio

    async def publish_success(self, state: SandboxHarnessState) -> dict[str, Any]:
        attempt = (state.get("attempts") or [])[-1]
        document = dict(attempt["output_document"])
        metadata = dict(document["metadata"])
        metadata.update(
            {
                "kind": "sandbox_result",
                "status": "succeeded",
                "source_attempt": attempt["iteration"],
            }
        )
        document["metadata"] = metadata
        return self._append_result_uri(state, attempt, document)

    async def publish_failure(self, state: SandboxHarnessState) -> dict[str, Any]:
        attempts = state.get("attempts") or []
        document = {
            "metadata": {"kind": "sandbox_result", "status": "failed", "attempt_count": len(attempts)},
            "data": {"objective": state.get("objective"), "attempts": attempts},
        }
        return self._append_result_uri(state, attempts[-1] if attempts else {}, document)

    def _append_result_uri(
        self,
        state: SandboxHarnessState,
        target_attempt: dict[str, Any],
        document: dict[str, Any],
    ) -> dict[str, Any]:
        if self._minio is None:
            self._minio = MinioClient()
        result_uri = self._minio.put_result_json(document)
        attempts = [dict(attempt) for attempt in state.get("attempts") or []]
        if attempts:
            attempts[-1]["result_uri"] = result_uri
        return {"attempts": attempts}

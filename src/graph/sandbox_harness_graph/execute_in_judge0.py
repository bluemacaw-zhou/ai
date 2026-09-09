"""节点 4：在 Judge0 执行当前代码，并持久记录一次 Attempt。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from config.judge0_client import Judge0Client
from config.minio_client import MinioClient
from graph.sandbox_harness_graph.sandbox_harness_state import SandboxHarnessState


class ExecuteInJudge0:
    def __init__(self, judge0: Judge0Client, minio: MinioClient | None = None) -> None:
        self._judge0 = judge0
        self._minio = minio

    async def execute(self, state: SandboxHarnessState) -> dict[str, Any]:
        if self._minio is None:
            self._minio = MinioClient()
        code = str(state.get("pending_code") or "")
        language = str(state.get("pending_language") or "python")
        try:
            execution = await self._judge0.execute(
                code=self._with_input_assets(code, self._minio.execution_assets(state.get("input_uris") or [])),
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            execution = {"status": {"description": "client_error"}, "stderr": str(exc)}

        attempt = {
            "iteration": len(state.get("attempts") or []) + 1,
            "language": language,
            "code": code,
            "judge0_result": execution,
        }
        if self._judge0_succeeded(execution):
            try:
                attempt["output_document"] = self._parse_output_document(str(execution.get("stdout") or ""))
                attempt["outcome"] = "succeeded"
            except ValueError as exc:
                attempt["outcome"] = "invalid_output"
                attempt["output_error"] = str(exc)
        else:
            attempt["outcome"] = "failed"

        attempts = [*(state.get("attempts") or []), attempt]
        feedback = {
            "iteration": attempt["iteration"],
            "code": code,
            "outcome": attempt["outcome"],
            "judge0_result": execution,
            "output_error": attempt.get("output_error"),
        }
        return {
            "attempts": attempts,
            "messages": [
                *state["messages"],
                HumanMessage("上一版代码及执行结果；请仅输出修复后的代码：\n" + json.dumps(feedback, ensure_ascii=False)),
            ],
        }

    @staticmethod
    def _judge0_succeeded(execution: dict[str, Any]) -> bool:
        return int((execution.get("status") or {}).get("id") or 0) == 3

    @staticmethod
    def _parse_output_document(stdout: str) -> dict[str, Any]:
        try:
            document = json.loads(stdout.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("代码 stdout 必须是一个 JSON 信封") from exc
        if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict) or "data" not in document:
            raise ValueError("代码输出必须包含 metadata 对象和 data 字段")
        return document

    @staticmethod
    def _with_input_assets(code: str, assets: list[dict[str, str]]) -> str:
        payload = json.dumps(assets, ensure_ascii=False)
        bootstrap = f'''import json\nimport urllib.request\nfrom pathlib import Path\n\n_ASSET_SPECS = json.loads({payload!r})\nDATA_ASSETS = {{}}\nfor _asset in _ASSET_SPECS:\n    _path = Path(_asset["sandbox_path"])\n    _path.parent.mkdir(parents=True, exist_ok=True)\n    urllib.request.urlretrieve(_asset["url"], _path)\n    DATA_ASSETS[_asset["asset_id"]] = str(_path)\n\n'''
        return bootstrap + code

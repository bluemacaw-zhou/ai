"""节点 3：模型只生成待执行代码，不参与图的结束或路由决策。"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from graph.sandbox_harness_graph.sandbox_harness_state import SandboxHarnessState


class GenerateCode:
    """将模型输出解析为 pending_code；后续是否重试完全由执行状态决定。"""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def generate(self, state: SandboxHarnessState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        response = await self._model.ainvoke(state["messages"], config=config)
        language, code = self._parse(response.content)
        return {
            "pending_language": language,
            "pending_code": code,
            "messages": [*state["messages"], response],
        }

    @staticmethod
    def _parse(content: Any) -> tuple[str, str]:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start : end + 1]) if start >= 0 and end > start else {}
        if not isinstance(parsed, dict) or not str(parsed.get("code") or "").strip():
            raise ValueError("代码生成模型必须返回包含非空 code 的 JSON")
        return str(parsed.get("language") or "python"), str(parsed["code"])

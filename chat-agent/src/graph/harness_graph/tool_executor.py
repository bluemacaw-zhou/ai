"""工具执行器（05 图内部使用）。

从 ``graph.legacy_main_graph.data_graph.tool_executor.ToolExecutor`` 原样
复制而来：新体系不再依赖已归档、放弃维护的旧图代码。把“确认副作用工具 →
并发执行 → 结果标准化为干净 JSON”这一组内聚职责收敛到一个类里。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from mcp_tools.tool_result import parse_tool_result


class ToolExecutor:
    """按工具名索引工具，负责确认、执行与结果压缩。"""

    #: 回喂给模型的结果中保留的字段（去掉 wind 传输层噪音）。
    _RESULT_KEYS: tuple[str, ...] = ("ok", "data", "error_code", "error_message", "cancelled")

    def __init__(self, tools: list[BaseTool]):
        self._tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> dict[str, BaseTool]:
        """按名索引的工具表（只读视图供外部检视）。"""
        return self._tools

    async def execute(self, calls: list[dict[str, Any]]) -> list[ToolMessage]:
        """执行一批工具调用，返回与调用顺序对齐的 ToolMessage 列表。

        阶段一：先为所有副作用工具收集用户确认。interrupt 阶段本身没有副作用，
        因此 resume 后该流程从头重放也是安全的。
        阶段二：所有 interrupt 解决后，把可执行的调用并发跑（asyncio.gather），
        结果按 calls 原始顺序对齐，保证消息顺序稳定。
        """
        decisions: dict[str, bool] = {}
        for call in calls:
            tool = self._tools.get(call["name"])
            if tool is not None and self._requires_confirmation(tool):
                decision = interrupt(
                    {
                        "kind": "action_confirmation_required",
                        "tool_name": call["name"],
                        "arguments": call["args"],
                        "message": "该工具会产生外部副作用，是否确认执行？",
                    }
                )
                decisions[call["id"]] = self._is_confirmed(decision)

        return list(
            await asyncio.gather(*(self._run_call(call, decisions) for call in calls))
        )

    async def _run_call(
        self,
        call: dict[str, Any],
        decisions: dict[str, bool],
    ) -> ToolMessage:
        tool = self._tools.get(call["name"])
        if tool is None:
            return self._tool_message(
                call, {"ok": False, "error_message": f"未知工具 {call['name']}"}
            )
        if call["id"] in decisions and not decisions[call["id"]]:
            return self._tool_message(
                call,
                {"ok": False, "cancelled": True, "error_message": "用户取消了该操作。"},
            )
        result = await self._invoke_tool(tool, call["args"])
        return self._tool_message(call, result)

    @staticmethod
    def _requires_confirmation(tool: BaseTool) -> bool:
        return bool((getattr(tool, "extras", None) or {}).get("requires_confirmation", False))

    @staticmethod
    def _is_confirmed(decision: Any) -> bool:
        if isinstance(decision, dict):
            return bool(decision.get("confirmed"))
        return bool(decision)

    @staticmethod
    async def _invoke_tool(tool: BaseTool, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = await tool.ainvoke(arguments)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error_message": f"工具 {tool.name} 调用异常：{exc}"}
        return parse_tool_result(raw)

    @classmethod
    def _tool_message(cls, call: dict[str, Any], result: dict[str, Any]) -> ToolMessage:
        """把工具结果压缩成模型易读的干净 JSON，去掉 wind 传输层噪音。"""
        compact = {key: result.get(key) for key in cls._RESULT_KEYS if key in result}
        return ToolMessage(
            content=json.dumps(compact, ensure_ascii=False),
            tool_call_id=call["id"],
            name=call["name"],
        )

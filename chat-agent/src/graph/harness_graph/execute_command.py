"""ExecuteCurrentCommand：05 图核心循环的执行节点。

对应 design/harness/05-harness-loop-overview.puml 的 ``ExecuteCurrentCommand``
（当前最小闭环简化版，见 05-minimal-loop-implementation-plan.md）：

    ExecuteCurrentCommand : 输入：完整 HarnessState，执行 CommandList[-1]（status=pending）
    ExecuteCurrentCommand : 按 requirement、plan 和历史 output 自主决定是否调用工具
    ExecuteCurrentCommand : 可用工具：查询、计算、渲染等专业 MCP Tool

本次实现固定跑单轮：模型决定是否调用工具（查询工具或渲染工具
``create_markdown_surface``），若调用则一次性执行全部工具调用（不反复
调用模型）。执行结果（是否有工具调用、工具名、解析后的结果、渲染工具是否
产出了新 surface）交给下游 ``RegisterCommandResult`` 登记到
``CommandList[-1]``。

渲染工具通过 ``MarkdownSurfaceRecorder``（闭包持有的可写容器）回传结果，
不直接写 ``HarnessState``；本节点在工具执行结束后读取 recorder，把结果
放进返回给 LangGraph 的 dict 里，由 LangGraph 的 reducer 合并进
``HarnessState``（当前实现里 ``RegisterCommandResult`` 负责真正回写
``latest_a2ui_surface``，本节点只负责把 recorder 的快照传递下去）。

本节点每执行一次就把 ``HarnessState.loop_iterations`` 加一，供
``HarnessGraph`` 组装时的路由条件判断是否达到 ``max_loop_iterations``
硬上限（``CheckFailureThresholds`` 真正实现前的临时兜底，见计划文档
"细节 7"）。
"""

from __future__ import annotations

from typing import Any, Optional
import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.create_markdown_surface import (
    TOOL_NAME as MARKDOWN_SURFACE_TOOL_NAME,
    MarkdownSurfaceRecorder,
    create_markdown_surface_tool,
)
from graph.harness_graph.harness_state import HarnessState
from graph.harness_graph.message_utils import MessageUtils
from graph.harness_graph.tool_executor import ToolExecutor
from utils.prompt_loader import load_prompt

#: ExecuteCurrentCommand 本轮执行结果，供 RegisterCommandResult 消费。
ExecutionOutcome = dict[str, Any]


class ExecuteCommand:
    """执行 CommandList[-1] 的最小执行节点：固定单轮 模型->工具（可选）。"""

    def __init__(self, model: Any, tools: list[BaseTool]):
        """构造执行节点。

        Args:
            model: 未绑定工具的基础聊天模型。
            tools: 查询类工具（financial_query_data / fin_doc_searchV3）；
                渲染工具 ``create_markdown_surface`` 由本节点自行构造并
                追加，每次 ``execute`` 调用都使用独立的
                ``MarkdownSurfaceRecorder``，避免不同请求间串写。
        """
        self._query_tools = list(tools)
        self._model = model

    async def execute(
        self, state: HarnessState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """执行 CommandList[-1]（status=pending），产出本轮 ExecutionOutcome。

        ``config`` 由 LangGraph 自动注入，转发给 ``model_with_tools.ainvoke``
        才能让本次模型调用挂上 ``HarnessGraph.run`` 配置好的 Langfuse
        callback。
        """
        command_list = list(state.get("command_list") or [])
        if not command_list:
            raise ValueError("ExecuteCurrentCommand requires a non-empty command_list")
        current_command = command_list[-1]

        recorder = MarkdownSurfaceRecorder()
        tools = [*self._query_tools, create_markdown_surface_tool(recorder)]
        tool_executor = ToolExecutor(tools)
        model_with_tools = self._model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(
                content=load_prompt(
                    "execute_command",
                    ORIGINAL_QUESTION=json.dumps(
                        str(state.get("original_question") or ""), ensure_ascii=False
                    ),
                    CURRENT_REQUIREMENT=json.dumps(
                        str(current_command.get("requirement") or ""), ensure_ascii=False
                    ),
                    COMMAND_HISTORY=json.dumps(
                        self._summarize_history(command_list[:-1]), ensure_ascii=False
                    ),
                )
            ),
            HumanMessage(content="请按上述契约完成当前命令；需要时调用可用工具。"),
        ]

        response = await model_with_tools.ainvoke(messages, config=config)
        messages.append(response)

        calls = MessageUtils.last_tool_calls({"messages": messages})
        outcome: ExecutionOutcome = {
            "tool_called": False,
            "tool_name": None,
            "tool_result": None,
            "rendered_surface": None,
            "final_answer": MessageUtils.content_text(response.content),
        }

        if calls:
            tool_messages = await tool_executor.execute(calls)
            messages.extend(tool_messages)
            call = calls[0]
            tool_message = tool_messages[0]
            outcome["tool_called"] = True
            outcome["tool_name"] = call["name"]
            outcome["tool_result"] = MessageUtils.extract_json(
                MessageUtils.content_text(tool_message.content)
            )
            if call["name"] == MARKDOWN_SURFACE_TOOL_NAME:
                outcome["rendered_surface"] = recorder.latest_surface

        loop_iterations = int(state.get("loop_iterations") or 0) + 1
        return {"execution_outcome": outcome, "loop_iterations": loop_iterations}

    @staticmethod
    def _summarize_history(history: list[CommandRecord]) -> list[dict[str, Any]]:
        """把历史命令压缩成 requirement/status/output 摘要，供 prompt 使用。"""
        summaries: list[dict[str, Any]] = []
        for record in history:
            summaries.append(
                {
                    "sequence_no": record.get("sequence_no"),
                    "kind": record.get("kind"),
                    "requirement": record.get("requirement"),
                    "status": record.get("status"),
                    "output": record.get("output"),
                    "diagnostics": record.get("diagnostics"),
                }
            )
        return summaries

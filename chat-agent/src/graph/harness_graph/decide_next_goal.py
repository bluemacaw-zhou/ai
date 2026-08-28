"""DecideNextGoal：LLM 节点，决定是否还有下一步推进根目标。

对应 design/harness/05-harness-loop-overview.puml 的 ``DecideNextGoal``：

    DecideNextGoal : 输入：HarnessState + CheckGoalCompletion 已判定 is_completed=false
    DecideNextGoal : 输出：next_command 或 no_next_command
    DecideNextGoal : no_next_command -> 收尾为 failed，
        diagnostics.reason_code=goal_unreachable，返回 route=fallback_required

允许的 next_command.kind：query、calculate（不允许 render，呈现由
``AssessPresentation``/``AppendRenderCommand`` 负责）。
"""

from __future__ import annotations

from typing import Any, Optional
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graph.harness_graph.harness_state import HarnessState
from graph.harness_graph.message_utils import MessageUtils
from utils.prompt_loader import load_prompt

_ALLOWED_NEXT_KINDS = ("query", "calculate")


class DecideNextGoal:
    """LLM 节点：判断是否存在能推进根目标的下一条命令。"""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def decide(
        self, state: HarnessState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """``config`` 由 LangGraph 自动注入，转发给 ``model.ainvoke`` 才能让
        本次模型调用挂上 ``HarnessGraph.run`` 配置好的 Langfuse callback。
        """
        command_list = list(state.get("command_list") or [])

        messages = [
            SystemMessage(
                content=load_prompt(
                    "decide_next_goal",
                    ORIGINAL_QUESTION=json.dumps(
                        str(state.get("original_question") or ""), ensure_ascii=False
                    ),
                    COMMAND_HISTORY=json.dumps(
                        [dict(record) for record in command_list], ensure_ascii=False
                    ),
                )
            ),
            HumanMessage(content="请按输出契约返回下一目标决策。"),
        ]

        response = await self._model.ainvoke(messages, config=config)
        parsed = MessageUtils.extract_json(MessageUtils.message_text(response))

        next_command = parsed.get("next_command")
        if isinstance(next_command, dict) and next_command.get("kind") in _ALLOWED_NEXT_KINDS:
            return {
                "next_command_decision": {
                    "kind": next_command.get("kind"),
                    "requirement": str(next_command.get("requirement") or ""),
                    "plan": str(next_command.get("plan") or ""),
                }
            }

        return {
            "next_command_decision": None,
            "no_next_command_reason": str(parsed.get("reason") or ""),
        }

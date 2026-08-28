"""CheckGoalCompletion：LLM 节点，判断根问题是否已被完整解答。

对应 design/harness/05-harness-loop-overview.puml 的 ``CheckGoalCompletion``：

    CheckGoalCompletion : 输入：HarnessState，尤其是完整 CommandList + original_question
    CheckGoalCompletion : 输出：is_completed = true | false，附带 reason
    CheckGoalCompletion : is_completed=true -> 直接向上返回 route=normal_completed
    CheckGoalCompletion : is_completed=false -> 进入 DecideNextGoal
"""

from __future__ import annotations

from typing import Any, Optional
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graph.harness_graph.harness_state import HarnessState
from graph.harness_graph.message_utils import MessageUtils
from utils.prompt_loader import load_prompt


class CheckGoalCompletion:
    """LLM 节点：判断根问题 original_question 是否已经被完整、真实地解答。"""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def check(
        self, state: HarnessState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """``config`` 由 LangGraph 自动注入，转发给 ``model.ainvoke`` 才能让
        本次模型调用挂上 ``HarnessGraph.run`` 配置好的 Langfuse callback。
        """
        command_list = list(state.get("command_list") or [])

        messages = [
            SystemMessage(
                content=load_prompt(
                    "check_goal_completion",
                    ORIGINAL_QUESTION=json.dumps(
                        str(state.get("original_question") or ""), ensure_ascii=False
                    ),
                    COMMAND_HISTORY=json.dumps(
                        [dict(record) for record in command_list], ensure_ascii=False
                    ),
                )
            ),
            HumanMessage(content="请按输出契约返回完成度判断。"),
        ]

        response = await self._model.ainvoke(messages, config=config)
        parsed = MessageUtils.extract_json(MessageUtils.message_text(response))
        is_completed = bool(parsed.get("is_completed", False))

        return {
            "is_goal_completed": is_completed,
            "goal_completion_reason": str(parsed.get("reason") or ""),
        }

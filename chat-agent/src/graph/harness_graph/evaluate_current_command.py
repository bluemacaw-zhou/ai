"""EvaluateCurrentCommand：LLM 节点，评审 CommandList[-1] 是否业务达成。

对应 design/harness/05-harness-loop-overview.puml：

    EvaluateCurrentCommand : 输入：HarnessState + 刚登记的 CommandList[-1]
    EvaluateCurrentCommand : 输出：evaluation = succeeded | failed，附带 reason
"""

from __future__ import annotations

from typing import Any, Optional
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.harness_state import HarnessState
from graph.harness_graph.message_utils import MessageUtils
from utils.prompt_loader import load_prompt

EVALUATION_SUCCEEDED = "succeeded"
EVALUATION_FAILED = "failed"


class EvaluateCurrentCommand:
    """LLM 节点：判断刚登记的最后一条命令是否达成业务目标。"""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def evaluate(
        self, state: HarnessState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """``config`` 由 LangGraph 自动注入，转发给 ``model.ainvoke`` 才能让
        本次模型调用挂上 ``HarnessGraph.run`` 配置好的 Langfuse callback。
        """
        command_list = list(state.get("command_list") or [])
        if not command_list:
            raise ValueError("EvaluateCurrentCommand requires a non-empty command_list")
        latest_command: CommandRecord = command_list[-1]

        messages = [
            SystemMessage(
                content=load_prompt(
                    "evaluate_current_command",
                    ORIGINAL_QUESTION=json.dumps(
                        str(state.get("original_question") or ""), ensure_ascii=False
                    ),
                    LATEST_COMMAND=json.dumps(dict(latest_command), ensure_ascii=False),
                    COMMAND_HISTORY=json.dumps(
                        [dict(record) for record in command_list[:-1]], ensure_ascii=False
                    ),
                )
            ),
            HumanMessage(content="请按输出契约返回评审结果。"),
        ]

        response = await self._model.ainvoke(messages, config=config)
        parsed = MessageUtils.extract_json(MessageUtils.message_text(response))
        evaluation = parsed.get("evaluation")
        if evaluation not in (EVALUATION_SUCCEEDED, EVALUATION_FAILED):
            evaluation = EVALUATION_FAILED

        return {
            "current_evaluation": evaluation,
            "current_evaluation_reason": str(parsed.get("reason") or ""),
        }

"""AnswerFallback：08 图的降级回答 LLM 节点。

对应 design/harness/08-input-preprocessing-and-routing.puml：

    AnswerFallback : 输入：全局 HarnessState + fallback_reason
    AnswerFallback : 输出：fallback_answer

当 StartHarness 调用 05 图返回 fallback_required 时进入本节点：根据用户的
original_question 和 StartHarness 附带的 fallback_reason，给出能力边界内的
回答，或清楚说明无法继续的原因。不得虚构未取得的金融数据或结论。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from config import get_logger
from graph.harness_graph.message_utils import MessageUtils
from graph.main_graph.main_graph_state import MainGraphState
from utils.prompt_loader import load_prompt

log = get_logger(__name__)


class AnswerFallback:
    """基于 LLM 的降级回答：能力边界内作答或说明无法继续的原因。"""

    def __init__(self, model: Any):
        """使用一个未绑定工具的基础模型进行降级回答。"""
        self._model = model

    async def answer(
        self, state: MainGraphState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """读取 original_question + fallback_reason，输出 fallback_answer。

        ``config`` 由 LangGraph 自动注入，必须转发给 ``model.ainvoke`` 才能
        让本次模型调用挂上 ``MainGraph.run`` 配置好的 Langfuse callback。
        """
        original_question = str(state.get("original_question") or "")
        fallback_reason = str(state.get("fallback_reason") or "")

        response = await self._model.ainvoke(
            [
                SystemMessage(
                    content=load_prompt(
                        "answer_fallback",
                        CURRENT_QUESTION=json.dumps(original_question, ensure_ascii=False),
                        RELEVANT_HISTORY="[]",
                        EXECUTION_CONTEXT=json.dumps(
                            {"fallback_reason": fallback_reason},
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                ),
                HumanMessage(content="请在能力边界内给出最终答复。"),
            ],
            config=config,
        )
        fallback_answer = MessageUtils.message_text(response).strip()
        log.info(
            "main_graph.answer_fallback.fallback_answer",
            original_question=original_question,
            fallback_reason=fallback_reason,
        )
        return {"fallback_answer": fallback_answer}

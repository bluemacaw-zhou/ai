"""AnswerDirectly：08 图的直接回答 LLM 节点。

对应 design/harness/08-input-preprocessing-and-routing.puml：

    AnswerDirectly : 输入：全局 HarnessState
    AnswerDirectly : 输出：direct_answer

当 PreprocessQuestion 判定 route=direct_answer 时进入本节点：直接回答用户
的非金融信息获取问题，不调用任何工具。
"""

from __future__ import annotations

from typing import Any, Optional
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from config import get_logger
from graph.harness_graph.message_utils import MessageUtils
from graph.main_graph.main_graph_state import MainGraphState
from utils.prompt_loader import load_prompt

log = get_logger(__name__)


class AnswerDirectly:
    """基于 LLM 的直接回答：非金融信息获取问题的通用作答。"""

    def __init__(self, model: Any):
        """使用一个未绑定工具的基础模型直接回答。"""
        self._model = model

    async def answer(
        self, state: MainGraphState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """读取 original_question，输出 direct_answer。

        ``config`` 由 LangGraph 自动注入，必须转发给 ``model.ainvoke`` 才能
        让本次模型调用挂上 ``MainGraph.run`` 配置好的 Langfuse callback。
        """
        original_question = str(state.get("original_question") or "")
        response = await self._model.ainvoke(
            [
                SystemMessage(
                    content=load_prompt(
                        "answer_directly",
                        CURRENT_QUESTION=json.dumps(original_question, ensure_ascii=False),
                        RELEVANT_HISTORY="[]",
                    )
                ),
                HumanMessage(content="请直接给出最终答复。"),
            ],
            config=config,
        )
        direct_answer = MessageUtils.message_text(response).strip()
        log.info(
            "main_graph.answer_directly.direct_answer",
            original_question=original_question,
        )
        return {"direct_answer": direct_answer}

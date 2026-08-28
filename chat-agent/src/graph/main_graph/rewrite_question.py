"""RewriteQuestion：08 图的问题改写 LLM 节点。

对应 design/harness/08-input-preprocessing-and-routing.puml：

    RewriteQuestion : 输入：raw_question + A2A task_history
    RewriteQuestion : 输出：original_question（消解代词、省略后的独立问句）
    RewriteQuestion : original_question 是全链路唯一使用的问题变量名，此后不再改名

用一个不绑定工具的模型，结合当前用户输入与同一 A2A task 的历史对话，消解代词、
省略与追问，生成语义等价、可脱离历史独立理解的 original_question。若没有历史或
历史与当前问题无关，模型应原样返回当前输入（prompts/rewrite_question.md 中的
约束），本节点自身不做该判断，只负责组装 payload、调用模型与解析结果。

历史对话的数据契约是 ``HumanMessage``/``AIMessage`` 交替列表（
``task_history_messages``，由 ``ReceiveUserQuestion`` 从 ``related_tasks``
转换而来）：``HumanMessage.content`` 是历史某一轮的用户原话，
``AIMessage.content`` 是从对应 AI 回答的 A2UI render surface 中提取出的文本
摘要（见 ``SurfaceTextExtractor``），不是 AI 的逐字原话——因为本项目的
``Task.history`` 只保存用户消息，AI 回答实际落在 ``Task.artifacts`` 里。
prompts/rewrite_question.md 中已提示模型这一点。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from config import get_logger
from graph.harness_graph.message_utils import MessageUtils
from graph.main_graph.main_graph_state import MainGraphState
from utils.prompt_loader import load_prompt

log = get_logger(__name__)


class RewriteQuestion:
    """基于 LLM 的问题改写：消解代词/省略，产出可脱离历史独立理解的问句。"""

    def __init__(self, model: Any):
        """使用一个未绑定工具的基础模型进行改写。"""
        self._model = model

    async def rewrite(
        self, state: MainGraphState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """读取 raw_question + task_history_messages，输出 original_question。

        ``config`` 由 LangGraph 自动注入（携带 ``MainGraph.run`` 挂好的
        Langfuse callback），必须原样转发给 ``model.ainvoke``，否则本节点
        的模型调用不会出现在 Langfuse trace 里。
        """
        raw_question = str(state.get("raw_question") or "")
        history_messages = list(state.get("task_history_messages") or [])

        payload = {
            "raw_question": raw_question,
            "task_history": self._format_history(history_messages),
        }
        response = await self._model.ainvoke(
            [
                SystemMessage(
                    content=load_prompt(
                        "rewrite_question",
                        CURRENT_QUESTION=json.dumps(raw_question, ensure_ascii=False),
                        RELEVANT_HISTORY=json.dumps(
                            payload["task_history"], ensure_ascii=False, indent=2
                        ),
                    )
                ),
                HumanMessage(content="请仅按上述输出契约处理该输入。"),
            ],
            config=config,
        )
        parsed = MessageUtils.extract_json(MessageUtils.message_text(response))
        original_question = str(parsed.get("original_question") or "").strip()
        if not original_question:
            # 模型未按约定返回时，保守兜底为原始输入，不阻断链路。
            original_question = raw_question
        log.info(
            "main_graph.rewrite_question.original_question",
            raw_question=raw_question,
            original_question=original_question,
        )
        return {"original_question": original_question}

    @staticmethod
    def _format_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
        """把历史消息压平成 {role, content} 列表，供模型阅读。"""
        formatted: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            content = MessageUtils.content_text(message.content)
            if content:
                formatted.append({"role": role, "content": content})
        return formatted

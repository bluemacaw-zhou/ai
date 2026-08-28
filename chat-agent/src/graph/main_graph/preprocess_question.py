"""PreprocessQuestion：08 图的问题预处理 LLM 节点。

对应 design/harness/08-input-preprocessing-and-routing.puml：

    PreprocessQuestion : 输入：original_question
    PreprocessQuestion : 输出：route（harness | direct_answer），不改写、不重命名 original_question

判断用户问题是否属于金融信息获取（需要查询金融主体、板块、行情、指标、时序
数据或金融文档证据）：是则 route=harness，进入 StartHarness；否则
route=direct_answer，进入 AnswerDirectly。不改写 original_question，不调用
任何工具，不创建 HarnessState。
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

ROUTE_HARNESS = "harness"
ROUTE_DIRECT_ANSWER = "direct_answer"


class PreprocessQuestion:
    """基于 LLM 的问题分类：判断是否属于金融信息获取，产出 route。"""

    def __init__(self, model: Any):
        """使用一个未绑定工具的基础模型进行分类。"""
        self._model = model

    async def preprocess(
        self, state: MainGraphState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """读取 original_question，输出 route。

        ``config`` 由 LangGraph 自动注入（携带 ``MainGraph.run`` 挂好的
        Langfuse callback），必须原样转发给 ``model.ainvoke``，否则本节点
        的模型调用不会出现在 Langfuse trace 里。
        """
        original_question = str(state.get("original_question") or "")
        response = await self._model.ainvoke(
            [
                SystemMessage(
                    content=load_prompt(
                        "preprocess_question",
                        CURRENT_QUESTION=json.dumps(original_question, ensure_ascii=False),
                        RELEVANT_HISTORY="[]",
                    )
                ),
                HumanMessage(content="请仅按上述输出契约返回分类结果。"),
            ],
            config=config,
        )
        parsed = MessageUtils.extract_json(MessageUtils.message_text(response))
        route = str(parsed.get("route") or "").strip()
        if route not in (ROUTE_HARNESS, ROUTE_DIRECT_ANSWER):
            # 模型未按约定返回时，保守兜底为 harness，交给核心链路处理，
            # 不阻断请求。
            route = ROUTE_HARNESS
        log.info(
            "main_graph.preprocess_question.route",
            original_question=original_question,
            route=route,
        )
        return {"route": route}

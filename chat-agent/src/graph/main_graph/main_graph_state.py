"""MainGraphState：08 图（design/harness/08-input-preprocessing-and-routing.puml）
的图状态，贯穿全部 7 个节点：

    ReceiveUserQuestion -> RewriteQuestion -> PreprocessQuestion
        -> StartHarness -> [*] | AnswerFallback -> CreateMarkdownSurfaceAndSend -> [*]
        -> AnswerDirectly -> CreateMarkdownSurfaceAndSend -> [*]

- ``raw_question``：用户原始文本，未经任何加工（ReceiveUserQuestion 的输入）。
- ``related_tasks``：客户端在 ``Message.reference_task_ids`` 里携带的最近若干轮
  历史 task_id，经 ``SimpleRequestContextBuilder``（已开启
  ``should_populate_referred_tasks=True``）从 task_store 批量拉取出的完整历史
  ``a2a.types.Task`` 对象列表；由 ``A2aAgentExecutor`` 从
  ``context.related_tasks`` 原样传入。ReceiveUserQuestion 是唯一的消费方。
- ``task_history_messages``：ReceiveUserQuestion 把 ``related_tasks`` 转换出的
  历史对话（用户问题 + AI 回答摘要交替的 HumanMessage/AIMessage 列表），供
  RewriteQuestion 消解代词/省略时参考。
- ``original_question``：RewriteQuestion 消解代词、省略后产出的独立问句；
  全链路只用这一个变量名，此后不再改名（对应 review-issues.md 问题7的结论）。
- ``route``：PreprocessQuestion 产出的路由结果（``harness`` |
  ``direct_answer``），决定进入 StartHarness 还是 AnswerDirectly。
- ``harness_route``：StartHarness 调用 05 图（HarnessGraph）后得到的结果
  （``normal_completed`` | ``fallback_required``），决定直接结束还是走
  AnswerFallback。
- ``context_id`` / ``task_id`` / ``session_id``：A2A 任务身份信息，贯穿整张图。
- ``updater``：``a2a.server.tasks.TaskUpdater`` 实例，由
  ``A2aAgentExecutor`` 注入；StartHarness 把它转交给 05 图内部的
  ``CompleteTask`` 使用（05图自己完成 completed 推送）；
  CreateMarkdownSurfaceAndSend 使用它推送直接回答/降级回答的 markdown surface。
  这是一个有状态的运行时对象，不参与任何持久化或跨进程序列化——MainGraph
  编译时不使用 checkpointer。
- ``metadata``：A2A 请求的原始 metadata（例如 selectedContext）。
- ``fallback_reason``：StartHarness 判定 ``fallback_required`` 时附带的说明，
  供 AnswerFallback 使用。
- ``direct_answer`` / ``fallback_answer``：AnswerDirectly / AnswerFallback
  产出的回答文本，供 CreateMarkdownSurfaceAndSend 统一构造 markdown surface。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class MainGraphState(TypedDict, total=False):
    """08 图全部 7 个节点共享的状态。"""

    raw_question: str
    related_tasks: list[Any]
    task_history_messages: list[BaseMessage]
    original_question: str
    route: str
    harness_route: str
    context_id: str
    task_id: str
    session_id: str
    updater: Any
    metadata: dict[str, Any] | None
    fallback_reason: str
    direct_answer: str
    fallback_answer: str

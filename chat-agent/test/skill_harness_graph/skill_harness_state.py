"""分层加载示例图的状态定义。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SkillHarnessState(TypedDict, total=False):
    """仿照 Wind.Weaver.ChatAgent 的 Skill 元数据注入、模型选择与执行状态。"""

    # 用户提交的真实问题，是模型选择 Skill 的业务依据。
    question: str
    # 图中的 LangChain 消息轨迹，模型和工具的输出均通过 add_messages 追加。
    messages: Annotated[list[BaseMessage], add_messages]
    # 系统提示词参数；SKILLS_META 只包含 frontmatter 元数据，不包含 Skill 正文。
    system_prompt_params: dict[str, str]
    # 已完成的模型决策轮数，用于防止 agent -> tools 循环无限执行。
    loop_iterations: int
    # 允许的最大模型决策轮数。
    max_loop_iterations: int
    # 模型停止调用工具后输出的最终用户答案。
    final_answer: str

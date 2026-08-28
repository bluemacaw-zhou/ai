"""消息与 JSON 解析工具集（05 图内部使用）。

从 ``graph.legacy_main_graph.data_graph.message_utils.MessageUtils`` 复制而来：
新体系（HarnessGraph/MainGraph）不再依赖已归档、放弃维护的旧图代码，因此把
这个纯工具类单独复制一份，去掉了对已归档 ``MainGraphState`` 类型的依赖（
``last_tool_calls`` 原本只是用它做类型标注，不影响实际逻辑，改为通用
``dict[str, Any]``）。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class MessageUtils:
    """对消息历史的只读检视与 JSON 解析辅助方法。"""

    REVISION_FEEDBACK_PREFIX: str = "【目标评审未通过"

    @staticmethod
    def first_user_question(messages: list[BaseMessage]) -> str:
        """返回首个真实用户问题。"""
        for message in messages:
            if isinstance(message, HumanMessage) and not str(message.content).startswith(
                MessageUtils.REVISION_FEEDBACK_PREFIX
            ):
                return MessageUtils.content_text(message.content)
        return ""

    @staticmethod
    def last_final_answer(messages: list[BaseMessage]) -> str:
        """返回最近一次无 tool_calls 的 AI 消息文本。"""
        message = MessageUtils.last_final_answer_message(messages)
        if message is None:
            return ""
        return MessageUtils.content_text(message.content)

    @staticmethod
    def last_final_answer_message(messages: list[BaseMessage]) -> BaseMessage | None:
        """返回最近一次无 tool_calls 的 AI 消息对象。"""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message
        return None

    @staticmethod
    def collect_tool_facts(messages: list[BaseMessage]) -> list[str]:
        """汇总所有工具返回，作为目标评审的工具证据。"""
        facts: list[str] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                facts.append(f"{message.name}: {MessageUtils.content_text(message.content)}")
        return facts

    @staticmethod
    def last_tool_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
        """返回状态中最后一条 AI 消息发出的 tool_calls。"""
        messages = state["messages"]
        if not messages:
            return []
        last = messages[-1]
        if isinstance(last, AIMessage):
            return list(getattr(last, "tool_calls", []) or [])
        return []

    @staticmethod
    def message_text(response: Any) -> str:
        """从模型响应对象中提取纯文本内容。"""
        return MessageUtils.content_text(getattr(response, "content", response))

    @staticmethod
    def content_text(content: Any) -> str:
        """把 str、富文本 list 或其它内容统一压平成纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
        return str(content)

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """从模型文本中鲁棒地解析出 JSON 对象。"""
        text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return {}
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return parsed if isinstance(parsed, dict) else {}

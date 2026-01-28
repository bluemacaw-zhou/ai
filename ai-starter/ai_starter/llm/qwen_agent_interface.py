"""
Qwen-Agent LLM 接口定义

使用 ABC 显式声明 Qwen-Agent 需要的 LLM 接口，
让类的接口实现关系一目了然。
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

# Qwen-Agent imports (optional)
try:
    from qwen_agent.llm.schema import Message
except ImportError:
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content


class QwenAgentLLM(ABC):
    """
    Qwen-Agent LLM 接口

    任何实现此接口的类都可以被 Qwen-Agent 的 Agent 使用。
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Optional[Dict] = None,
    ):
        """
        Qwen-Agent 的 chat 接口

        Args:
            messages: 输入消息列表
            functions: 函数列表（可选）
            stream: 是否使用流式生成
            delta_stream: 是否增量流式输出
            extra_generate_cfg: 额外的生成参数

        Returns:
            生成的消息列表或迭代器
        """
        pass

    @property
    @abstractmethod
    def support_multimodal_input(self) -> bool:
        """是否支持多模态输入"""
        pass

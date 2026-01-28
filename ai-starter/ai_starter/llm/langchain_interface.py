"""
LangChain LLM 接口定义

显式声明 LangChain BaseChatModel 需要实现的接口方法，
让类的接口实现关系一目了然。
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

# LangChain imports
try:
    from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    # 创建显式的接口定义，继承 LangChain 的 BaseChatModel
    class LangChainChatModel(LangChainBaseChatModel):
        """
        LangChain ChatModel 接口

        继承 LangChain 的 BaseChatModel 并显式声明核心接口方法。
        任何实现此接口的类都可以作为 LangChain 的 ChatModel 使用。
        """

        @abstractmethod
        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """
            生成回复（非流式）- LangChain 核心接口

            Args:
                messages: 输入消息列表
                stop: 停止词列表
                run_manager: 运行管理器
                **kwargs: 其他参数

            Returns:
                ChatResult: 包含生成结果的对象
            """
            pass

        @abstractmethod
        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
        ) -> Iterator:
            """
            流式生成 - LangChain 核心接口

            Args:
                messages: 输入消息列表
                stop: 停止词列表
                run_manager: 运行管理器
                **kwargs: 其他参数

            Returns:
                Iterator: 流式生成结果的迭代器
            """
            pass

except ImportError:
    # 如果没有安装 langchain，提供占位符
    BaseMessage = object
    ChatGeneration = object
    ChatResult = object

    class LangChainChatModel(ABC):
        @abstractmethod
        def _generate(self, messages, stop=None, run_manager=None, **kwargs): pass

        @abstractmethod
        def _stream(self, messages, stop=None, run_manager=None, **kwargs): pass

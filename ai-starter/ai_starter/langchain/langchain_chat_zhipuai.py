"""
LangChain ZhipuAI 适配器（组合模式）

将 ZhipuAI HTTP API 适配为 LangChain BaseChatModel 接口
"""

from typing import Any, Dict, Iterator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from ai_starter.llm.zhipuai_base import ZhipuAIBase
from ai_starter import get_logger

logger = get_logger(__name__)


class LangChainChatZhipuAI(BaseChatModel):
    """
    LangChain ZhipuAI 适配器（组合模式）

    实现 LangChain BaseChatModel 接口：
    - _generate(): 非流式生成
    - _stream(): 流式生成
    - _llm_type: LLM 类型标识

    使用组合模式：包含 ZhipuAIBase 实例而非继承它

    配置内聚设计：所有配置从 config.yaml 读取，不接受外部参数覆盖
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )

    def __init__(self, **kwargs):
        """
        初始化 LangChain ZhipuAI 适配器

        Args:
            model: 模型名称（已弃用，总是从 config.yaml 读取）
            temperature: 温度参数（已弃用，总是从 config.yaml 读取）
            **kwargs: 其他参数
        """
        # 创建 ZhipuAIBase 实例
        # 初始化 BaseChatModel（使用从 config.yaml 读取的值）
        zhipu = ZhipuAIBase()

        # 设置 BaseChatModel 需要的属性
        kwargs['model'] = zhipu.model
        kwargs['temperature'] = zhipu.temperature

        # 将 zhipu 对象通过 kwargs 传递（extra='allow' 允许这样做）
        kwargs['_zhipu'] = zhipu

        super().__init__(**kwargs)

        logger.info(f"LangChainChatZhipuAI initialized (model: {self._zhipu.model}, temperature: {self._zhipu.temperature})")

    @property
    def _llm_type(self) -> str:
        """LLM 类型标识（LangChain 接口）"""
        return "zhipuai"

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict:
        """将 LangChain Message 转换为 API 格式"""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"

        return {"role": role, "content": message.content}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成回复（LangChain 接口）

        实现 BaseChatModel._generate()
        """
        # 转换消息格式
        message_dicts = [self._convert_message_to_dict(m) for m in messages]

        # 委托给 ZhipuAIBase 调用 API
        result = self._zhipu._call_api(messages=message_dicts, **kwargs)

        # 解析响应
        content = result["choices"][0]["message"]["content"]
        generation = ChatGeneration(message=AIMessage(content=content))

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator:
        """
        流式生成（LangChain 接口）

        实现 BaseChatModel._stream()
        """
        raise NotImplementedError("Stream mode not implemented yet")

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, '_zhipu'):
                self._zhipu._http_client.close()
        except Exception:
            pass

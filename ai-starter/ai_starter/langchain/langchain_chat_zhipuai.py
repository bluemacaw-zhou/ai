"""
LangChain ZhipuAI 适配器（组合模式）

将 ZhipuAI HTTP API 适配为 LangChain BaseChatModel 接口
"""

from typing import Any, Dict, Iterator, List, Optional, Type, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict
import json

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
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式生成（LangChain 接口）

        实现 BaseChatModel._stream()
        """
        # 转换消息格式
        message_dicts = [self._convert_message_to_dict(m) for m in messages]

        # 委托给 ZhipuAIBase 调用流式 API
        for content in self._zhipu._call_api_stream(messages=message_dicts, **kwargs):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            yield chunk

            # 如果有 run_manager，通知新的 token
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk)

    def _convert_pydantic_to_function_tool(self, schema: Type[BaseModel]) -> Dict:
        """
        将 Pydantic 模型转换为 ZhipuAI 函数调用工具格式

        Args:
            schema: Pydantic 模型类

        Returns:
            函数调用工具定义字典
        """
        # 获取 Pydantic 模型的 JSON Schema
        json_schema = schema.model_json_schema()

        # 构造 ZhipuAI 函数调用格式
        function_def = {
            "type": "function",
            "function": {
                "name": schema.__name__,
                "description": json_schema.get("description", f"Extract {schema.__name__} from input"),
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", [])
                }
            }
        }

        return function_def

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs: Any
    ) -> Runnable:
        """
        返回一个支持结构化输出的 Runnable

        Args:
            schema: Pydantic 模型类
            **kwargs: 其他参数

        Returns:
            支持结构化输出的 Runnable
        """
        from langchain_core.runnables import RunnableLambda

        # 创建函数调用工具定义
        tool_def = self._convert_pydantic_to_function_tool(schema)

        def parse_output(response: Union[str, List[BaseMessage]]) -> BaseModel:
            """解析 LLM 响应并返回结构化对象"""
            # 如果输入是字符串，转换为消息
            if isinstance(response, str):
                messages = [HumanMessage(content=response)]
            else:
                messages = response

            # 转换消息格式
            message_dicts = [self._convert_message_to_dict(m) for m in messages]

            # 调用 API，传递 tools 参数
            result = self._zhipu._call_api(
                messages=message_dicts,
                tools=[tool_def],
                tool_choice="auto"
            )

            # 解析响应
            choice = result["choices"][0]
            message = choice.get("message", {})

            # 检查是否有函数调用
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # 如果没有函数调用，尝试从 content 中解析
                content = message.get("content", "")
                logger.warning(f"No tool calls found, trying to parse from content: {content}")
                try:
                    # 尝试直接解析为 JSON
                    data = json.loads(content)
                    return schema(**data)
                except Exception as e:
                    logger.error(f"Failed to parse content as JSON: {e}")
                    raise ValueError(f"LLM did not return structured output: {content}")

            # 提取第一个函数调用的参数
            function_call = tool_calls[0].get("function", {})
            arguments_str = function_call.get("arguments", "{}")

            try:
                # 解析参数并构造 Pydantic 对象
                arguments = json.loads(arguments_str)
                return schema(**arguments)
            except Exception as e:
                logger.error(f"Failed to parse function arguments: {e}, arguments: {arguments_str}")
                raise ValueError(f"Failed to create {schema.__name__} from arguments: {arguments_str}")

        # 返回包装的 Runnable
        return RunnableLambda(parse_output)

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, '_zhipu'):
                self._zhipu._http_client.close()
        except Exception:
            pass

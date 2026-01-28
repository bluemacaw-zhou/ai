"""
自定义 ZhipuAI LLM 实现

使用多重继承显式声明接口实现：
- 继承 LangChainChatModel（本模块定义的 LangChain 接口）
- 继承 QwenAgentLLM（本模块定义的 Qwen-Agent 接口）

所有配置从 Config 读取，内聚设计。
自动从配置文件读取代理和 SSL 设置，使用 HttpClientFactory 创建 http_client。
"""

import httpx
import time
import jwt
from typing import Any, Dict, Iterator, List, Optional, Union

# LangChain imports
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

# Qwen-Agent imports
try:
    from qwen_agent.llm.base import ModelServiceError, register_llm
    from qwen_agent.llm.schema import ASSISTANT, Message as QwenMessage
    QWEN_AGENT_AVAILABLE = True
except ImportError:
    ModelServiceError = Exception
    ASSISTANT = "assistant"

    def register_llm(model_type):
        def decorator(cls):
            return cls
        return decorator

    class QwenMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    QWEN_AGENT_AVAILABLE = False

# 本地导入
try:
    from ai_starter.core.log.logging_utils import get_logger
    from ai_starter.core.http_client.http_client_factory import HttpClientFactory
    from ai_starter.core.config.config import Config
    from .langchain_interface import LangChainChatModel
    from .qwen_agent_interface import QwenAgentLLM
except ImportError:
    from ai_starter import get_logger, HttpClientFactory, Config, LangChainChatModel, QwenAgentLLM

logger = get_logger(__name__)


def _get_jwt_token(api_key: str) -> str:
    """生成 JWT Token"""
    try:
        api_key, secret = api_key.split(".")
    except Exception as e:
        raise ValueError(f"Invalid API key format: {e}")

    payload = {
        "api_key": api_key,
        "exp": int(round(time.time() * 1000)) + 3600 * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


# 多重继承：同时实现 LangChain 和 Qwen-Agent 接口
@register_llm('zhipuai')
class CustomChatZhipuAI(LangChainChatModel, QwenAgentLLM):
    """
    自定义智谱 AI LLM

    实现接口:
    - LangChainChatModel (继承)
      - _generate(): 非流式生成
      - _stream(): 流式生成

    - QwenAgentLLM (继承)
      - chat(): Qwen-Agent chat 接口
      - support_multimodal_input: 是否支持多模态输入

    所有配置从 Config 读取，内聚设计。
    """

    # Pydantic 字段（LangChain 要求）
    model: str = Field(default="glm-4-flash")
    temperature: float = Field(default=0.7)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, cfg: Optional[Dict] = None, **kwargs):
        """
        初始化（不接受参数，所有配置从 Config 读取）

        Args:
            cfg: Qwen-Agent 兼容性参数，将被忽略
            **kwargs: 其他参数，将被忽略
        """
        # 从 Config 读取所有配置
        config = Config()

        # 设置模型参数
        kwargs['model'] = config.get("models.llm.model", "glm-4-flash")
        kwargs['temperature'] = config.get("models.llm.temperature", 0.7)

        # 调用父类初始化（会初始化 LangChainChatModel，进而初始化 LangChain 的 BaseChatModel）
        super().__init__(**kwargs)

        # 读取 API 配置
        self._api_key = config.get("api.zhipuai.key")
        self._api_base = config.get("api.zhipuai.api_base", "https://open.bigmodel.cn/api/paas/v4/chat/completions")

        if not self._api_key:
            raise ValueError("api_key is required. 请在配置文件中配置 api.zhipuai.key")

        # 创建自定义 http_client（自动从配置读取代理和 SSL 设置）
        self._http_client = HttpClientFactory.create()

        logger.info(f"CustomChatZhipuAI initialized (model: {self.model})")

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def http_client(self) -> httpx.Client:
        return self._http_client

    # ========== LangChain 接口实现 ==========

    @property
    def _llm_type(self) -> str:
        """LLM 类型标识"""
        return "custom-zhipuai"

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

        实现 LangChainChatModel._generate()
        """
        message_dicts = [self._convert_message_to_dict(m) for m in messages]

        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            **kwargs
        }

        headers = {
            "Authorization": _get_jwt_token(self._api_key),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        logger.info(f"调用 ZhipuAI API: {self._api_base}")

        try:
            response = self._http_client.post(
                self._api_base,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"API 调用成功")

            content = result["choices"][0]["message"]["content"]
            generation = ChatGeneration(message=AIMessage(content=content))

            return ChatResult(generations=[generation])

        except httpx.HTTPStatusError as e:
            logger.error(f"API 调用失败: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"API 调用异常: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator:
        """
        流式生成（LangChain 接口）

        实现 LangChainChatModel._stream()
        """
        raise NotImplementedError("Stream mode not implemented yet")

    # ========== Qwen-Agent 接口实现 ==========

    @property
    def support_multimodal_input(self) -> bool:
        """
        是否支持多模态输入（QwenAgentLLM 接口）

        实现 QwenAgentLLM.support_multimodal_input
        """
        return False

    def chat(
        self,
        messages: List[Union[QwenMessage, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[QwenMessage], Iterator[List[QwenMessage]]]:
        """
        Qwen-Agent 的 chat 接口（QwenAgentLLM 接口）

        实现 QwenAgentLLM.chat()
        """
        # 统一消息格式为 List[QwenMessage]
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(QwenMessage(**msg))
            else:
                new_messages.append(msg)

        generate_cfg = extra_generate_cfg or {}

        if functions:
            raise NotImplementedError("Function calling is not supported yet")

        if stream:
            return self._chat_stream(new_messages, delta_stream, generate_cfg)
        else:
            return self._chat_no_stream(new_messages, generate_cfg)

    def _chat_stream(
        self,
        messages: List[QwenMessage],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[QwenMessage]]:
        """流式聊天（Qwen-Agent 内部方法）"""
        api_messages = []
        for msg in messages:
            api_messages.append({
                'role': msg.role,
                'content': msg.content
            })

        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": generate_cfg.get('temperature', self.temperature),
        }

        headers = {
            "Authorization": _get_jwt_token(self._api_key),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        try:
            response = self._http_client.post(
                self._api_base,
                json=payload,
                headers=headers,
                timeout=generate_cfg.get('timeout', 60)
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]

            if delta_stream:
                yield [QwenMessage(role=ASSISTANT, content=content)]
            else:
                yield [QwenMessage(role=ASSISTANT, content=content)]

        except Exception as ex:
            if QWEN_AGENT_AVAILABLE:
                raise ModelServiceError(exception=ex)
            else:
                raise

    def _chat_no_stream(
        self,
        messages: List[QwenMessage],
        generate_cfg: dict,
    ) -> List[QwenMessage]:
        """非流式聊天（Qwen-Agent 内部方法）"""
        api_messages = []
        for msg in messages:
            api_messages.append({
                'role': msg.role,
                'content': msg.content
            })

        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": generate_cfg.get('temperature', self.temperature),
        }

        headers = {
            "Authorization": _get_jwt_token(self._api_key),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        try:
            response = self._http_client.post(
                self._api_base,
                json=payload,
                headers=headers,
                timeout=generate_cfg.get('timeout', 60)
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            return [QwenMessage(role=ASSISTANT, content=content)]

        except Exception as ex:
            if QWEN_AGENT_AVAILABLE:
                raise ModelServiceError(exception=ex)
            else:
                raise

    def __del__(self):
        """清理资源"""
        try:
            self._http_client.close()
        except Exception:
            pass

"""
Qwen-Agent ZhipuAI 适配器

将 ZhipuAI HTTP API 适配为 Qwen-Agent LLM 接口
使用组合模式替代多重继承
"""

from typing import Dict, Iterator, List, Optional, Union

from qwen_agent.llm.base import BaseChatModel, ModelServiceError, register_llm
from qwen_agent.llm.schema import ASSISTANT, Message as QwenMessage

from ai_starter.llm.zhipuai_base import ZhipuAIBase
from ai_starter import get_logger

logger = get_logger(__name__)


@register_llm('zhipuai')
class QwenAgentChatZhipuAI(BaseChatModel):
    """
    Qwen-Agent ZhipuAI 适配器（组合模式）

    继承 Qwen-Agent 的 BaseChatModel 基类，实现 ZhipuAI API 调用。

    实现的抽象方法：
    - _chat_with_functions(): 处理带函数调用的对话
    - _chat_stream(): 流式生成
    - _chat_no_stream(): 非流式生成

    使用组合模式：包含 ZhipuAIBase 实例而非继承它

    配置内聚设计：所有配置从 config.yaml 读取，不接受外部参数覆盖
    """

    def __init__(self, cfg: Optional[Dict] = None, **kwargs):
        """
        初始化 Qwen-Agent ZhipuAI 适配器

        Args:
            cfg: Qwen-Agent 配置（注意：model/temperature 从 config.yaml 读取）
            **kwargs: 其他参数（model/temperature 参数会被忽略）
        """
        # 初始化 BaseChatModel（Qwen-Agent 基类）
        super().__init__(cfg=cfg, **kwargs)

        # 创建 ZhipuAIBase 实例（总是从 config.yaml 读取配置）
        self._zhipu = ZhipuAIBase()

        logger.info(f"QwenAgentChatZhipuAI initialized (model: {self._zhipu.model}, temperature: {self._zhipu.temperature})")

    # ========== Qwen-Agent BaseChatModel 必须实现的属性 ==========

    @property
    def support_multimodal_input(self) -> bool:
        """是否支持多模态输入（Qwen-Agent 接口）"""
        return False

    @property
    def support_multimodal_output(self) -> bool:
        """是否支持多模态输出（Qwen-Agent 接口）"""
        return False

    @property
    def support_audio_input(self) -> bool:
        """是否支持音频输入（Qwen-Agent 接口）"""
        return False

    # ========== Qwen-Agent BaseChatModel 必须实现的方法 ==========

    def _chat_with_functions(
        self,
        messages: List[QwenMessage],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[QwenMessage], Iterator[List[QwenMessage]]]:
        """
        处理带函数调用的对话（Qwen-Agent 接口）

        Args:
            messages: 输入消息列表
            functions: 函数定义列表
            stream: 是否使用流式生成
            delta_stream: 是否增量流式输出
            generate_cfg: 生成配置

        Returns:
            生成的消息列表或迭代器
        """
        raise NotImplementedError("Function calling is not supported yet for ZhipuAI")

    def _chat_stream(
        self,
        messages: List[QwenMessage],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[QwenMessage]]:
        """
        流式生成（Qwen-Agent 接口）

        Args:
            messages: 输入消息列表
            delta_stream: 是否增量流式输出
            generate_cfg: 生成配置

        Returns:
            生成的消息迭代器
        """
        # 转换消息格式
        api_messages = []
        for msg in messages:
            api_messages.append({
                'role': msg.role,
                'content': msg.content
            })

        try:
            # 委托给 ZhipuAIBase 调用 API
            result = self._zhipu._call_api(
                messages=api_messages,
                temperature=generate_cfg.get('temperature', self._zhipu.temperature),
                timeout=generate_cfg.get('timeout', 60)
            )

            content = result["choices"][0]["message"]["content"]

            # 返回完整消息（目前不支持真正的流式）
            yield [QwenMessage(role=ASSISTANT, content=content)]

        except Exception as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[QwenMessage],
        generate_cfg: dict,
    ) -> List[QwenMessage]:
        """
        非流式生成（Qwen-Agent 接口）

        Args:
            messages: 输入消息列表
            generate_cfg: 生成配置

        Returns:
            生成的消息列表
        """
        # 转换消息格式
        api_messages = []
        for msg in messages:
            api_messages.append({
                'role': msg.role,
                'content': msg.content
            })

        try:
            # 委托给 ZhipuAIBase 调用 API
            result = self._zhipu._call_api(
                messages=api_messages,
                temperature=generate_cfg.get('temperature', self._zhipu.temperature),
                timeout=generate_cfg.get('timeout', 60)
            )

            content = result["choices"][0]["message"]["content"]
            return [QwenMessage(role=ASSISTANT, content=content)]

        except Exception as ex:
            raise ModelServiceError(exception=ex)

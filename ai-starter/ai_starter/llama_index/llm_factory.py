"""ZhipuAI LLM Factory"""
import os
from llama_index.llms.openai_like import OpenAILike
from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.http_client.http_client_factory import HttpClientFactory

logger = get_logger(__name__)


class ZhipuLLMFactory:
    """Factory for creating ZhipuAI LLM instances."""

    @staticmethod
    def create(
        model: str | None = None,
        api_base: str | None = None,
    ) -> OpenAILike:
        """Create ZhipuAI LLM instance.

        API key is read from ZHIPUAI_API_KEY environment variable.
        Other configurations are read from config.yaml.
        Priority: parameter > config.yaml

        Config structure:
            zhipu:
              api_base: "https://open.bigmodel.cn/api/paas/v4/"
              llm:
                model: "glm-4-flash"

        Args:
            model: Model name. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            OpenAILike LLM instance.

        Raises:
            ValueError: If ZHIPUAI_API_KEY env var is not set.
            KeyError: If required config keys are missing.

        Examples:
            >>> llm = ZhipuLLMFactory.create()
            >>> llm = ZhipuLLMFactory.create(model="glm-4-plus")
        """
        config = Config()

        # API Key 从环境变量读取
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("环境变量 ZHIPUAI_API_KEY 未设置，请先设置后再运行")

        # 其他配置必须在 config.yaml 中显式配置
        model = model or config.get_required("zhipu.llm.model")
        api_base = api_base or config.get_required("zhipu.api_base")

        # Create HTTP client (automatically reads from http: config)
        http_client = HttpClientFactory.create()

        logger.info(f"初始化智谱AI LLM: {model}")

        llm = OpenAILike(
            model=model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            http_client=http_client,
        )

        logger.info("✓ 智谱AI LLM 创建成功")
        return llm

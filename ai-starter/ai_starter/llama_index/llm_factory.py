"""ZhipuAI LLM Factory"""
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
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> OpenAILike:
        """Create ZhipuAI LLM instance.

        All configurations are read from config.yaml.
        Priority: parameter > config.yaml > default value

        Config structure:
            zhipu:
              api_key: "your_api_key"
              api_base: "https://open.bigmodel.cn/api/paas/v4/"  # optional
              llm:
                model: "glm-4-flash"

        Args:
            model: Model name. If None, reads from config.yaml.
            api_key: API key. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            OpenAILike LLM instance.

        Raises:
            ValueError: If api_key is not provided in config or parameter.

        Examples:
            >>> # Use config.yaml
            >>> llm = ZhipuLLMFactory.create()
            >>>
            >>> # Override specific parameter
            >>> llm = ZhipuLLMFactory.create(model="glm-4-plus")
        """
        config = Config()

        # Priority: parameter > config > default
        model = model or config.get("zhipu.llm.model", "glm-4-flash")
        api_key = api_key or config.get("zhipu.api_key")
        api_base = api_base or config.get("zhipu.api_base", "https://open.bigmodel.cn/api/paas/v4/")

        if not api_key:
            raise ValueError(
                "api_key is required. Please set zhipu.api_key in config.yaml "
                "or pass api_key parameter."
            )

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

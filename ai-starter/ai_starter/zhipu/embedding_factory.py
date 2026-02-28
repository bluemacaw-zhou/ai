"""ZhipuAI Embedding Factory"""
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.http_client.http_client_factory import HttpClientFactory

logger = get_logger(__name__)


class ZhipuEmbeddingFactory:
    """Factory for creating ZhipuAI Embedding instances."""

    @staticmethod
    def create(
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> OpenAILikeEmbedding:
        """Create ZhipuAI Embedding instance.

        All configurations are read from config.yaml.
        Priority: parameter > config.yaml > default value

        Config structure:
            zhipu:
              api_key: "your_api_key"
              api_base: "https://open.bigmodel.cn/api/paas/v4/"  # optional
              embedding:
                model: "embedding-3"

        Args:
            model: Model name. If None, reads from config.yaml.
            api_key: API key. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            OpenAILikeEmbedding instance.

        Raises:
            ValueError: If api_key is not provided in config or parameter.

        Examples:
            >>> # Use config.yaml
            >>> embedding = ZhipuEmbeddingFactory.create()
            >>>
            >>> # Override specific parameter
            >>> embedding = ZhipuEmbeddingFactory.create(model="embedding-2")
        """
        config = Config()

        # Priority: parameter > config > default
        model = model or config.get("zhipu.embedding.model", "embedding-3")
        api_key = api_key or config.get("zhipu.api_key")
        api_base = api_base or config.get("zhipu.api_base", "https://open.bigmodel.cn/api/paas/v4/")

        if not api_key:
            raise ValueError(
                "api_key is required. Please set zhipu.api_key in config.yaml "
                "or pass api_key parameter."
            )

        # Use HttpClientFactory for consistent HTTP configuration
        http_client = HttpClientFactory.create()

        logger.info(f"初始化智谱AI Embedding: {model}")

        # Create embedding
        embedding = OpenAILikeEmbedding(
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            http_client=http_client,
        )

        logger.info("✓ 智谱AI Embedding 创建成功")
        return embedding

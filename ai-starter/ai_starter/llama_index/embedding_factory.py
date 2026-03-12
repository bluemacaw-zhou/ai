"""ZhipuAI Embedding Factory"""
import os
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
        api_base: str | None = None,
    ) -> OpenAILikeEmbedding:
        """Create ZhipuAI Embedding instance.

        API key is read from ZHIPUAI_API_KEY environment variable.
        Other configurations are read from config.yaml.
        Priority: parameter > config.yaml

        Config structure:
            zhipu:
              api_base: "https://open.bigmodel.cn/api/paas/v4/"
              embedding:
                model: "embedding-3"

        Args:
            model: Model name. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            OpenAILikeEmbedding instance.

        Raises:
            ValueError: If ZHIPUAI_API_KEY env var is not set.
            KeyError: If required config keys are missing.

        Examples:
            >>> embedding = ZhipuEmbeddingFactory.create()
            >>> embedding = ZhipuEmbeddingFactory.create(model="embedding-2")
        """
        config = Config()

        # API Key 从环境变量读取
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("环境变量 ZHIPUAI_API_KEY 未设置，请先设置后再运行")

        # 其他配置必须在 config.yaml 中显式配置
        model = model or config.get_required("zhipu.embedding.model")
        api_base = api_base or config.get_required("zhipu.api_base")

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

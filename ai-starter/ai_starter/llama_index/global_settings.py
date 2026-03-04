"""ZhipuAI Global Settings Configuration"""
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.llama_index.llm_factory import ZhipuLLMFactory
from ai_starter.llama_index.embedding_factory import ZhipuEmbeddingFactory

logger = get_logger(__name__)


class ZhipuGlobalSettings:
    """Configure global LlamaIndex settings for ZhipuAI models."""

    @staticmethod
    def setup(
        llm_model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[OpenAILike, OpenAILikeEmbedding]:
        """Setup global LLM and Embedding settings for LlamaIndex.

        This method creates LLM and Embedding instances and configures them
        in Settings.llm and Settings.embed_model for global use in LlamaIndex.

        All configurations are read from config.yaml.
        Priority: parameter > config.yaml > default value

        Config structure:
            zhipu:
              api_key: "your_api_key"
              api_base: "https://open.bigmodel.cn/api/paas/v4/"  # optional
              llm:
                model: "glm-4-flash"
              embedding:
                model: "embedding-3"

        Args:
            llm_model: LLM model name. If None, reads from config.yaml.
            embedding_model: Embedding model name. If None, reads from config.yaml.
            api_key: API key. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            Tuple of (llm, embed_model).

        Raises:
            ValueError: If api_key is not provided in config or parameter.

        Examples:
            >>> # Use config.yaml
            >>> llm, embedding = ZhipuGlobalSettings.setup()
            >>>
            >>> # Override specific parameters
            >>> llm, embedding = ZhipuGlobalSettings.setup(
            ...     llm_model="glm-4-plus",
            ...     embedding_model="embedding-2"
            ... )
        """
        # Reuse factory methods to avoid code duplication
        Settings.llm = ZhipuLLMFactory.create(llm_model, api_key, api_base)
        Settings.embed_model = ZhipuEmbeddingFactory.create(embedding_model, api_key, api_base)

        logger.info("✓ 智谱AI 全局设置配置完成")

        return Settings.llm, Settings.embed_model

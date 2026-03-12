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
        api_base: str | None = None,
    ) -> tuple[OpenAILike, OpenAILikeEmbedding]:
        """Setup global LLM and Embedding settings for LlamaIndex.

        API key is read from ZHIPUAI_API_KEY environment variable.
        Other configurations are read from config.yaml.
        Priority: parameter > config.yaml

        Config structure:
            zhipu:
              api_base: "https://open.bigmodel.cn/api/paas/v4/"
              llm:
                model: "glm-4-flash"
              embedding:
                model: "embedding-3"

        Args:
            llm_model: LLM model name. If None, reads from config.yaml.
            embedding_model: Embedding model name. If None, reads from config.yaml.
            api_base: API base URL. If None, reads from config.yaml.

        Returns:
            Tuple of (llm, embed_model).

        Raises:
            ValueError: If ZHIPUAI_API_KEY env var is not set.
            KeyError: If required config keys are missing.

        Examples:
            >>> llm, embedding = ZhipuGlobalSettings.setup()
            >>> llm, embedding = ZhipuGlobalSettings.setup(llm_model="glm-4-plus")
        """
        Settings.llm = ZhipuLLMFactory.create(llm_model, api_base)
        Settings.embed_model = ZhipuEmbeddingFactory.create(embedding_model, api_base)

        logger.info("✓ 智谱AI 全局设置配置完成")

        return Settings.llm, Settings.embed_model

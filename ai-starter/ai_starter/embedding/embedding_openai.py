from typing import List
from ai_starter.embedding.embedding_interface import EmbeddingInterface
from ai_starter import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI Embedding 实现（待实现）"""

    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        初始化 OpenAI Embedding 服务

        Args:
            api_key: OpenAI API 密钥
            model: 模型名称，可选:
                - text-embedding-ada-002 (默认)
                - text-embedding-3-small
                - text-embedding-3-large
        """
        self.api_key = api_key
        self.model = model

        logger.info(f"OpenAIEmbedding initialized (model: {model}, not implemented)")

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示（待实现）

        Args:
            text: 输入文本

        Returns:
            List[float]: 文本的向量表示

        Raises:
            NotImplementedError: 该方法尚未实现
        """
        raise NotImplementedError(
            "OpenAI Embedding 功能尚未实现。\n"
            "需要安装: pip install openai\n"
            f"模型: {self.model}"
        )

    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            str: 模型名称
        """
        return f"OpenAI-{self.model}"

    def get_vector_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        return self.MODEL_DIMENSIONS.get(self.model, 1536)

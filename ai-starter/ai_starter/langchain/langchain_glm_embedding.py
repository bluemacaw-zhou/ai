"""
LangChain 兼容的 GLM Embedding 适配器

使用多重继承同时实现 LangChain 和内部 Embedding 接口：
- 继承 LangChain Embeddings（LangChain 核心接口）
- 继承 EmbeddingInterface（内部 Embedding 接口）
"""

from typing import List
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from ai_starter.embedding.embedding_glm import GLMEmbedding
from ai_starter.embedding.embedding_interface import EmbeddingInterface
from ai_starter.core.log.logging_utils import get_logger

logger = get_logger(__name__)


class LangChainGLMEmbedding(LangChainEmbeddings, EmbeddingInterface):
    """
    LangChain 兼容的智谱 AI Embedding 适配器

    实现接口:
    - LangChain Embeddings
      - embed_documents(): 批量嵌入
      - embed_query(): 单条嵌入

    - EmbeddingInterface
      - get_embedding(): 获取向量
      - get_model_name(): 模型名称
      - get_vector_dimension(): 向量维度

    内部组合 GLMEmbedding 实现核心功能。
    """

    def __init__(
        self,
        model: str = None
    ):
        """
        初始化 LangChain GLM Embedding 适配器

        Args:
            model: 模型名称（可选，自动从配置读取）

        Examples:
            >>> # 自动从配置读取
            >>> embedding = LangChainGLMEmbedding()
            >>>
            >>> # 嵌入多个文档
            >>> vectors = embedding.embed_documents(["文本1", "文本2"])
            >>>
            >>> # 嵌入查询
            >>> query_vector = embedding.embed_query("查询文本")
        """
        self.glm_embedding = GLMEmbedding(
            model=model
        )
        logger.info("LangChainGLMEmbedding initialized")

    # ========== LangChain 接口实现 ==========

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档（LangChain 接口）

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        logger.debug(f"Embedding {len(texts)} documents")

        vectors = []
        for i, text in enumerate(texts):
            vector = self.glm_embedding.get_embedding(text)
            vectors.append(vector)

            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} documents")

        logger.info(f"Embedded {len(texts)} documents successfully")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本（LangChain 接口）

        Args:
            text: 查询文本

        Returns:
            List[float]: 向量
        """
        logger.debug(f"Embedding query: {text[:50]}...")
        vector = self.glm_embedding.get_embedding(text)
        logger.debug("Query embedded successfully")
        return vector

    # ========== EmbeddingInterface 接口实现 ==========

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示（EmbeddingInterface 接口）

        委托给内部 GLMEmbedding
        """
        return self.glm_embedding.get_embedding(text)

    def get_model_name(self) -> str:
        """
        获取模型名称（EmbeddingInterface 接口）
        """
        return self.glm_embedding.get_model_name()

    def get_vector_dimension(self) -> int:
        """
        获取向量维度（EmbeddingInterface 接口）
        """
        return self.glm_embedding.get_vector_dimension()

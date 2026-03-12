import os
from typing import List
from zhipuai import ZhipuAI
from ai_starter.embedding.embedding_interface import EmbeddingInterface
from ai_starter import get_logger, Config, HttpClientFactory

logger = get_logger(__name__)


class GLMEmbedding(EmbeddingInterface):
    """智谱 AI Embedding 实现"""

    MODEL_DIMENSIONS = {
        "embedding-2": 1024,
        "embedding-3": 1024,
    }

    def __init__(
        self,
        model: str = None
    ):
        """
        初始化 GLM Embedding 服务

        API Key 从环境变量 ZHIPUAI_API_KEY 读取。

        Args:
            model: 模型名称，可选 "embedding-2" 或 "embedding-3"（可选，从配置读取）

        Raises:
            ValueError: 如果环境变量 ZHIPUAI_API_KEY 未设置
        """
        config = Config()

        # API Key 从环境变量读取
        self.api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("环境变量 ZHIPUAI_API_KEY 未设置，请先设置后再运行")

        self.model = model or config.get_required("zhipu.embedding.model")

        http_client = HttpClientFactory.create()
        self.client = ZhipuAI(api_key=self.api_key, http_client=http_client)

        logger.info(f"GLMEmbedding initialized successfully (model: {self.model})")

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示

        Args:
            text: 输入文本

        Returns:
            List[float]: 文本的向量表示
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            str: 模型名称
        """
        return f"GLM-{self.model}"

    def get_vector_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        return self.MODEL_DIMENSIONS.get(self.model, 1024)

    def __del__(self):
        """清理资源"""
        try:
            self.client.close()
        except Exception:
            pass

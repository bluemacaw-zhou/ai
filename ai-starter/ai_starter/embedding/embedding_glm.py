from typing import List
from zhipuai import ZhipuAI
from .embedding_interface import EmbeddingInterface
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.core.config.config import Config
from ai_starter.core.http_client.http_client_factory import HttpClientFactory

logger = get_logger(__name__)


class GLMEmbedding(EmbeddingInterface):
    """智谱 AI Embedding 实现"""

    MODEL_DIMENSIONS = {
        "embedding-2": 1024,
        "embedding-3": 1024,
    }

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        """
        初始化 GLM Embedding 服务

        Args:
            api_key: 智谱 AI API 密钥（可选，覆盖配置文件）
            model: 模型名称，可选 "embedding-2" 或 "embedding-3"（可选，覆盖配置文件）

        Raises:
            ValueError: 如果无法获取 api_key
        """
        config = Config()

        self.api_key = api_key or config.get("api.zhipuai.key")
        self.model = model or config.get("models.embedding.model") or config.get("embedding.model", "embedding-2")

        if not self.api_key:
            raise ValueError("api_key is required. 请在配置文件中配置 api.zhipuai.key，或通过参数传入。")

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

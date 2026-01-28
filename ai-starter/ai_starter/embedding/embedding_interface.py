from abc import ABC, abstractmethod
from typing import List


class EmbeddingInterface(ABC):
    """Embedding 接口定义（抽象基类）"""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示

        Args:
            text: 输入文本

        Returns:
            List[float]: 文本的向量表示（浮点数数组）
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            str: 模型名称
        """
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        pass

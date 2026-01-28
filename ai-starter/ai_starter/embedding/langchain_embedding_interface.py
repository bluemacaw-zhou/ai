"""
LangChain Embeddings 接口定义

显式声明 LangChain Embeddings 需要实现的接口方法，
让类的接口实现关系一目了然。
"""

from abc import ABC, abstractmethod
from typing import List

# LangChain imports
try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
except ImportError:
    # 如果没有安装 langchain，提供占位符
    LangChainEmbeddings = ABC


class LangChainEmbeddingInterface(LangChainEmbeddings):
    """
    LangChain Embeddings 接口

    继承 LangChain 的 Embeddings 并显式声明核心接口方法。
    任何实现此接口的类都可以作为 LangChain 的 Embeddings 使用。
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档 - LangChain 核心接口

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表，每个文本对应一个向量
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本 - LangChain 核心接口

        Args:
            text: 查询文本

        Returns:
            List[float]: 向量
        """
        pass

"""
LangChain ChromaDB 向量存储组件
"""

from typing import List, Optional
from langchain_community.vectorstores import Chroma
from ai_starter.chromadb.chromadb import ChromaDB
from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger

logger = get_logger(__name__)


class LangchainChromadb:
    """LangChain ChromaDB 向量存储管理器"""

    def __init__(self, embeddings=None):
        """
        初始化向量存储

        Args:
            embeddings: LangChain Embeddings 实例
        """
        config = Config()
        self.collection_name = config.get("chromadb.collection_name", "test_collection")

        self.embeddings = embeddings
        self.chroma_client = ChromaDB()

        # 如果提供了 embeddings，则初始化 vectorstore 连接到现有 collection
        if self.embeddings is not None:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                client=self.chroma_client.client
            )
            logger.info(f"LangchainChromadb initialized (collection={self.collection_name})")
        else:
            logger.info(f"LangchainChromadb initialized without embeddings (collection={self.collection_name})")

    def clear_collection(self):
        """
        清空当前集合的所有数据

        注意：这会删除集合中的所有文档
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"已清空集合: {self.collection_name}")

            # 重新初始化 vectorstore（连接到新的空 collection）
            if hasattr(self, 'vectorstore'):
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    client=self.chroma_client.client
                )
        except Exception as e:
            # 集合不存在时会抛出异常，可以忽略
            logger.debug(f"清空集合时出错 (集合可能不存在): {e}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> 'LangchainChromadb':
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）

        Returns:
            LangchainChromadb: 返回自身，支持链式调用
        """
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量存储未初始化，请确保在初始化时传入 embeddings")

        logger.info(f"开始添加 {len(texts)} 个文本到向量库")

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

        logger.info(f"成功添加 {len(texts)} 个文档到向量库")

        return self

    def get_retriever(self, k: int = 4):
        """
        获取 LangChain Retriever

        Args:
            k: 返回的文档数量

        Returns:
            Retriever: LangChain 检索器
        """
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量存储未初始化，请确保在初始化时传入 embeddings")

        logger.debug(f"创建 retriever (k={k})")
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

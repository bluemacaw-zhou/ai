import os
from typing import List
import chromadb
from chromadb.config import Settings
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.core.config.config import Config

logger = get_logger(__name__)


class ChromaDB:
    """ChromaDB 数据库连接类（仅负责数据库连接和基础操作）"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None
    ):
        """
        初始化 ChromaDB 数据库连接

        Args:
            host: ChromaDB 服务器地址（可选，覆盖配置文件）
            port: ChromaDB 服务器端口（可选，覆盖配置文件）
            username: 认证用户名（可选，覆盖配置文件）
            password: 认证密码（可选，覆盖配置文件）
        """
        # 从全局配置读取，参数可覆盖（懒加载）
        config = Config()

        self.host = host or config.get("database.chromadb.host", "localhost")
        self.port = port or config.get("database.chromadb.port", 8000)
        self.username = username or config.get("database.chromadb.username", "admin")
        self.password = password or config.get("database.chromadb.password", "admin")

        os.environ['NO_PROXY'] = self.host
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{self.username}:{self.password}"
            )
        )

        logger.info(f"ChromaDB connected successfully: {self.host}:{self.port}")

    def heartbeat(self) -> int:
        """
        测试数据库连接心跳

        Returns:
            int: 心跳时间戳
        """
        return self.client.heartbeat()

    def get_or_create_collection(self, collection_name: str):
        """
        获取或创建集合

        Args:
            collection_name: 集合名称

        Returns:
            Collection: ChromaDB 集合对象
        """
        return self.client.get_or_create_collection(collection_name)

    def get_collection(self, collection_name: str):
        """
        获取已存在的集合

        Args:
            collection_name: 集合名称

        Returns:
            Collection: ChromaDB 集合对象
        """
        return self.client.get_collection(collection_name)

    def delete_collection(self, collection_name: str):
        """
        删除集合

        Args:
            collection_name: 集合名称
        """
        self.client.delete_collection(collection_name)

    def list_collections(self) -> List:
        """
        列出所有集合

        Returns:
            List: 集合列表
        """
        return self.client.list_collections()

    def get_collection_data(self, collection_name: str, limit: int = None):
        """
        获取集合中的所有数据

        Args:
            collection_name: 集合名称
            limit: 限制返回数量（可选，默认返回全部）

        Returns:
            Dict: 包含 ids, documents, embeddings, metadatas 的字典
        """
        collection = self.client.get_collection(collection_name)
        if limit:
            return collection.get(limit=limit)
        return collection.get()

    def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        vectors: List[List[float]],
        ids: List[str],
        metadatas: List[dict] = None
    ):
        """
        添加文本和向量到集合

        Args:
            collection_name: 集合名称
            texts: 文本列表
            vectors: 向量列表
            ids: 文档 ID 列表
            metadatas: 元数据列表（可选）
        """
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            documents=texts,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} documents to collection '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        n_results: int = 4
    ) -> dict:
        """
        向量搜索

        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            n_results: 返回结果数量

        Returns:
            dict: 包含 documents, distances, metadatas 的字典
        """
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        return results

    def count(self, collection_name: str) -> int:
        """
        获取集合中的文档数量

        Args:
            collection_name: 集合名称

        Returns:
            int: 文档数量
        """
        collection = self.get_collection(collection_name)
        return collection.count()

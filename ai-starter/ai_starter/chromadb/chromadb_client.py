"""ChromaDB 向量数据库客户端（集成 Embedding）"""

import os
from typing import List, Optional, Dict, Any, overload
import chromadb
from chromadb.config import Settings
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.core.config.config import Config
from ai_starter.embedding.embedding_interface import EmbeddingInterface

logger = get_logger(__name__)


class ChromaDB:
    """
    ChromaDB 向量数据库客户端

    设计理念：
    - 向量数据库需要 Embedding 模型才能工作（文本 → 向量）
    - 集成 Embedding 后，接口自动处理文本和向量转换
    - 提供简洁的统一接口，无需区分文本/向量操作
    """

    def __init__(
        self,
        embedding: EmbeddingInterface | None = None,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None
    ):
        """
        初始化 ChromaDB 数据库连接

        Args:
            embedding: Embedding 接口实现（推荐提供，可自动处理文本）
            host: ChromaDB 服务器地址
            port: ChromaDB 服务器端口
            username: 认证用户名
            password: 认证密码
        """
        config = Config()

        self.host = host or config.get("chromadb.host", "localhost")
        self.port = port or config.get("chromadb.port", 8000)
        self.username = username or config.get("chromadb.username", "admin")
        self.password = password or config.get("chromadb.password", "admin")

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

        self.embedding = embedding

        if self.embedding:
            logger.info(f"ChromaDB connected: {self.host}:{self.port} (embedding: {self.embedding.get_model_name()})")
        else:
            logger.info(f"ChromaDB connected: {self.host}:{self.port} (no embedding)")

    # ========== 基础操作 ==========

    def heartbeat(self) -> int:
        """测试数据库连接心跳"""
        return self.client.heartbeat()

    def delete(self, collection_name: str) -> None:
        """删除集合"""
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    def list(self) -> List[str]:
        """列出所有集合名称"""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def count(self, collection_name: str) -> int:
        """获取集合中的文档数量"""
        collection = self.client.get_collection(collection_name)
        return collection.count()

    # ========== 添加数据 ==========

    @overload
    def add(
        self,
        collection_name: str,
        texts: List[str],
        ids: List[str] | None = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """添加文本（自动向量化）"""
        ...

    @overload
    def add(
        self,
        collection_name: str,
        texts: List[str],
        vectors: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """添加文本和向量（不使用 embedding）"""
        ...

    def add(
        self,
        collection_name: str,
        texts: List[str],
        vectors: List[List[float]] | None = None,
        ids: List[str] | None = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        添加数据到集合

        Args:
            collection_name: 集合名称
            texts: 文本列表
            vectors: 向量列表（可选，不提供则自动使用 embedding 转换）
            ids: 文档 ID 列表（可选，默认自动生成）
            metadatas: 元数据列表（可选）

        Returns:
            List[str]: 文档 ID 列表

        Raises:
            ValueError: 未提供 vectors 且未初始化 embedding
        """
        # 自动转换文本为向量
        if vectors is None:
            if self.embedding is None:
                raise ValueError("Embedding not provided. Please pass vectors directly or initialize with embedding.")
            vectors = [self.embedding.get_embedding(text) for text in texts]
            logger.info(f"Converted {len(texts)} texts to vectors")
        else:
            if len(vectors) != len(texts):
                raise ValueError(f"vectors length {len(vectors)} != texts length {len(texts)}")

        # 自动生成 ID
        if ids is None:
            try:
                existing_count = self.count(collection_name)
            except Exception:
                existing_count = 0
            ids = [f"doc_{existing_count + i:04d}" for i in range(len(texts))]

        # 添加到集合
        collection = self.client.get_or_create_collection(collection_name)
        collection.add(
            documents=texts,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} documents to '{collection_name}'")
        return ids

    # ========== 搜索数据 ==========

    @overload
    def search(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """搜索相似文本（自动向量化）"""
        ...

    @overload
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """使用向量搜索"""
        ...

    def search(
        self,
        collection_name: str,
        query: str | List[float],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        搜索相似数据

        Args:
            collection_name: 集合名称
            query: 查询文本或查询向量
            n_results: 返回结果数量
            where: 元数据过滤条件（可选）

        Returns:
            Dict: 包含 documents, distances, metadatas 的字典

        Raises:
            ValueError: query 是文本但未初始化 embedding
        """
        # 自动转换查询文本为向量
        if isinstance(query, str):
            if self.embedding is None:
                raise ValueError("Embedding not provided. Please pass query vector directly or initialize with embedding.")
            query_vector = self.embedding.get_embedding(query)
            logger.info(f"Searching: '{query[:50]}...'")
        else:
            query_vector = query
            logger.debug(f"Searching with vector")

        collection = self.client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where
        )

        if isinstance(query, str):
            logger.info(f"Found {len(results['documents'][0])} results")
        return results

    # ========== 获取数据 ==========

    def get(
        self,
        collection_name: str,
        limit: int | None = None
    ) -> Dict[str, Any]:
        """
        获取集合中的数据

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

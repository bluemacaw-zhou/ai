from typing import List, Dict, Optional, Any
import numpy as np
from ai_starter import ChromaDB, EmbeddingInterface, GLMEmbedding, get_logger, trace_context, Config

logger = get_logger(__name__)


class VectorOperations:
    """向量操作类（负责文本向量化和相似度搜索等业务逻辑）"""

    def __init__(self, db: ChromaDB, embedding: EmbeddingInterface):
        """
        初始化向量操作类

        Args:
            db: ChromaDB 数据库实例
            embedding: Embedding 接口实现（GLMEmbedding、OpenAIEmbedding 等）
        """
        self.db = db
        self.embedding = embedding
        logger.info(f"VectorOperations initialized (embedding: {embedding.get_model_name()})")

    def text_to_vector(self, text: str) -> List[float]:
        """
        将文本转换为向量

        Args:
            text: 输入文本

        Returns:
            List[float]: 文本的向量表示
        """
        logger.debug(f"Converting text to vector: '{text[:20]}...'")
        vector = self.embedding.get_embedding(text)
        logger.debug(f"Vector generated, dimension: {len(vector)}")
        return vector

    def add_texts_to_db(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        将文本列表添加到数据库（自动转为向量）

        Args:
            collection_name: 集合名称
            texts: 文本列表
            metadatas: 元数据列表（可选）
            ids: 文档ID列表（可选，默认自动生成）

        Returns:
            bool: 是否添加成功
        """
        try:
            collection = self.db.get_or_create_collection(collection_name)

            # 批量转换文本为向量
            logger.info(f"Converting {len(texts)} texts to vectors...")
            vectors = [self.text_to_vector(text) for text in texts]

            # 自动生成 ID
            if ids is None:
                ids = [f"doc_{i:04d}" for i in range(len(texts))]

            # 添加到数据库
            collection.add(
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(texts)} texts to collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            return False

    def search_similar_texts(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        在数据库中搜索与查询文本相似的内容

        Args:
            collection_name: 集合名称
            query_text: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件（可选）

        Returns:
            Dict: 查询结果，包含 documents, distances, metadatas 等
        """
        try:
            collection = self.db.get_collection(collection_name)

            # 将查询文本转为向量
            logger.info(f"Searching for similar texts: '{query_text}'")
            query_vector = self.text_to_vector(query_text)

            # 执行向量相似度搜索
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where
            )

            logger.info(f"Found {len(results['documents'][0])} similar results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {}

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的相似度（余弦相似度）

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            float: 相似度分数（-1 到 1，越接近 1 越相似）
        """
        logger.debug(f"Calculating text similarity")
        logger.debug(f"  Text 1: '{text1[:30]}...'")
        logger.debug(f"  Text 2: '{text2[:30]}...'")

        # 获取向量
        vec1 = np.array(self.text_to_vector(text1))
        vec2 = np.array(self.text_to_vector(text2))

        # 计算余弦相似度
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        logger.info(f"Similarity: {similarity:.4f}")
        return float(similarity)

    def get_all_texts(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合中的所有文本

        Args:
            collection_name: 集合名称

        Returns:
            Dict: 所有文档数据
        """
        try:
            collection = self.db.get_collection(collection_name)
            results = collection.get()
            logger.info(f"Retrieved {len(results['ids'])} documents")
            return results
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return {}


def test_add_and_search():
    """测试添加文本和相似度搜索"""
    logger.info("测试: 添加文本并搜索相似内容")

    # 初始化
    db = ChromaDB()
    embedding = GLMEmbedding()
    vector_ops = VectorOperations(db, embedding)
    collection_name = "test_collection"

    # 准备测试数据
    texts = [
        "雨伞是最常见的雨具，可以遮挡雨水",
        "雨衣能够保护全身不被雨水淋湿",
        "雨靴防水性能好，适合在雨天穿着",
        "雨帽可以保护头部免受雨水",
        "防水包能保护包内物品不受潮"
    ]

    metadatas = [
        {"type": "umbrella", "waterproof": True, "portable": True},
        {"type": "raincoat", "waterproof": True, "portable": False},
        {"type": "rain_boots", "waterproof": True, "portable": False},
        {"type": "rain_hat", "waterproof": True, "portable": True},
        {"type": "waterproof_bag", "waterproof": True, "portable": True}
    ]

    # 添加文本到数据库
    vector_ops.add_texts_to_db(
        collection_name=collection_name,
        texts=texts,
        metadatas=metadatas
    )

    # 搜索相似文本
    query_text = "我需要一个能遮雨的工具"
    results = vector_ops.search_similar_texts(
        collection_name=collection_name,
        query_text=query_text,
        n_results=3
    )

    logger.info(f"查询: '{query_text}'")
    logger.info("最相似的内容:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        logger.info(f"  {i+1}. {doc}")
        logger.info(f"      距离: {distance:.4f}")


def main():
    """主测试函数"""
    logger.info("=" * 50)
    test_add_and_search()
    logger.info("=" * 50)
    logger.info("测试完成")


if __name__ == "__main__":
    main()

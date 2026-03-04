"""ChromaDB 向量数据库测试"""

from typing import Dict, Any
import numpy as np
from ai_starter import get_logger, Config
from ai_starter.chromadb import ChromaDB
from ai_starter.embedding import GLMEmbedding

logger = get_logger(__name__)


def calculate_text_similarity(embedding, text1: str, text2: str) -> float:
    """计算两个文本之间的相似度（余弦相似度）"""
    logger.debug(f"Calculating text similarity")
    logger.debug(f"  Text 1: '{text1[:30]}...'")
    logger.debug(f"  Text 2: '{text2[:30]}...'")

    # 获取向量
    vec1 = np.array(embedding.get_embedding(text1))
    vec2 = np.array(embedding.get_embedding(text2))

    # 计算余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    logger.info(f"Similarity: {similarity:.4f}")
    return float(similarity)


def test_add_and_search(collection_name: str):
    """测试添加文本和相似度搜索"""
    logger.info("测试: 添加文本并搜索相似内容")

    # 初始化 ChromaDB（集成 Embedding）
    db = ChromaDB(embedding=GLMEmbedding())

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

    # 添加文本到数据库（自动向量化）
    db.add(collection_name, texts, metadatas=metadatas)

    # 搜索相似文本
    query_text = "我需要一个能遮雨的工具"
    results = db.search(collection_name, query_text, n_results=3)

    logger.info(f"查询: '{query_text}'")
    logger.info("最相似的内容:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        logger.info(f"  {i+1}. {doc}")
        logger.info(f"      距离: {distance:.4f}")


def test_text_similarity():
    """测试文本相似度计算"""
    logger.info("测试: 文本相似度计算")

    embedding = GLMEmbedding()

    test_pairs = [
        ("机器学习是AI的分支", "深度学习是机器学习的子集"),
        ("Python 编程语言", "自然语言处理"),
        ("猫是一种动物", "狗是一种宠物"),
    ]

    for text1, text2 in test_pairs:
        similarity = calculate_text_similarity(embedding, text1, text2)
        logger.info(f"\n文本1: {text1}")
        logger.info(f"文本2: {text2}")
        logger.info(f"相似度: {similarity:.4f}")


def test_basic_operations(collection_name: str):
    """测试基础操作"""
    logger.info("测试: 基础操作")

    db = ChromaDB(embedding=GLMEmbedding())

    # 测试文档数量
    count = db.count(collection_name)
    logger.info(f"当前文档数量: {count}")

    # 测试列出集合
    collections = db.list()
    logger.info(f"所有集合: {collections}")

    # 测试获取数据
    data = db.get(collection_name, limit=2)
    logger.info(f"获取前2条数据: {len(data['ids'])} 条")
    for i, doc_id in enumerate(data['ids']):
        logger.info(f"  [{i+1}] ID: {doc_id}, 内容: {data['documents'][i][:50]}...")


def main():
    """主测试函数"""
    config = Config()
    collection_name = config.get("chromadb.collection.name", "chromadb_demo_collection")

    logger.info("=" * 50)
    logger.info(f"使用集合: {collection_name}")
    logger.info("=" * 50)

    # 先清空集合
    db = ChromaDB(embedding=GLMEmbedding())
    try:
        db.delete(collection_name)
        logger.info(f"已清空集合: {collection_name}")
    except Exception:
        pass

    # 按顺序测试，都操作同一个集合
    logger.info("=" * 50)
    test_add_and_search(collection_name)

    logger.info("=" * 50)
    test_text_similarity()

    logger.info("=" * 50)
    test_basic_operations(collection_name)

    logger.info("=" * 50)
    logger.info("测试完成")


if __name__ == "__main__":
    main()

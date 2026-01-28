"""
Step 3: 向量存储测试
"""

from ai_starter import LangChainGLMEmbedding, LangchainChromadb, get_logger

logger = get_logger(__name__)


def main():
    """测试 LangchainChromadb 组件"""
    logger.info("=" * 60)
    logger.info("Step 3: 向量存储测试")
    logger.info("=" * 60)

    # 创建 embeddings
    embeddings = LangChainGLMEmbedding()

    # 创建向量存储
    storage = LangchainChromadb(embeddings=embeddings)

    # 示例文本块
    texts = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子集，使用神经网络进行学习。",
        "自然语言处理是人工智能的另一个重要领域，处理人类语言。",
        "计算机视觉使计算机能够理解和分析图像和视频。",
        "强化学习是机器学习的一种，通过奖励和惩罚来训练模型。"
    ]

    logger.info(f"添加 {len(texts)} 个文本块到向量库...")
    storage.add_texts(texts)

    # 获取 retriever
    retriever = storage.get_retriever(k=3)

    # 测试检索
    query = "什么是深度学习？"
    logger.info(f"查询: {query}")

    docs = retriever.invoke(query)
    logger.info(f"找到 {len(docs)} 个相关文档")

    for i, doc in enumerate(docs, 1):
        logger.info(f"{i}. {doc.page_content}")


if __name__ == "__main__":
    main()

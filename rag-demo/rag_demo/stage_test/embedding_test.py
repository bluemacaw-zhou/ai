"""
Step 2: 文本向量化测试
"""

from ai_starter import get_logger
from ai_starter.langchain import LangChainGLMEmbedding

logger = get_logger(__name__)


def main():
    """测试 LangChainGLMEmbedding 组件"""
    logger.info("=" * 60)
    logger.info("Step 2: 文本向量化测试")
    logger.info("=" * 60)

    # 创建 embeddings
    embeddings = LangChainGLMEmbedding()

    # 示例文本块
    chunks = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子集，使用神经网络进行学习。",
        "自然语言处理是人工智能的另一个重要领域。"
    ]

    logger.info(f"文本块数量: {len(chunks)}")

    # 向量化文本
    logger.info("开始向量化...")
    vectors = embeddings.embed_documents(chunks)

    logger.info(f"向量化结果:")
    logger.info(f"  - 生成向量数: {len(vectors)}")
    logger.info(f"  - 向量维度: {len(vectors[0])}")
    logger.debug(f"  - 第一个向量前5维: {vectors[0][:5]}")

    # 向量化查询
    query = "什么是机器学习？"
    logger.info(f"查询文本: {query}")
    query_vector = embeddings.embed_query(query)
    logger.info(f"查询向量维度: {len(query_vector)}")
    logger.debug(f"查询向量前5维: {query_vector[:5]}")


if __name__ == "__main__":
    main()

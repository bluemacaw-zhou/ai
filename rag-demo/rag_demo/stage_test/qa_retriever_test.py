"""
Step 4: 问答检索测试
"""

from ai_starter import get_logger
from ai_starter.langchain import LangChainGLMEmbedding, LangchainChromadb
from ..retriever.langchain_qa_retriever import LangchainQARetriever

logger = get_logger(__name__)


def main():
    """测试 LangchainQARetriever 组件"""
    logger.info("=" * 60)
    logger.info("Step 4: 问答检索测试")
    logger.info("=" * 60)

    # 创建 embeddings
    embeddings = LangChainGLMEmbedding()

    # 创建向量存储并添加文档
    storage = LangchainChromadb(embeddings=embeddings)

    # knowledge_base = [
    #     "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。",
    #     "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。",
    #     "自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。",
    #     "计算机视觉使计算机能够从图像和视频中提取信息和理解视觉世界。",
    #     "强化学习是一种机器学习方法，智能体通过与环境交互并获得奖励来学习最优策略。",
    #     "监督学习使用标记的数据来训练模型，模型学习输入和输出之间的映射关系。",
    #     "无监督学习从未标记的数据中发现模式和结构，如聚类和降维。"
    # ]
    #
    # logger.info(f"构建知识库：添加 {len(knowledge_base)} 个文档...")
    # storage.add_texts(knowledge_base)

    # 获取 retriever
    retriever = storage.get_retriever(k=3)

    # 创建问答检索器
    logger.info("初始化问答系统...")
    qa = LangchainQARetriever(retriever)

    # 测试问题
    questions = [
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？",
        "NLP是什么？"
    ]

    for i, question in enumerate(questions, 1):
        logger.info("=" * 60)
        logger.info(f"问题 {i}: {question}")
        logger.info("=" * 60)

        # 获取答案和来源
        response = qa.ask(question)

        logger.info(f"回答:\n{response['result']}")

        logger.info(f"来源文档 ({len(response['source_documents'])} 个):")
        for j, doc in enumerate(response['source_documents'], 1):
            logger.info(f"  [{j}] {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()

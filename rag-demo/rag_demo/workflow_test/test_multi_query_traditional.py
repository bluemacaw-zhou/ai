"""
MultiQueryRetriever 传统方式测试

MultiQueryRetriever 通过生成多个查询变体来改进检索效果：
1. 接收用户的一个查询
2. 使用 LLM 生成多个类似但措辞不同的查询
3. 对所有生成的查询分别进行检索
4. 合并并去重所有检索结果

这样可以克服基于距离的检索的局限性,提高检索的召回率。

【关键区别】与 LCEL 方式 (test_multi_query_lcel.py) 的不同：
1. 实现方式：
   - test_multi_query_lcel.py: 使用 LCEL (LangChain Expression Language) 链式构建
   - test_multi_query_traditional.py: 使用传统的 MultiQueryRetriever + RetrievalQA 方式

2. 检索策略：
   - LCEL 方式: 链式调用，更灵活的控制
   - 传统方式: 使用 MultiQueryRetriever 组件封装

3. LLM 使用：
   - 两种方式都用于生成查询变体 + 生成最终答案（调用 2 次 LLM）
"""

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from ai_starter.langchain import LangChainChatZhipuAI, LangChainGLMEmbedding, LangchainChromadb
from ai_starter import get_logger
logger = get_logger(__name__)


def main():
    """测试 MultiQueryRetriever 组件（传统方式）"""
    logger.info("=" * 60)
    logger.info("MultiQueryRetriever 传统方式测试")
    logger.info("=" * 60)

    # 配置将自动从 config.yaml 读取

    # 创建 embeddings
    embeddings = LangChainGLMEmbedding()

    # 创建向量存储并添加文档
    storage = LangchainChromadb(embeddings=embeddings)

    # 清空数据库
    storage.clear_collection()

    knowledge_base = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。",
        "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。",
        "自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。",
        "计算机视觉使计算机能够从图像和视频中提取信息和理解视觉世界。",
        "强化学习是一种机器学习方法，智能体通过与环境交互并获得奖励来学习最优策略。",
        "监督学习使用标记的数据来训练模型，模型学习输入和输出之间的映射关系。",
        "无监督学习从未标记的数据中发现模式和结构，如聚类和降维。",
        "迁移学习允许模型将从一个任务学到的知识应用到另一个相关任务上。",
        "神经网络是由相互连接的节点层组成的计算模型，模仿人脑的结构。",
        "卷积神经网络（CNN）特别适合处理图像数据，能够自动学习空间层次特征。"
    ]

    logger.info(f"构建知识库：添加 {len(knowledge_base)} 个文档...")
    storage.add_texts(knowledge_base)

    # 创建基础 retriever
    base_retriever = storage.get_retriever(k=3)

    # 创建 LLM（用于生成查询变体 + 生成答案）
    # 所有配置（model, temperature, api_key, proxy, ssl等）从 config.yaml 读取
    logger.info("初始化 LangChainChatZhipuAI（配置从 config.yaml 读取）")
    llm = LangChainChatZhipuAI()

    # 创建 MultiQueryRetriever
    logger.info("初始化 MultiQueryRetriever（将使用 LLM 生成查询变体）...")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # 测试问题
    question = "什么是深度学习？"

    logger.info("=" * 60)
    logger.info(f"原始问题: {question}")
    logger.info("=" * 60)

    # 结合 RAG 使用 - 展示完整的 MultiQueryRetriever 工作流程
    # 流程：
    # 1. LLM 生成查询变体
    # 2. 每个查询变体分别检索文档
    # 3. 合并去重所有文档
    # 4. LLM 基于检索到的文档生成答案

    from langchain_classic.chains import RetrievalQA

    logger.info("[第1步: LLM 生成查询变体]")
    # 使用 MultiQueryRetriever 内部的 llm_chain 生成查询变体
    # llm_chain 会返回包含查询列表的字典
    response = multi_query_retriever.llm_chain.invoke({"question": question})
    # 解析返回的查询列表（通常在 'lines' 或 'text' 字段中）
    if isinstance(response, dict):
        if 'lines' in response:
            generated_queries = response['lines']
        elif 'text' in response:
            # 如果返回的是文本，按行分割
            generated_queries = [line.strip() for line in response['text'].strip().split('\n') if line.strip()]
        else:
            # 尝试获取第一个值
            generated_queries = list(response.values())[0] if response else []
    else:
        # 如果返回的是字符串，按行分割
        generated_queries = [line.strip() for line in str(response).strip().split('\n') if line.strip()]

    logger.info(f"从 '{question}' 生成了 {len(generated_queries)} 个查询变体:")
    for idx, query in enumerate(generated_queries, 1):
        logger.info(f"  {idx}. {query}")

    # 展示每个查询召回的文档
    logger.info("[第2步: 每个查询变体分别检索文档]")
    all_docs_per_query = []
    for idx, query in enumerate(generated_queries, 1):
        logger.info(f"查询 {idx}: {query}")
        docs = base_retriever.invoke(query)
        all_docs_per_query.append(docs)
        logger.info(f"召回 {len(docs)} 个文档:")
        for i, doc in enumerate(docs, 1):
            logger.info(f"  [{i}] {doc.page_content}")

    # 展示合并去重的过程
    logger.info("[第3步: 合并去重所有文档]")
    # 使用 MultiQueryRetriever 检索（会自动去重）
    merged_docs = multi_query_retriever.invoke(question)
    logger.info(f"合并前总文档数: {sum(len(docs) for docs in all_docs_per_query)}")
    logger.info(f"去重后文档数: {len(merged_docs)}")
    logger.info("去重后的文档:")
    for i, doc in enumerate(merged_docs, 1):
        logger.info(f"  [{i}] {doc.page_content}")

    # 创建 RAG 问答链并生成答案
    logger.info("[第4步: LLM 基于文档生成答案]")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=multi_query_retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})
    logger.info(f"问题: {question}")
    logger.info(f"答案: {result['result']}")
    logger.info(f"实际使用的来源文档 ({len(result['source_documents'])} 个):")
    for i, doc in enumerate(result['source_documents'], 1):
        logger.info(f"  [{i}] {doc.page_content}")


if __name__ == "__main__":
    main()

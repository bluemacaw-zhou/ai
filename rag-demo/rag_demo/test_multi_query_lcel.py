"""
MultiQueryRetriever LCEL 方式测试

使用 LCEL (LangChain Expression Language) 链式语法实现多查询检索：
1. 生成多个查询变体
2. 并行检索所有变体
3. 合并去重结果
4. 生成最终答案

[关键区别] 与传统方式 (test_multi_query_traditional.py) 的不同：
1. 实现方式：
   - test_multi_query_traditional.py: 使用 MultiQueryRetriever + RetrievalQA (传统 API)
   - test_multi_query_lcel.py: 使用 LCEL 链式表达式实现相同功能

2. 优势：
   - 代码更清晰，链式调用易读
   - 可以自定义每个步骤的 prompt 和逻辑
   - 更容易调试和扩展
   - 更灵活的控制流程
"""

import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai_starter.llm.custom_zhipuai_llm import CustomChatZhipuAI
from ai_starter import Config, LangChainGLMEmbedding, LangchainChromadb, get_logger

logger = get_logger(__name__)

# 生成查询变体的 prompt
QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """你是一个AI助手，帮助用户生成多个不同表述的查询问题。

原始问题: {question}

请生成3个与原始问题意思相同但表述不同的查询问题。要求：
1. 保持原问题的核心意图
2. 使用不同的措辞和角度
3. 每个问题独立一行
4. 不要编号，不要解释

直接输出3个问题即可："""
)

# 生成最终答案的 prompt
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """你是一个有用的AI助手。请根据以下上下文信息回答用户的问题。

上下文信息:
{context}

问题: {question}

请基于上下文信息给出准确、详细的回答。如果上下文信息不足以回答问题，请明确说明。"""
)


def main():
    """测试使用 LCEL 实现的多查询检索"""
    logger.info("=" * 60)
    logger.info("MultiQueryRetriever LCEL 方式测试")
    logger.info("=" * 60)

    # 加载配置
    config = Config()
    api_key = config.get("api.zhipuai.key")
    model = config.get("models.llm.model", "glm-4-flash")
    verify_ssl = config.get("api.zhipuai.verify_ssl", False)
    use_proxy = config.get("api.zhipuai.use_proxy", True)
    proxy = config.get("api.zhipuai.proxy")

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

    logger.info(f"构建知识库: 添加 {len(knowledge_base)} 个文档...")
    storage.add_texts(knowledge_base)

    # 创建基础 retriever
    base_retriever = storage.get_retriever(k=3)

    # 重新设置代理环境变量（ChromaDB 初始化时清理了代理）
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
        logger.info(f"重新设置代理环境变量: {proxy}")

    # 创建 LLM
    logger.info(f"初始化 CustomChatZhipuAI (model={model}, verify_ssl={verify_ssl}, use_proxy={use_proxy})")
    llm = CustomChatZhipuAI(
        api_key=api_key,
        model=model,
        temperature=0.7,
        verify_ssl=verify_ssl,
        use_proxy=use_proxy
    )

    # 测试问题
    question = "什么是深度学习？"

    logger.info("=" * 60)
    logger.info(f"原始问题: {question}")
    logger.info("=" * 60)

    # 结合 RAG 使用 - 展示完整的 LCEL 工作流程
    # 流程：
    # 1. LLM 生成查询变体
    # 2. 每个查询变体分别检索文档
    # 3. 合并去重所有文档
    # 4. LLM 基于文档生成答案

    # ========== 第1步：生成查询变体 ==========
    logger.info("[第1步: LLM 生成查询变体]")

    # 使用 LCEL 链生成查询变体
    query_generation_chain = (
        QUERY_GENERATION_PROMPT
        | llm
        | StrOutputParser()
    )

    llm_output = query_generation_chain.invoke({"question": question})
    generated_queries = [q.strip() for q in llm_output.strip().split('\n') if q.strip()]

    logger.info(f"从 '{question}' 生成了 {len(generated_queries)} 个查询变体:")
    for idx, query in enumerate(generated_queries, 1):
        logger.info(f"  {idx}. {query}")

    # ========== 第2步：每个查询分别检索文档 ==========
    logger.info("[第2步: 每个查询变体分别检索文档]")

    all_docs_per_query = []
    for idx, query in enumerate(generated_queries, 1):
        logger.info(f"查询 {idx}: {query}")
        docs = base_retriever.invoke(query)
        all_docs_per_query.append(docs)
        logger.info(f"召回 {len(docs)} 个文档:")
        for i, doc in enumerate(docs, 1):
            logger.info(f"  [{i}] {doc.page_content}")

    # ========== 第3步：合并去重 ==========
    logger.info("[第3步: 合并去重所有文档]")

    # 收集所有文档
    all_docs = []
    for docs in all_docs_per_query:
        all_docs.extend(docs)

    # 去重（基于文档内容）
    unique_docs = []
    seen_contents = set()
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    logger.info(f"合并前总文档数: {sum(len(docs) for docs in all_docs_per_query)}")
    logger.info(f"去重后文档数: {len(unique_docs)}")
    logger.info("去重后的文档:")
    for i, doc in enumerate(unique_docs, 1):
        logger.info(f"  [{i}] {doc.page_content}")

    # ========== 第4步：生成答案 ==========
    logger.info("[第4步: LLM 基于文档生成答案]")

    # 使用 LCEL 链生成答案
    def format_docs(docs: List) -> str:
        """格式化文档为上下文字符串"""
        return "\n\n".join([f"[文档{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    qa_chain = (
        ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    context = format_docs(unique_docs)
    result = qa_chain.invoke({"question": question, "context": context})

    logger.info(f"问题: {question}")
    logger.info(f"答案: {result}")
    logger.info(f"实际使用的来源文档 ({len(unique_docs)} 个):")
    for i, doc in enumerate(unique_docs, 1):
        logger.info(f"  [{i}] {doc.page_content}")


if __name__ == "__main__":
    main()

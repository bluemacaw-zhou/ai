"""
LangChain RAG 问答检索组件
"""

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ai_starter.langchain import LangChainChatZhipuAI
from ai_starter import get_logger

logger = get_logger(__name__)

# RAG 提示模板
RAG_PROMPT_TEMPLATE = """你是一个有用的AI助手。请根据以下上下文信息回答用户的问题。

上下文信息:
{context}

问题: {question}

请基于上下文信息给出准确、详细的回答。如果上下文信息不足以回答问题，请明确说明。
"""


ANSWER_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


class LangchainQARetriever:
    """LangChain RAG 问答检索器（LCEL 方式）"""

    def __init__(self, retriever):
        """
        初始化问答检索器

        Args:
            retriever: LangChain Retriever
        """
        self.retriever = retriever

        # 初始化 LangChain LLM（配置从 config.yaml 读取）
        logger.info("初始化 LangChainChatZhipuAI（配置从 config.yaml 读取）")
        self.llm = LangChainChatZhipuAI()

        # 创建 RAG 链 (使用 LCEL)
        self.qa_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"LangchainQARetriever initialized (temperature={self.llm._zhipu.temperature})")

    def _format_docs(self, docs):
        """格式化文档为上下文字符串"""
        return "\n\n".join([f"[文档{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    def ask(self, question: str) -> Dict:
        """
        提问并获取答案

        Args:
            question: 问题

        Returns:
            Dict: 包含答案和来源文档的字典
        """
        logger.info(f"提问: {question}")

        # 获取相关文档
        source_documents = self.retriever.invoke(question)
        logger.info(f"检索到 {len(source_documents)} 个相关文档")

        # 生成答案
        result = self.qa_chain.invoke(question)

        return {
            "result": result,
            "source_documents": source_documents
        }

    def ask_simple(self, question: str) -> str:
        """提问并只返回答案"""
        response = self.ask(question)
        return response["result"]

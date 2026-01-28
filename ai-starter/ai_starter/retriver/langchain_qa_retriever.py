"""
LangChain RAG 问答检索组件
"""

import os
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ai_starter.llm.custom_zhipuai_llm import CustomChatZhipuAI
from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger

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

    def __init__(self, retriever, temperature: float = 0.7):
        """
        初始化问答检索器

        Args:
            retriever: LangChain Retriever
            temperature: LLM 温度参数
        """
        self.retriever = retriever

        # 清理可能存在的代理环境变量（保留以兼容旧配置）
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy']:
            os.environ.pop(key, None)

        # 从配置读取 LLM 配置
        config = Config()
        api_key = config.get("api.zhipuai.key")
        model = config.get("models.llm.model", "glm-4-flash")
        verify_ssl = config.get("api.zhipuai.verify_ssl", False)
        use_proxy = config.get("api.zhipuai.use_proxy", True)

        if not api_key:
            raise ValueError("缺少 ZhipuAI API Key，请在配置文件中配置 api.zhipuai.key")

        # 初始化自定义 LLM（支持代理环境和 SSL 验证配置）
        logger.info(f"初始化 CustomChatZhipuAI (model={model}, verify_ssl={verify_ssl}, use_proxy={use_proxy})")

        self.llm = CustomChatZhipuAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            verify_ssl=verify_ssl,
            use_proxy=use_proxy
        )

        # 创建 RAG 链 (使用 LCEL)
        self.qa_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"LangchainQARetriever initialized (model={model}, temperature={temperature})")

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

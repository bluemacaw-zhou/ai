"""
完整的 RAG 流程测试

演示完整的 RAG（检索增强生成）流程：
1. PDF 分割成 chunks
2. 文本向量化（LangChain Embeddings）
3. 存储到向量数据库（LangChain VectorStore）
4. 问答检索（LangChain LCEL）
"""

from pathlib import Path
from ai_starter import Config, get_logger
from ai_starter.langchain import PDFChunker, LangChainGLMEmbedding, LangchainChromadb
from ..retriever.langchain_qa_retriever import LangchainQARetriever

logger = get_logger(__name__)


class RAGPipeline:
    """RAG 完整流程管道"""

    def __init__(self):
        """初始化 RAG 流程"""
        config = Config()
        self.pdf_path = Path(__file__).parent.parent / config.get("rag.pdf_path", "项目经理资格考试题库.pdf")
        self.chunk_size = config.get("rag.chunk_size", 500)
        self.chunk_overlap = config.get("rag.chunk_overlap", 50)
        self.top_k = config.get("rag.top_k", 3)

        self.qa_retriever = None

        logger.info("RAG Pipeline 初始化完成")

    def build_knowledge_base(self):
        """
        构建知识库（Step 1+2+3）

        只需执行一次，数据会持久化到 ChromaDB
        """
        if not self.pdf_path.exists():
            logger.error(f"PDF 文件不存在: {self.pdf_path}")
            raise FileNotFoundError(f"请将 PDF 文件放在: {self.pdf_path}")

        logger.info("=" * 60)
        logger.info("开始构建知识库")
        logger.info("=" * 60)

        # 初始化 embeddings 和 storage
        embeddings = LangChainGLMEmbedding()
        storage = LangchainChromadb(embeddings=embeddings)

        # Step 0: 清空现有数据 (避免与前面步骤的数据混淆)
        logger.info("\n[Step 0] 清空现有数据")
        storage.clear_collection()
        logger.info("✓ 数据清空完成")

        # Step 1: 加载并分割 PDF
        logger.info("\n[Step 1] PDF 文本分割")
        chunker = PDFChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.load_and_split(str(self.pdf_path))
        logger.info(f"✓ 分割完成: {len(chunks)} 个文本块")

        # Step 2+3: 向量化并存储到向量数据库
        logger.info("\n[Step 2+3] 文本向量化并存储到向量数据库")
        storage.add_texts(chunks)
        logger.info(f"✓ 存储完成: {len(chunks)} 个文档")

        # Step 4: 初始化问答检索器
        logger.info("\n[Step 4] 初始化问答检索器")
        retriever = storage.get_retriever(k=self.top_k)
        self.qa_retriever = LangchainQARetriever(retriever)
        logger.info("✓ 问答系统初始化完成")

        logger.info("\n" + "=" * 60)
        logger.info("知识库构建完成！")
        logger.info("=" * 60)

    def ask(self, question: str) -> dict:
        """
        提问并获取答案

        Args:
            question: 问题

        Returns:
            dict: 包含答案和来源文档
        """
        if self.qa_retriever is None:
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        response = self.qa_retriever.ask(question)

        logger.info(f"回答:\n{response['result']}")

        # if response['source_documents']:
        #     logger.info(f"来源文档 ({len(response['source_documents'])} 个):")
        #     for i, doc in enumerate(response['source_documents'], 1):
        #         content = doc.page_content.strip().replace('\n', ' ')
        #         preview = content[:150] + "..." if len(content) > 150 else content
        #         logger.info(f"  [{i}] {preview}")

        return response


def main():
    """测试主函数"""
    from ai_starter import Config

    config = Config()

    # 是否构建知识库（由配置控制）
    build_kb = config.get("rag.build_knowledge_base", True)

    # 创建 RAG 流程
    pipeline = RAGPipeline()

    # 构建知识库（只需执行一次）
    if build_kb:
        pipeline.build_knowledge_base()
    else:
        # 直接初始化问答检索器（知识库已存在）
        logger.info("从现有知识库加载")
        embeddings = LangChainGLMEmbedding()
        storage = LangchainChromadb(embeddings=embeddings)
        retriever = storage.get_retriever(k=pipeline.top_k)
        pipeline.qa_retriever = LangchainQARetriever(retriever)
        logger.info("知识库加载完成")

    # 定义问题列表
    questions = [
        "软件质量是什么",
        "万得的项目流程有哪些",
        "项目迭代的长度是多久",
    ]

    # 依次提问
    logger.info("开始问答测试")

    for question in questions:
        pipeline.ask(question)


if __name__ == "__main__":
    main()

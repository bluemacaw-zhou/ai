"""
PDF 文本分割组件
"""

from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ai_starter.core.log.logging_utils import get_logger

logger = get_logger(__name__)


class PDFChunker:
    """PDF 文档分块处理器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化 PDF 分块器

        Args:
            chunk_size: 每个文本块的大小（字符数）
            chunk_overlap: 文本块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        logger.info(f"PDFChunker initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def load_and_split(self, pdf_path: str) -> List[str]:
        """
        加载 PDF 文件并分割成文本块

        Args:
            pdf_path: PDF 文件路径

        Returns:
            List[str]: 文本块列表
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        logger.info(f"开始加载 PDF: {pdf_path}")

        # 使用 LangChain 的 PyPDFLoader 加载 PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info(f"PDF 加载完成，共 {len(documents)} 页")

        # 分割文档
        texts = self.text_splitter.split_documents(documents)

        # 提取文本内容
        chunks = [doc.page_content for doc in texts]

        logger.info(f"文档分割完成，共 {len(chunks)} 个文本块")

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        分割纯文本

        Args:
            text: 输入文本

        Returns:
            List[str]: 文本块列表
        """
        logger.debug(f"分割文本，长度: {len(text)} 字符")

        # 分割文本
        texts = self.text_splitter.split_text(text)

        logger.debug(f"文本分割完成，共 {len(texts)} 个文本块")

        return texts

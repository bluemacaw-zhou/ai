"""
Step 1: PDF 文本分割测试
"""

from pathlib import Path
from ai_starter import Config, get_logger
from ai_starter.langchain import PDFChunker

logger = get_logger(__name__)


def main():
    """测试 PDFChunker 组件"""
    logger.info("=" * 60)
    logger.info("Step 1: PDF 文本分割测试")
    logger.info("=" * 60)

    # 从配置读取参数
    config = Config()
    pdf_path = config.get("rag.pdf_path", "项目经理资格考试题库.pdf")
    chunk_size = config.get("rag.chunk_size", 500)
    chunk_overlap = config.get("rag.chunk_overlap", 50)

    # 如果是相对路径，转换为绝对路径
    if not Path(pdf_path).is_absolute():
        pdf_path = Path(__file__).parent.parent / pdf_path

    logger.info(f"PDF 文件路径: {pdf_path}")
    logger.info(f"chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    if not Path(pdf_path).exists():
        logger.error(f"PDF 文件不存在: {pdf_path}")
        logger.error("请在 config.yaml 中配置正确的 PDF 路径，或将 PDF 文件放在正确位置")
        return

    # 创建分块器
    chunker = PDFChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 加载并分割 PDF
    logger.info("开始加载和分割 PDF...")
    chunks = chunker.load_and_split(str(pdf_path))

    logger.info("=" * 60)
    logger.info(f"分割结果: {len(chunks)} 个文本块")
    logger.info("=" * 60)

    # 显示前 3 个文本块
    logger.info("前 3 个文本块:")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"--- Chunk {i+1} ({len(chunk)} 字符) ---")
        logger.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    # 统计信息
    avg_size = sum(len(c) for c in chunks) // len(chunks)
    min_size = min(len(c) for c in chunks)
    max_size = max(len(c) for c in chunks)
    logger.info("=" * 60)
    logger.info("统计信息:")
    logger.info(f"  总文本块数: {len(chunks)}")
    logger.info(f"  平均块大小: {avg_size} 字符")
    logger.info(f"  最小块: {min_size} 字符")
    logger.info(f"  最大块: {max_size} 字符")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

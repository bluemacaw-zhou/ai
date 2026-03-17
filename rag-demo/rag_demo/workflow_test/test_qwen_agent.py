"""
Qwen-Agent RAG 流程测试

演示使用 Qwen-Agent 实现 RAG（检索增强生成）流程：
完全委托 Qwen-Agent 处理文件读取、分块、向量化和检索

[关键区别] 与 test_rag_pipeline 的不同：
1. 实现方式：
   - test_rag_pipeline: 手动构建向量库 (ai-starter: PDF分割 -> 向量化 -> ChromaDB -> LangChain QA)
   - test_qwen_agent: Qwen-Agent 内置文件处理能力（完全自动化）

2. 优势：
   - 代码更简洁（无需 ai-starter 向量库逻辑）
   - Qwen-Agent 自动处理文档分块和检索
   - 支持多种文件格式（PDF、TXT、DOCX 等）

[智谱 AI 集成]：
- 使用 QwenAgentChatZhipuAI（Qwen-Agent 专用适配器）
- 所有配置从 config.yaml 读取（内聚设计）
- 自动从配置文件读取代理和 SSL 设置
"""

import shutil
from pathlib import Path
from qwen_agent.agents import Assistant
from ai_starter import Config, get_logger
from ai_starter.qwen_agent import QwenAgentChatZhipuAI

logger = get_logger(__name__)


class QwenAgentPipeline:
    """使用 Qwen-Agent 的 RAG 流程管道"""

    def __init__(self):
        """
        初始化 Qwen-Agent RAG 流程
        """
        config = Config()

        # PDF 文件路径
        self.pdf_path = Path(__file__).parent.parent / config.get("rag.pdf_path", "项目经理资格考试题库.pdf")

        # 直接传入 QwenAgentChatZhipuAI 对象
        # 所有配置从 config.yaml 读取
        logger.info("使用 QwenAgentChatZhipuAI 对象（配置从 config.yaml 读取）")
        self.llm = QwenAgentChatZhipuAI()

        self.agent = None

        logger.info("Qwen-Agent Pipeline 初始化完成")

    def build_knowledge_base(self):
        """
        构建知识库

        完全委托 Qwen-Agent 处理：
        - PDF 文件读取和解析
        - 文档分块（chunking）
        - 文本向量化（embedding）
        - 向量存储和检索
        """
        # 清空 Qwen-Agent 本地工作空间
        workspace_dir = Path(__file__).parent.parent.parent / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
            logger.info(f"已清空本地工作空间: {workspace_dir}")

        if not self.pdf_path.exists():
            logger.error(f"PDF 文件不存在: {self.pdf_path}")
            raise FileNotFoundError(f"请将 PDF 文件放在: {self.pdf_path}")

        logger.info("=" * 60)
        logger.info("初始化 Qwen-Agent")
        logger.info("=" * 60)

        # 创建 Assistant（直接传入 PDF 文件）
        # Qwen-Agent 会自动：
        # 1. 解析 PDF
        # 2. 文档分块
        # 3. 向量化存储
        # 4. 在回答时自动检索相关内容
        self.agent = Assistant(
            llm=self.llm,
            system_message='你是一个专业的问答助手。请根据提供的文档内容回答用户的问题。',
            files=[str(self.pdf_path)]  # Qwen-Agent 自动处理
        )

        logger.info(f"✓ 已加载 PDF 文件: {self.pdf_path.name}")
        logger.info("✓ Qwen-Agent 已自动完成文档处理和向量化")
        logger.info("=" * 60)
        logger.info("知识库构建完成!")
        logger.info("=" * 60)

    def ask(self, question: str) -> dict:
        """
        提问并获取答案

        Args:
            question: 问题

        Returns:
            dict: 包含答案的字典
        """
        if self.agent is None:
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        logger.info(f"问题: {question}")

        # 调用 Agent（会自动检索相关文档并生成答案）
        messages = [{'role': 'user', 'content': question}]
        response = []

        for response_chunk in self.agent.run(messages=messages):
            response.append(response_chunk)

        # 提取最终答案
        if response:
            final_response = response[-1]
            if isinstance(final_response, list) and len(final_response) > 0:
                answer = final_response[0].get('content', '未获取到答案')
            else:
                answer = '未获取到答案'
        else:
            answer = '未获取到答案'

        logger.info(f"回答: {answer}")

        return {
            'result': answer,
            'question': question
        }


def main():
    """测试主函数"""
    # 创建 Qwen-Agent RAG 流程
    pipeline = QwenAgentPipeline()

    # 构建知识库（加载 PDF）
    pipeline.build_knowledge_base()

    # 定义问题列表
    questions = [
        "软件质量是什么",
        "万得的项目流程有哪些",
        "项目迭代的长度是多久",
    ]

    # 依次提问
    logger.info("开始问答测试")

    for question in questions:
        logger.info("=" * 60)
        pipeline.ask(question)
        logger.info("")


if __name__ == "__main__":
    main()

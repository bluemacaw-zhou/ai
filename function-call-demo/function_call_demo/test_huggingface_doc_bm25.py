"""Load Hugging Face docs and answer a question with a smolagents retriever tool."""

import os
from pathlib import Path
from typing import Any

from ai_starter import get_logger
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from function_call_demo.components.huggingface_http_client_adapter import (
    HuggingFaceHttpClientAdapter,
)

logger = get_logger(__name__)

# 项目根目录是 function-call-demo，而不是整个 D:\workspace\ai。
# 这个路径用于定位 config.yaml 和 Hugging Face 本地缓存目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"


class HuggingFaceDocBm25Demo:
    """从 Hugging Face 文档数据集构建本地检索工具，并交给 smolagents 使用。"""

    # 数据集来自 Hugging Face Hub，包含多类 Hugging Face 文档。
    _dataset_name = "m-ric/huggingface_doc"

    # 这个 demo 只关注 Transformers 文档，避免其它 Hugging Face 产品文档干扰检索结果。
    _source_prefix = "huggingface/transformers"

    # 这个问题需要比较训练中的 forward pass 和 backward pass。
    # 它比简单事实问答更适合展示 agent 先拆分问题、再多次检索、最后汇总答案。
    _agent_question = (
        "For a transformers model training, which is slower, the forward or the backward pass?"
    )

    # InferenceClientModel 会调用 Hugging Face Inference Providers。
    # 运行前需要 HF_TOKEN 具备 Make calls to Inference Providers 权限。
    _agent_model_id = "meta-llama/Llama-3.3-70B-Instruct"

    # 这里给 agent 足够步数完成：检索 forward、检索 backward、比较、输出最终答案。
    _agent_max_steps = 6

    # 故意设置为 1。单次检索返回的信息较少，agent 才更有动力按 instructions 多检索几次。
    _retriever_top_k = 1

    # smolagents 的 CodeAgent 不是天然“必须多轮检索”。
    # 这里通过 instructions 约束它：最终回答前至少用两个不同 query 调用 retriever。
    _agent_instructions = (
        "You answer questions by using the retriever tool. Before giving the final answer, "
        "you must call the retriever at least twice with different affirmative search queries: "
        "one query focused on the forward pass during training, and one query focused on the "
        "backward pass during training. Compare the retrieved evidence, then answer briefly."
    )

    def __init__(self) -> None:
        # HuggingFaceHttpClientAdapter 负责把 ai-starter 的代理、SSL 和 timeout 配置接到 HF SDK。
        # demo 主流程不直接关心代理细节，只拿到 datasets.DownloadConfig。
        self._http_adapter = HuggingFaceHttpClientAdapter(
            project_root=PROJECT_ROOT,
            cache_dir=HF_CACHE_DIR,
        )
        self._download_config = self._http_adapter.configure()

    def close(self) -> None:
        # huggingface_hub 使用全局共享 httpx.Client，结束时显式关闭，避免连接资源悬挂。
        self._http_adapter.close()

    def tokenizer_question_test(self) -> None:
        # 数据准备由 demo 显式完成；agent 只负责决定如何检索、检索几次、如何总结。
        knowledge_base = self._load_knowledge_base()
        source_docs = self._build_source_docs(knowledge_base)
        docs_processed = self._split_documents(source_docs)

        logger.info("Hugging Face 文档数据准备完成")
        logger.info(f"数据集: {self._dataset_name}")
        logger.info(f"缓存目录: {HF_CACHE_DIR}")
        logger.info(f"过滤来源: {self._source_prefix}")
        logger.info(f"原始文档数: {len(source_docs)}")
        logger.info(f"切分后 chunk 数: {len(docs_processed)}")

        self._run_agent(docs_processed)

    def _run_agent(self, docs_processed: list[Document]) -> None:
        # smolagents 是这个 demo 的可选主角，延迟导入能让数据准备部分的依赖错误更清晰。
        from smolagents import CodeAgent, InferenceClientModel
        from smolagents.monitoring import LogLevel

        from function_call_demo.components.retriever_tool import RetrieverTool

        hf_token = self._get_hf_token()

        logger.info("=" * 80)
        logger.info("smolagents CodeAgent 检索问答测试")
        logger.info(f"模型: {self._agent_model_id}")
        logger.info(f"问题: {self._agent_question}")
        logger.info(f"Agent 最大推理步数: {self._agent_max_steps}")
        logger.info(f"每次 retriever 工具调用返回 TopK: {self._retriever_top_k}")
        logger.info("强制策略: 最终回答前至少用不同 query 调用 retriever 两次")
        logger.info(
            "HF_TOKEN 状态: "
            f"present={hf_token is not None}, "
            f"length={len(hf_token) if hf_token else 0}, "
            f"starts_with_hf={hf_token.startswith('hf_') if hf_token else False}"
        )

        if not hf_token:
            logger.error("未检测到 HF_TOKEN，请在 PyCharm Run Configuration 的环境变量中配置。")
            logger.info("=" * 80)
            return

        # RetrieverTool 内部仍然是 BM25，本地执行；CodeAgent 只把它当作一个可调用工具。
        retriever_tool = RetrieverTool(docs_processed, top_k=self._retriever_top_k)

        # verbosity_level=OFF 是为了避免 smolagents 自己往控制台画 step 面板。
        # 工具调用由 RetrieverTool.forward() 里的 ai-starter logger 记录，更方便和项目日志统一。
        agent = CodeAgent(
            tools=[retriever_tool],
            model=InferenceClientModel(model_id=self._agent_model_id, token=hf_token),
            instructions=self._agent_instructions,
            max_steps=self._agent_max_steps,
            verbosity_level=LogLevel.OFF,
        )

        try:
            agent_output = agent.run(self._agent_question)
        except Exception as exc:
            logger.error("Agent 调用失败，请检查 Hugging Face token 是否具备 Inference Providers 权限。")
            logger.error(f"失败原因: {exc}")
            logger.info("=" * 80)
            return

        logger.info("Agent 最终回答:")
        logger.info(agent_output)
        logger.info("=" * 80)

    @staticmethod
    def _get_hf_token() -> str | None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            return None

        # 允许用户在 PyCharm 环境变量里误加引号；规范化后再交给 HF SDK。
        # 同时写回两个常见变量名，保证 huggingface_hub 和 smolagents 都能读到同一个值。
        token = token.strip().strip('"').strip("'")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    def _load_knowledge_base(self) -> Any:
        import datasets

        # datasets.load_dataset 会优先使用本地 cache；首次运行时才会访问 Hugging Face Hub。
        # download_config 里已经包含代理、SSL、重试、cache_dir 和 token 设置。
        knowledge_base = datasets.load_dataset(
            self._dataset_name,
            split="train",
            download_config=self._download_config,
            cache_dir=str(HF_CACHE_DIR),
        )
        return knowledge_base.filter(lambda row: row["source"].startswith(self._source_prefix))

    @staticmethod
    def _build_source_docs(knowledge_base: Any) -> list[Document]:
        # LangChain 的 BM25Retriever 接收 Document 列表。
        # page_content 是参与检索的正文，metadata 用于保留来源信息，便于后续排查命中结果。
        return [
            Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
            for doc in knowledge_base
        ]

    @staticmethod
    def _split_documents(source_docs: list[Document]) -> list[Document]:
        # BM25 在 chunk 级别检索。chunk 太大召回不精细，太小又容易丢上下文。
        # add_start_index=True 会保留 chunk 在原文中的起始位置，方便需要时定位原始文档片段。
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return text_splitter.split_documents(source_docs)


def main() -> None:
    demo = HuggingFaceDocBm25Demo()
    try:
        demo.tokenizer_question_test()
    finally:
        demo.close()


if __name__ == "__main__":
    main()

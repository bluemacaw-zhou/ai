"""smolagents tool for retrieving local documentation chunks."""

from ai_starter import get_logger
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

logger = get_logger(__name__)


class RetrieverTool(Tool):
    """把本地 BM25 检索器包装成 smolagents 可调用的 Tool。"""

    # smolagents 会把 name/description/inputs 暴露给模型。
    # 模型不是直接知道 Python 对象怎么用，而是根据这些字段决定何时调用工具。
    name = "retriever"
    description = (
        "Uses keyword search to retrieve the parts of transformers documentation "
        "that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your "
                "target documents. Use the affirmative form rather than a question."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs: list[Document], *, top_k: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        # BM25Retriever 是纯本地检索，不需要远端 embedding 服务或向量库。
        # 这里把已经切分好的文档 chunk 建成检索器，agent 每次调用 forward 时复用它。
        self.retriever = BM25Retriever.from_documents(docs, k=top_k)
        self.top_k = top_k

    def forward(self, query: str) -> str:
        # forward 是 smolagents 真正执行工具时调用的方法。
        # query 来自模型生成，因此这里保留类型断言，避免模型传错结构后静默失败。
        assert isinstance(query, str), "Your search query must be a string"

        logger.info(f"Agent 调用 retriever 工具，query={query!r}, top_k={self.top_k}")
        docs = self.retriever.invoke(query)
        logger.info(f"retriever 返回文档数: {len(docs)}")

        # 返回字符串而不是 Document 对象，是因为工具结果会回到 LLM 上下文里。
        # 分隔符能帮助模型区分不同命中文档，减少把多个片段混成一段的概率。
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {index} =====\n{doc.page_content}"
                for index, doc in enumerate(docs, start=1)
            ]
        )

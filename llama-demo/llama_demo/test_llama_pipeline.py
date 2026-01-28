"""
LlamaIndex RAG Pipeline 测试 - 使用智谱AI

演示完整的 RAG 流程：
1. 使用智谱AI LLM (glm-4-flash)
2. 使用智谱AI Embedding (embedding-3)
3. 使用 Qdrant 向量数据库
4. 支持代理配置
"""

import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from ai_starter import HttpClientFactory
from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger

logger = get_logger(__name__)


class LlamaRAGPipeline:
    """LlamaIndex RAG 完整流程管道"""

    # 配置常量
    COLLECTION_NAME = "llama_index_demo"
    EMBEDDING_DIM = 1024  # 智谱 embedding-2 的维度

    def __init__(self):
        """初始化 RAG Pipeline"""
        config = Config()

        # 读取配置参数
        self.data_path = Path(__file__).parent.parent / config.get("llama.data_path", "./data")
        self.chunk_size = config.get("llama.chunk_size", 512)
        self.chunk_overlap = config.get("llama.chunk_overlap", 200)
        self.similarity_top_k = config.get("llama.similarity_top_k", 5)
        self.rerank_top_n = config.get("llama.rerank_top_n", 2)
        self.similarity_cutoff = config.get("llama.similarity_cutoff", 0.6)
        self.num_queries = config.get("llama.num_queries", 3)

        # 智谱AI 配置
        self.api_key = config.get("api.zhipuai.key")
        self.llm_model = config.get("models.llm.model", "glm-4-flash")
        self.embedding_model = config.get("models.embedding.model", "embedding-3")
        self.verify_ssl = config.get("api.zhipuai.verify_ssl", False)

        # Qdrant 配置（本地存储）
        self.qdrant_path = config.get("qdrant.path", "./qdrant_db")

        # 内部状态
        self.qdrant_client = None
        self.index = None
        self.chat_engine = None

        logger.info("LlamaRAGPipeline 初始化开始")

        # 1. 配置智谱AI模型
        self.setup_models()

        # 2. 配置 Qdrant 客户端
        self.setup_qdrant_client()

        # 3. 是否重建知识库
        build_kb = config.get("llama.build_knowledge_base", True)
        if build_kb:
            self.build_knowledge_base()
        else:
            self.load_existing_index()

        logger.info("LlamaRAGPipeline 初始化完成")

    def setup_models(self):
        """配置智谱AI的LLM和Embedding模型（使用 OpenAILike）"""
        if not self.api_key:
            raise ValueError("缺少 ZhipuAI API Key，请在配置文件中配置 api.zhipuai.key")

        http_client = HttpClientFactory.create()

        logger.info(f"初始化智谱AI LLM: {self.llm_model}")
        logger.info(f"初始化智谱AI Embedding: {self.embedding_model}")

        # 1. 配置 LLM（使用 OpenAILike）
        # OpenAILike 是 LlamaIndex 提供的通用类，兼容 OpenAI 风格的 API
        Settings.llm = OpenAILike(
            model=self.llm_model,
            api_base="https://open.bigmodel.cn/api/paas/v4/",
            api_key=self.api_key,
            is_chat_model=True,
            http_client=http_client
        )

        # 2. 配置 Embedding（使用 OpenAIEmbeddingLike）
        # OpenAIEmbeddingLike 支持任意模型名称
        Settings.embed_model = OpenAILikeEmbedding(
            model_name=self.embedding_model,  # 支持 embedding-2, embedding-3
            api_key=self.api_key,
            api_base="https://open.bigmodel.cn/api/paas/v4/",
            http_client=http_client,
        )

        logger.info("✓ 智谱AI 模型配置完成（使用 OpenAILike）")

        # 3. 配置文档处理的 Ingestion Pipeline
        Settings.transformations = [
            SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        ]

    def setup_qdrant_client(self):
        """配置 Qdrant 客户端（本地文件系统存储）"""
        qdrant_path = Path(__file__).parent.parent / self.qdrant_path
        logger.info(f"使用本地 Qdrant 存储: {qdrant_path}")

        self.qdrant_client = QdrantClient(path=str(qdrant_path))

        logger.info("✓ Qdrant 客户端初始化成功")

    def build_knowledge_base(self):
        """
        构建知识库

        加载文档 → 向量化 → 存储到 Qdrant
        """
        if not self.data_path.exists():
            logger.error(f"数据目录不存在: {self.data_path}")
            raise FileNotFoundError(f"请将文档文件放在: {self.data_path}")

        logger.info("=" * 60)
        logger.info("开始构建知识库")
        logger.info("=" * 60)

        # Step 1: 加载本地文档
        logger.info(f"[Step 1] 从目录加载文档: {self.data_path}")
        documents = SimpleDirectoryReader(str(self.data_path), required_exts=[".pdf"]).load_data()
        logger.info(f"✓ 加载完成: {len(documents)} 个文档")

        # Step 2: 重建 collection
        if self.qdrant_client.collection_exists(collection_name=self.COLLECTION_NAME):
            logger.info(f"[Step 2] 删除现有 collection: {self.COLLECTION_NAME}")
            self.qdrant_client.delete_collection(collection_name=self.COLLECTION_NAME)

        logger.info(f"[Step 3] 创建新 collection: {self.COLLECTION_NAME}")
        self.qdrant_client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=self.EMBEDDING_DIM, distance=Distance.COSINE)
        )
        logger.info("✓ Collection 创建完成")

        # Step 3: 创建 Vector Store
        logger.info("[Step 4] 创建 Vector Store")
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.COLLECTION_NAME
        )

        # Step 4: 创建索引并存储文档
        logger.info("[Step 5] 向量化并存储文档到 Vector Store")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        logger.info(f"✓ 存储完成: {len(documents)} 个文档已向量化并索引")

        # Step 5: 创建对话引擎
        self._create_chat_engine()

        logger.info("=" * 60)
        logger.info("知识库构建完成！")
        logger.info("=" * 60)

    def load_existing_index(self):
        """从现有 collection 加载索引"""
        logger.info(f"从现有 collection 加载: {self.COLLECTION_NAME}")

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.COLLECTION_NAME
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store)

        logger.info("✓ 索引加载完成")

        # 创建对话引擎
        self._create_chat_engine()

    def _create_chat_engine(self):
        """创建对话引擎（内部方法）"""
        logger.info("创建 RAG Fusion 查询引擎")

        # 1. 定义检索后排序模型
        reranker = LLMRerank(top_n=self.rerank_top_n)
        # 最终打分低于阈值的文档被过滤掉
        sp = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)

        # 2. 定义 RAG Fusion 检索器
        # 自定义中文 prompt，确保生成的查询变体是中文
        # 注意：源码调用 format() 时只传 num_queries 和 query 两个参数
        # query_gen_prompt = (
        #     "请基于以下原始问题，生成 {num_queries} 个不同角度的中文查询变体，以提高检索效果。\n"
        #     "原始问题: {query}\n"
        #     "请生成 {num_queries} 个中文查询变体，每行一个："
        # )

        fusion_retriever = QueryFusionRetriever(
            [self.index.as_retriever()],
            similarity_top_k=self.similarity_top_k,  # 检索召回 top k 结果
            num_queries=self.num_queries,  # 生成 query 数
            use_async=False,
            # query_gen_prompt=query_gen_prompt,
        )

        # 3. 构建单轮 query engine
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            node_postprocessors=[reranker],
            response_synthesizer=get_response_synthesizer(
                response_mode=ResponseMode.REFINE,
            )
        )

        logger.info("✓ 查询引擎创建完成")

        # 4. 创建对话引擎
        logger.info("创建对话引擎")
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            # condense_question_prompt=""  # 可以自定义 chat message prompt 模板
        )

        logger.info("✓ 对话引擎创建完成")

    def chat(self, message: str) -> str:
        """
        进行对话

        Args:
            message: 用户消息

        Returns:
            AI 回复
        """
        if self.chat_engine is None:
            raise ValueError("请先调用 build_knowledge_base() 或 load_existing_index() 初始化对话引擎")

        response = self.chat_engine.chat(message)
        return response.response

    def reset_chat(self):
        """重置对话历史"""
        if self.chat_engine:
            self.chat_engine.reset()
            logger.info("对话历史已重置")


def main():
    """测试主函数"""
    # 创建 RAG 流程（初始化时自动完成所有配置）
    pipeline = LlamaRAGPipeline()

    # 测试多轮对话
    logger.info("=" * 60)
    logger.info("开始测试多轮对话")
    logger.info("=" * 60)

    # 预定义的测试问题
    test_questions = [
        # "软件质量是什么",
        "万得的项目流程有哪些",
        # "项目迭代的长度是多久"
    ]

    for question in test_questions:
        logger.info(f"User: {question}")
        response = pipeline.chat(question)
        logger.info(f"AI: {response}")
        logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info("对话结束")
    logger.info("=" * 60)

    # 显式关闭 Qdrant 客户端，避免 Python 关闭时的异常
    if pipeline.qdrant_client:
        pipeline.qdrant_client.close()


if __name__ == "__main__":
    main()

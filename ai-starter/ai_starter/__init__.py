"""
AI Starter - 共享工具包

提供通用的工具函数和AI相关组件

使用方式:
    # 仅使用核心功能（无可选依赖）
    from ai_starter.core import Config, get_logger, HttpClientFactory

    # 使用完整功能（需要安装对应依赖）
    from ai_starter import ChromaDB, GLMEmbedding, CustomChatZhipuAI
"""

__version__ = "0.1.0.dev"

# 核心模块（无可选依赖）
from ai_starter.core.log.logging_utils import (
    say_hello,
    get_logger,
    setup_logging_from_config,
    generate_trace_id,
    get_trace_id,
    set_trace_id,
    clear_trace_id,
    trace_context,
    with_trace,
)
from ai_starter.core.config.config import Config, load_config
from ai_starter.http_client.http_client_factory import HttpClientFactory

# ZhipuAI integration (optional dependency)
try:
    from ai_starter.zhipu import (
        ZhipuLLMFactory,
        ZhipuEmbeddingFactory,
        ZhipuGlobalSettings,
    )
except Exception:
    ZhipuLLMFactory = None
    ZhipuEmbeddingFactory = None
    ZhipuGlobalSettings = None

# 可选依赖模块（使用 try-except 避免导入错误）
try:
    from ai_starter.chromadb.chromadb_client import ChromaDB
except Exception:
    ChromaDB = None

try:
    from ai_starter.embedding.embedding_interface import EmbeddingInterface
    from ai_starter.embedding.embedding_glm import GLMEmbedding
    from ai_starter.embedding.embedding_openai import OpenAIEmbedding
    from ai_starter.embedding.langchain_glm_embedding import LangChainGLMEmbedding
    from ai_starter.embedding.langchain_embedding_interface import LangChainEmbeddingInterface
except Exception:
    # 捕获所有异常：ImportError（缺少依赖）、TypeError（类定义错误）等
    EmbeddingInterface = None
    GLMEmbedding = None
    OpenAIEmbedding = None
    LangChainGLMEmbedding = None
    LangChainEmbeddingInterface = None

try:
    from ai_starter.llm.langchain_interface import LangChainChatModel
    from ai_starter.llm.custom_zhipuai_llm import CustomChatZhipuAI
    from ai_starter.llm.qwen_agent_interface import QwenAgentLLM
except Exception:
    LangChainChatModel = None
    CustomChatZhipuAI = None
    QwenAgentLLM = None

try:
    from ai_starter.pdf.pdf_chunker import PDFChunker
except Exception:
    PDFChunker = None

try:
    from ai_starter.chromadb.langchain_chromadb import LangchainChromadb
except Exception:
    LangchainChromadb = None

try:
    from ai_starter.retriever.langchain_qa_retriever import LangchainQARetriever
except Exception:
    LangchainQARetriever = None

__all__ = [
    # 核心模块
    "say_hello",
    "get_logger",
    "setup_logging_from_config",
    "generate_trace_id",
    "get_trace_id",
    "set_trace_id",
    "clear_trace_id",
    "trace_context",
    "with_trace",
    "Config",
    "load_config",
    "HttpClientFactory",
    # ZhipuAI 集成
    "ZhipuLLMFactory",
    "ZhipuEmbeddingFactory",
    "ZhipuGlobalSettings",
    # 可选依赖模块
    "ChromaDB",
    "EmbeddingInterface",
    "GLMEmbedding",
    "OpenAIEmbedding",
    "LangChainGLMEmbedding",
    "LangChainEmbeddingInterface",
    "LangChainChatModel",
    "PDFChunker",
    "LangchainChromadb",
    "LangchainQARetriever",
    "CustomChatZhipuAI",
    "QwenAgentLLM",
]

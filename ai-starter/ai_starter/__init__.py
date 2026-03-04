"""
AI Starter - 共享工具包

提供通用的工具函数和AI相关组件

使用方式:
    # 核心功能（总是可用）
    from ai_starter import Config, get_logger, HttpClientFactory

    # Embedding 功能
    from ai_starter.embedding import GLMEmbedding, EmbeddingInterface

    # ChromaDB 功能
    from ai_starter.chromadb import ChromaDB

    # LangChain 功能
    from ai_starter.langchain import LangChainChatZhipuAI, PDFChunker, LangChainGLMEmbedding

    # LlamaIndex 功能
    from ai_starter.llama_index import ZhipuLLMFactory, ZhipuEmbeddingFactory

    # Qwen-Agent 功能
    from ai_starter.qwen_agent import QwenAgentChatZhipuAI

可选依赖安装:
    pip install ai-starter[chromadb]    # ChromaDB + Embedding
    pip install ai-starter[langchain]   # LangChain 集成
    pip install ai-starter[llama-index] # LlamaIndex 集成
    pip install ai-starter[qwen-agent]  # Qwen-Agent 集成
    pip install ai-starter[all]         # 全部功能
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

__all__ = [
    # 版本
    "__version__",
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
]

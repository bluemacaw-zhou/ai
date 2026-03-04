"""
Embedding 模块
"""

from ai_starter.embedding.embedding_interface import EmbeddingInterface
from ai_starter.embedding.embedding_glm import GLMEmbedding
from ai_starter.embedding.embedding_openai import OpenAIEmbedding

__all__ = [
    "EmbeddingInterface",
    "GLMEmbedding",
    "OpenAIEmbedding",
]


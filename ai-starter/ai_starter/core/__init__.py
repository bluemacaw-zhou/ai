"""
ai-starter 核心模块

提供基础功能：配置管理、HTTP客户端、日志
无外部可选依赖，可独立使用
"""

from ai_starter.core.config.config import Config
from ai_starter.core.http_client.http_client_factory import HttpClientFactory
from ai_starter.core.log.logging_utils import get_logger

__all__ = [
    "Config",
    "HttpClientFactory",
    "get_logger",
]

"""
ai-starter 核心模块

提供基础功能：配置管理、日志
无外部可选依赖，可独立使用
"""

from ai_starter.core.config.config import Config
from ai_starter.core.log.logging_utils import get_logger

# HttpClientFactory has been moved to ai_starter.http_client
# For backward compatibility, re-export it here
from ai_starter.http_client.http_client_factory import HttpClientFactory

__all__ = [
    "Config",
    "HttpClientFactory",
    "get_logger",
]

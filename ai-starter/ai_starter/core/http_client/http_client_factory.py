"""
HTTP 客户端工厂

用于创建配置了代理和 SSL 验证的 httpx.Client 实例
"""

import ssl
import httpx
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.core.config.config import Config

logger = get_logger(__name__)


class HttpClientFactory:
    """
    HTTP 客户端工厂

    从配置文件读取配置，创建配置好的 httpx.Client 实例
    """

    @staticmethod
    def create(prefix: str = "api.zhipuai") -> httpx.Client:
        """
        创建配置好的 HTTP 客户端实例（自动从配置文件读取）

        Args:
            prefix: 配置前缀，默认为 "api.zhipuai"

        Returns:
            httpx.Client: 配置好的 HTTP 客户端

        Examples:
            >>> client = HttpClientFactory.create()
            >>> # 使用客户端...
            >>> client.close()
        """
        config = Config()

        verify_ssl = config.get(f"{prefix}.verify_ssl", False)
        use_proxy = config.get(f"{prefix}.use_proxy", True)
        timeout = config.get(f"{prefix}.timeout", 60.0)

        if use_proxy:
            if not verify_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                verify_param = ssl_context
            else:
                verify_param = True

            client = httpx.Client(
                verify=verify_param,
                trust_env=True,
                timeout=timeout
            )
            logger.info(f"创建 http_client (verify_ssl={verify_ssl}, use_proxy=True)")
        else:
            client = httpx.Client(
                verify=verify_ssl,
                trust_env=False,
                timeout=timeout
            )
            logger.info(f"创建 http_client (verify_ssl={verify_ssl}, use_proxy=False)")

        return client

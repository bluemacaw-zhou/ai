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

    从配置文件的 http: 节点读取配置，创建配置好的 httpx.Client 实例
    """

    @staticmethod
    def create() -> httpx.Client:
        """
        创建配置好的 HTTP 客户端实例（自动从配置文件读取）

        配置结构：
            http:
              proxy:
                enabled: true
                http: "http://10.200.86.85:8080"
                https: "http://10.200.86.85:8080"
              timeout: 30
              verify_ssl: false

        Returns:
            httpx.Client: 配置好的 HTTP 客户端

        Examples:
            >>> client = HttpClientFactory.create()
            >>> # 使用客户端...
            >>> client.close()
        """
        config = Config()

        # 从 http: 节点读取配置（必须显式配置，不使用默认值）
        verify_ssl = config.get_required("http.verify_ssl")
        proxy_enabled = config.get_required("http.proxy.enabled")
        timeout = config.get_required("http.timeout")

        if proxy_enabled:
            # 读取代理配置
            http_proxy = config.get_required("http.proxy.http")
            https_proxy = config.get_required("http.proxy.https")

            # 构建代理配置（httpx 使用单数 proxy，不是 proxies）
            # 如果 http 和 https 代理相同，使用字符串；否则使用字典
            if http_proxy == https_proxy and http_proxy:
                proxy = http_proxy
            else:
                proxy = {}
                if http_proxy:
                    proxy["http://"] = http_proxy
                if https_proxy:
                    proxy["https://"] = https_proxy
                proxy = proxy if proxy else None

            # 关键修复：当 verify_ssl=False 时，使用 SSL context 而不是直接传递布尔值
            # 这是之前能工作的原因
            if not verify_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                verify_param = ssl_context
            else:
                verify_param = True

            client = httpx.Client(
                verify=verify_param,
                proxy=proxy,
                trust_env=True,  # 使用 trust_env=True（与原始实现一致）
                timeout=timeout
            )
            logger.info(f"创建 http_client (verify_ssl={verify_ssl}, proxy_enabled=True, proxy={proxy})")
        else:
            client = httpx.Client(
                verify=verify_ssl,
                trust_env=False,
                timeout=timeout
            )
            logger.info(f"创建 http_client (verify_ssl={verify_ssl}, proxy_enabled=False)")

        return client

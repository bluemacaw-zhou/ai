"""
ZhipuAI HTTP API 调用的公共基类

封装 ZhipuAI API 的通用逻辑：
- 配置管理（API Key、API Base、模型参数）
- HTTP 客户端管理
- JWT Token 生成
- API 调用封装
"""

import os
import httpx
import time
import jwt
import json
from typing import Dict, List, Optional, Iterator
from ai_starter.core.log.logging_utils import get_logger
from ai_starter.http_client.http_client_factory import HttpClientFactory
from ai_starter.core.config.config import Config

logger = get_logger(__name__)


class ZhipuAIBase:
    """
    ZhipuAI 公共基类

    提供：
    - 配置读取和管理
    - HTTP 客户端创建
    - API 调用封装
    """

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        """
        初始化 ZhipuAI 基类

        Args:
            model: 模型名称（可选，从配置读取）
            temperature: 温度参数（可选，从配置读取）
        """
        # 从 Config 读取所有配置
        config = Config()

        # 设置模型参数（必须在配置中存在）
        self._model = model or config.get_required("zhipu.llm.model")
        self._temperature = temperature if temperature is not None else config.get_required("zhipu.llm.temperature")

        # API Key 从环境变量读取
        self._api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not self._api_key:
            raise ValueError("环境变量 ZHIPUAI_API_KEY 未设置，请先设置后再运行")

        # API Base URL 从配置读取（必须显式配置）
        self._api_base = config.get_required("zhipu.api_base")

        # 创建自定义 http_client（自动从配置读取代理和 SSL 设置）
        self._http_client = HttpClientFactory.create()

        logger.info(f"ZhipuAI initialized (model: {self._model}, temperature: {self._temperature})")

    @property
    def model(self) -> str:
        """模型名称"""
        return self._model

    @property
    def temperature(self) -> float:
        """温度参数"""
        return self._temperature

    @property
    def api_key(self) -> str:
        """API Key"""
        return self._api_key

    @property
    def api_base(self) -> str:
        """API Base URL"""
        return self._api_base

    @property
    def http_client(self) -> httpx.Client:
        """HTTP 客户端"""
        return self._http_client

    @staticmethod
    def _get_jwt_token(api_key: str) -> str:
        """
        生成 ZhipuAI JWT Token

        Args:
            api_key: ZhipuAI API Key (格式: "key.secret")

        Returns:
            JWT Token 字符串

        Raises:
            ValueError: API Key 格式错误
        """
        try:
            key, secret = api_key.split(".")
        except Exception as e:
            raise ValueError(f"Invalid API key format: {e}")

        payload = {
            "api_key": key,
            "exp": int(round(time.time() * 1000)) + 3600 * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

    def _call_api(
        self,
        messages: List[Dict],
        stream: bool = False,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        调用 ZhipuAI API

        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            stream: 是否使用流式生成
            timeout: 超时时间（秒）
            **kwargs: 其他 API 参数

        Returns:
            API 响应 JSON
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        }

        headers = {
            "Authorization": self._get_jwt_token(self._api_key),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        logger.info(f"调用 ZhipuAI API: {self._api_base}")

        try:
            response = self._http_client.post(
                self._api_base,
                json=payload,
                headers=headers,
                timeout=timeout or 60
            )
            response.raise_for_status()
            result = response.json()

            logger.info("API 调用成功")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"API 调用失败: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"API 调用异常: {e}")
            raise

    def _call_api_stream(
        self,
        messages: List[Dict],
        timeout: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        调用 ZhipuAI API（流式）

        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            timeout: 超时时间（秒）
            **kwargs: 其他 API 参数

        Yields:
            流式响应的文本片段
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "stream": True,
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "stream")}
        }

        headers = {
            "Authorization": self._get_jwt_token(self._api_key),
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }

        logger.info(f"调用 ZhipuAI API（流式）: {self._api_base}")

        try:
            with self._http_client.stream(
                "POST",
                self._api_base,
                json=payload,
                headers=headers,
                timeout=timeout or 60
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.strip():
                        continue

                    # 跳过注释行
                    if line.startswith(":"):
                        continue

                    # 解析 SSE 格式: "data: {...}"
                    if line.startswith("data:"):
                        data_str = line[5:].strip()

                        # 流结束标志
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            # 提取 delta content
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield content

                        except json.JSONDecodeError as e:
                            logger.warning(f"解析流式响应失败: {e}, line: {line}")
                            continue

        except httpx.HTTPStatusError as e:
            logger.error(f"流式 API 调用失败: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"流式 API 调用异常: {e}")
            raise

    def __del__(self):
        """清理资源"""
        try:
            self._http_client.close()
        except Exception:
            pass

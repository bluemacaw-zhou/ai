"""
日志工具模块
提供日志记录、链路追踪等功能
"""

import logging
import sys
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

from ai_starter.core.config.config import Config

# 全局logger缓存，避免重复创建
_loggers = {}

# 线程本地存储，用于链路追踪
_trace_context = threading.local()


class TraceFilter(logging.Filter):
    """日志过滤器，自动添加 trace_id 到日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = get_trace_id()
        if not trace_id:
            # 当前线程没有 trace_id，自动生成一个
            trace_id = set_trace_id()
        record.trace_id = trace_id
        return True


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    获取或创建logger实例（使用Python标准logging模块）

    链路追踪默认开启，日志格式固定。

    Args:
        name: Logger名称，通常使用 __name__
        level: 日志级别，可选 DEBUG, INFO, WARNING, ERROR, CRITICAL（优先使用此参数，其次从config读取，默认INFO）
        log_file: 日志文件路径（优先使用此参数，其次从config读取）

    Returns:
        logging.Logger实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        >>> logger.warning("This is a warning")
        >>> logger.error("This is an error")

        # 覆盖配置，强制使用DEBUG级别
        >>> logger = get_logger(__name__, level="DEBUG")

        # 指定日志文件
        >>> logger = get_logger(__name__, log_file="logs/app.log")
    """
    # 如果已经创建过，直接返回
    if name in _loggers:
        return _loggers[name]

    # 从全局配置读取，参数可覆盖（懒加载）
    config = Config()

    # 确定日志级别：参数 > 配置 > 默认值
    level = level or config.get("logging.level", "INFO")

    # 确定日志文件：参数 > 配置
    log_file = log_file or config.get("logging.file")

    # 硬编码日志格式（固定包含链路追踪和线程信息）
    format_str = "%(asctime)s - [%(trace_id)s] - %(levelname)-5s - [%(threadName)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()

    # 创建formatter
    formatter = logging.Formatter(format_str)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 添加链路追踪过滤器（强制开启）
    trace_filter = TraceFilter()
    logger.addFilter(trace_filter)

    # 缓存logger
    _loggers[name] = logger

    return logger


def setup_logging_from_config() -> None:
    """
    从Config配置全局日志

    日志格式固定，链路追踪默认开启。
    自动从全局配置读取 logging.level 和 logging.file。

    Examples:
        >>> from ai_starter import setup_logging_from_config
        >>> setup_logging_from_config()
    """
    # 从全局配置读取（懒加载）
    config = Config()

    level = config.get("logging.level", "INFO")
    log_file = config.get("logging.file")

    # 硬编码日志格式（固定包含链路追踪和线程信息）
    format_str = "%(asctime)s - [%(trace_id)s] - %(levelname)-5s - [%(threadName)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"

    # 配置root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 如果配置了日志文件
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)


# ==================== 链路追踪相关函数 ====================

def generate_trace_id() -> str:
    """
    生成新的 trace_id

    Returns:
        str: 格式为 "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx" 的 UUID

    Examples:
        >>> trace_id = generate_trace_id()
        >>> print(trace_id)
        'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
    """
    return str(uuid.uuid4())


def get_trace_id() -> Optional[str]:
    """
    获取当前线程的 trace_id

    Returns:
        str: 当前 trace_id，如果未设置则返回 None

    Examples:
        >>> trace_id = get_trace_id()
        >>> if trace_id:
        ...     print(f"当前链路: {trace_id}")
    """
    return getattr(_trace_context, 'trace_id', None)


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    设置当前线程的 trace_id

    Args:
        trace_id: trace_id，如果不提供则自动生成

    Returns:
        str: 设置的 trace_id

    Examples:
        >>> trace_id = set_trace_id()
        >>> print(f"开始链路: {trace_id}")
    """
    if trace_id is None:
        trace_id = generate_trace_id()
    _trace_context.trace_id = trace_id
    return trace_id


def clear_trace_id() -> None:
    """
    清除当前线程的 trace_id

    Examples:
        >>> clear_trace_id()
    """
    _trace_context.trace_id = None


@contextmanager
def trace_context(trace_id: Optional[str] = None) -> Generator[str, None, None]:
    """
    链路追踪上下文管理器

    用于手动控制 trace_id 的场景，比如：
    - 从 HTTP 请求头传递 trace_id
    - MQ 消息携带的 trace_id
    - 需要临时覆盖当前 trace_id

    注意：大多数情况下不需要手动使用，每个线程会自动获得 trace_id。

    Args:
        trace_id: trace_id，如果不提供则自动生成

    Yields:
        str: 当前 trace_id

    Examples:
        >>> # 自动生成新的 trace_id
        >>> with trace_context() as tid:
        ...     logger.info("这条日志会有新的 trace_id")
        ...     do_business_logic()

        >>> # 使用自定义 trace_id（比如从请求头传递）
        >>> with trace_context("custom-trace-123") as tid:
        ...     logger.info("使用自定义 trace_id")
    """
    old_trace_id = get_trace_id()
    new_trace_id = set_trace_id(trace_id)
    try:
        yield new_trace_id
    finally:
        if old_trace_id:
            set_trace_id(old_trace_id)
        else:
            clear_trace_id()


def with_trace(func=None, *, trace_id: Optional[str] = None):
    """
    装饰器：为函数执行自动分配新的 trace_id

    适用场景：
    - HTTP 请求处理函数
    - MQ 消息处理函数
    - 异步任务处理函数
    - 任何需要独立链路追踪的函数

    Args:
        func: 被装饰的函数
        trace_id: 可选的固定 trace_id，通常留空自动生成

    Examples:
        >>> @with_trace
        ... def handle_request(data):
        ...     logger.info("处理请求")
        ...     return process(data)

        >>> # 每次调用 handle_request 都会获得新的 trace_id
        >>> handle_request({"user_id": 123})
        >>> handle_request({"user_id": 456})

        >>> # 用于 MQ 消息处理
        >>> @with_trace
        ... def on_message(message):
        ...     logger.info(f"收到消息: {message}")
        ...     process_message(message)
    """
    from functools import wraps

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with trace_context(trace_id):
                return f(*args, **kwargs)
        return wrapper

    # 支持 @with_trace 和 @with_trace(trace_id="xxx") 两种用法
    if func is None:
        return decorator
    else:
        return decorator(func)


def say_hello() -> str:
    """输出 Hello World

    Returns:
        Hello World 字符串
    """
    # 测试用的局部变量 - 方便断点调试时查看
    greeting = "Hello"
    target = "World"
    punctuation = "!"

    # 拼接字符串
    message = f"{greeting}, {target}{punctuation}"

    # 额外的调试信息
    message_length = len(message)
    is_valid = message_length > 0

    # 可以在这里设置断点，查看上面所有局部变量的值
    return message

"""项目统一日志与 OpenTelemetry 配置。"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from collections.abc import Iterator
from typing import Any

import structlog
from opentelemetry import baggage, context as otel_context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


def _add_otel_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """注入当前链路和 A2A 标识；上下文缺失时使用占位符。"""
    span_context = trace.get_current_span().get_span_context()
    a2a_context_id = baggage.get_baggage("a2a.context.id")
    a2a_task_id = baggage.get_baggage("a2a.task.id")

    event_dict["trace_id"] = (
        format(span_context.trace_id, "032x") if span_context.is_valid else "-"
    )
    event_dict["span_id"] = (
        format(span_context.span_id, "016x") if span_context.is_valid else "-"
    )
    event_dict["a2a_context_id"] = (
        str(a2a_context_id) if a2a_context_id else "-"
    )
    event_dict["a2a_task_id"] = str(a2a_task_id) if a2a_task_id else "-"
    return event_dict


def _add_thread_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """注入当前 Python 物理线程标识。"""
    current_thread = threading.current_thread()
    event_dict["thread_id"] = threading.get_ident()
    event_dict["thread_name"] = current_thread.name
    return event_dict


def _add_local_timestamp(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """按 Spring Boot 常见格式注入本地时间，精确到毫秒。"""
    event_dict["timestamp"] = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]
    return event_dict


def _render_value(value: Any) -> str:
    """将附加字段渲染为紧凑且无歧义的单行文本。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        if value and not any(character.isspace() for character in value):
            return value
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _log4j_renderer(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> str:
    """渲染为接近 Spring Boot/Log4j 的单行日志格式。"""
    timestamp = event_dict.pop("timestamp", "-")
    level = str(event_dict.pop("level", "info")).upper()
    event = event_dict.pop("event", "-")
    event_dict.pop("logger", None)
    thread_name = event_dict.pop("thread_name", "-")
    thread_id = event_dict.pop("thread_id", "-")
    filename = event_dict.pop("filename", "-")
    lineno = event_dict.pop("lineno", "-")
    trace_id = event_dict.pop("trace_id", "-")
    span_id = event_dict.pop("span_id", "-")
    a2a_context_id = event_dict.pop("a2a_context_id", "-")
    a2a_task_id = event_dict.pop("a2a_task_id", "-")

    extra_fields = " ".join(
        f"{key}={_render_value(value)}"
        for key, value in sorted(event_dict.items())
    )
    extra_suffix = f" {extra_fields}" if extra_fields else ""

    return (
        f"{timestamp} {level:<5} --- "
        f"[contextId={a2a_context_id} taskId={a2a_task_id}] "
        f"[traceId={trace_id} spanId={span_id}] "
        f"[{thread_name}:{thread_id}] "
        f"{filename}:{lineno} - {event}{extra_suffix}"
    )


def get_logger(name: str | None = None) -> Any:
    """返回一个 structlog logger，供各模块创建模块级 ``log`` 变量。

    使用方式::

        from config import get_logger

        log = get_logger(__name__)
        log.info("agent.task.started")
    """
    return structlog.get_logger(name)


def configure_logging() -> None:
    """为 structlog 和标准库日志集中配置一个根 Handler。"""
    # Windows 控制台默认编码常为 cp936/gbk，输出中文会触发 UnicodeEncodeError。
    # StreamHandler 写 sys.stdout 时会被 logging 在 handleError 里静默吞掉，导致
    # 整条中文日志丢失。这里先把标准输出/错误显式切到 UTF-8，保证中文日志正常。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    callsite_adder = structlog.processors.CallsiteParameterAdder(
        parameters={
            structlog.processors.CallsiteParameter.FILENAME,
            structlog.processors.CallsiteParameter.LINENO,
        }
    )
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_local_timestamp,
        _add_otel_context,
        _add_thread_context,
        callsite_adder,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    output_renderer = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if os.getenv("LOG_FORMAT", "console").lower() == "json"
        else _log4j_renderer
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            output_renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def configure_tracing(
    *, resource_attributes: dict[str, Any] | None = None
) -> trace.Tracer:
    """幂等初始化全局 SDK Tracer，并允许观测组件补充 Resource 属性。"""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        attributes: dict[str, Any] = {"service.name": "wind-fixed-income-aipad"}
        if resource_attributes:
            attributes.update(resource_attributes)
        provider = TracerProvider(resource=Resource.create(attributes))
        trace.set_tracer_provider(provider)
    return provider.get_tracer("wind-fixed-income-aipad.logging")


@contextmanager
def a2a_trace_context(
    tracer: trace.Tracer,
    *,
    context_id: str,
    task_id: str,
) -> Iterator[None]:
    """将 A2A 标识绑定到当前 OTel Baggage 和根 Span。"""
    a2a_context = baggage.set_baggage("a2a.context.id", context_id)
    a2a_context = baggage.set_baggage(
        "a2a.task.id", task_id, context=a2a_context
    )
    context_token = otel_context.attach(a2a_context)
    try:
        with tracer.start_as_current_span(
            "a2a-task",
            attributes={
                "a2a.context.id": context_id,
                "a2a.task.id": task_id,
            },
        ):
            yield
    finally:
        otel_context.detach(context_token)

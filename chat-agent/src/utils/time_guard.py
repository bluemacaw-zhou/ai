"""行情类工具的时间戳新鲜度校验。

金价/油价/汇率等工具的 as_of 必须是刚获取的当前时间戳。若参数时间无法解析
（例如占位符 "%s"）或与当前时间相差超过 30 秒（例如凭空编造的历史日期），
则返回空数据 + ok=false + 明确的错误说明，从而触发上层 agent 重新获取时间并重试。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mcp_tools.tool_result import build_tool_result

FRESHNESS_SECONDS = 30


def stale_timestamp_error(as_of: Any, data_schema: dict[str, Any]) -> list[dict[str, Any]] | None:
    """as_of 合法且新鲜返回 None；否则返回一个空数据的错误结果。"""
    parsed = _parse_iso(as_of)
    if parsed is None:
        return _error(
            data_schema,
            f"as_of={as_of!r} 不是合法的 ISO 8601 时间戳",
        )

    now = datetime.now().astimezone()
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    delta = abs((now - parsed).total_seconds())
    if delta > FRESHNESS_SECONDS:
        return _error(
            data_schema,
            f"as_of 与当前时间相差 {int(delta)} 秒，超过 {FRESHNESS_SECONDS} 秒有效期",
        )
    return None


def _parse_iso(as_of: Any) -> datetime | None:
    if not isinstance(as_of, str):
        return None
    try:
        return datetime.fromisoformat(as_of.strip())
    except ValueError:
        return None


def _error(data_schema: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    return build_tool_result(
        {},
        data_schema,
        mcp_tool_error_code="STALE_TIMESTAMP",
        mcp_tool_error_msg=(
            f"{reason}；请先调用 current_datetime_tool 获取当前时间戳，"
            "并用返回的具体值重试。"
        ),
    )

"""当前日期时间 mock tool。"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool

from mcp_tools.tool_result import build_tool_result, output_schema

_DATA_SCHEMA = {
    "type": "object",
    "description": "当前日期与时间。",
    "properties": {
        "date": {"type": "string", "description": "当前日期（YYYY-MM-DD）。"},
        "time": {"type": "string", "description": "当前时间（HH:MM:SS）。"},
        "datetime": {"type": "string", "description": "本地完整日期时间（ISO 8601）。"},
        "utc_datetime": {"type": "string", "description": "UTC 完整日期时间（ISO 8601）。"},
        "weekday": {"type": "string", "description": "星期几（英文）。"},
    },
}


@tool(
    "current_datetime_tool",
    description=(
        "获取当前日期和时间。无需入参。"
        "产出当前的具体日期（YYYY-MM-DD）与 ISO 8601 时间戳，"
        "用于把“今天/明天/最近/此刻/下周”等相对时间换算成具体值:"
        "日期供天气、旅馆均价、旅馆搜索、预订等工具使用;"
        "时间戳供金价、油价、汇率等行情工具使用。"
    ),
    extras={"source": "local", "output_schema": output_schema(_DATA_SCHEMA)},
)
async def current_datetime_tool() -> list[dict]:
    """本地实现，返回真实的当前日期时间。"""
    now = datetime.now().astimezone()
    return build_tool_result(
        {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.isoformat(),
            "utc_datetime": now.astimezone(timezone.utc).isoformat(),
            "weekday": now.strftime("%A"),
        },
        _DATA_SCHEMA,
    )

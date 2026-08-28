"""油价查询本地 mock tool。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from mcp_tools.tool_result import build_tool_result, output_schema
from utils.time_guard import stale_timestamp_error

_DATA_SCHEMA = {
    "type": "object",
    "description": "指定时刻的油价报价。",
    "properties": {
        "symbol": {"type": "string", "description": "品种代码。"},
        "as_of": {"type": "string", "description": "报价时刻。"},
        "price": {"type": "number", "description": "价格。"},
        "currency": {"type": "string", "description": "币种。"},
        "unit": {"type": "string", "description": "计价单位。"},
    },
}


@tool(
    "oil_price_tool",
    description=(
        "查询指定时刻的油价（WTI 原油）。"
        "入参：时间戳 as_of（ISO 8601，如 2026-06-29T14:00:00+08:00；"
        "若用户说“最新/此刻/最近”，需先获取当前时间得到该时间戳）。"
        "as_of 必须是刚从 current_datetime_tool 获取的当前时间戳（有效期约 30 秒），"
        "过期或非法时间戳会返回空数据。"
        "返回该时刻的油价、币种、单位与报价时刻。"
    ),
    extras={"source": "local", "output_schema": output_schema(_DATA_SCHEMA)},
)
async def oil_price_tool(
    as_of: Annotated[str, "报价时刻的时间戳，ISO 8601。"],
) -> list[dict]:
    """本地 mock 实现，返回该时刻的固定油价。"""
    stale = stale_timestamp_error(as_of, _DATA_SCHEMA)
    if stale is not None:
        return stale
    return build_tool_result(
        {
            "symbol": "WTI",
            "as_of": as_of,
            "price": 79.3,
            "currency": "USD",
            "unit": "barrel",
        },
        _DATA_SCHEMA,
    )

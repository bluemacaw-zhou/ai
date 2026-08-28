"""货币汇率 mock tool。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from mcp_tools.tool_result import build_tool_result, output_schema
from utils.time_guard import stale_timestamp_error

_DATA_SCHEMA = {
    "type": "object",
    "description": "货币汇率结果。",
    "properties": {
        "base_currency": {"type": "string", "description": "基准货币。"},
        "quote_currency": {"type": "string", "description": "计价货币。"},
        "rate": {"type": "number", "description": "1 单位基准货币可兑换的计价货币数量。"},
        "as_of": {"type": "string", "description": "报价时间。"},
    },
}


@tool(
    "currency_exchange_rate_tool",
    description=(
        "查询指定时刻两种货币之间的汇率。"
        "入参：基准货币代码（如 USD）、计价货币代码（如 CNY）、"
        "时间戳 as_of（ISO 8601；若用户说“最新/此刻”，需先获取当前时间得到该时间戳）。"
        "as_of 必须是刚从 current_datetime_tool 获取的当前时间戳（有效期约 30 秒），"
        "过期或非法时间戳会返回空数据。"
    ),
    extras={"source": "local", "output_schema": output_schema(_DATA_SCHEMA)},
)
async def currency_exchange_rate_tool(
    base_currency: Annotated[str, "基准货币代码，如 USD。"],
    quote_currency: Annotated[str, "计价货币代码，如 CNY。"],
    as_of: Annotated[str, "报价时刻的时间戳，ISO 8601。"],
) -> list[dict]:
    """本地 mock 实现，只返回固定汇率。"""
    stale = stale_timestamp_error(as_of, _DATA_SCHEMA)
    if stale is not None:
        return stale
    return build_tool_result(
        {
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "rate": 7.18,
            "as_of": as_of,
        },
        _DATA_SCHEMA,
    )

"""MCP 工具结果相关工具。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from langchain_core.tools import BaseTool

from utils.mcp_payload import first_text_json_block, loads_json


def output_schema(data_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回与 MCP 报文一致的输出 schema。"""
    schema = {
        "type": "object",
        "description": "Normalized MCP text payload.",
        "properties": {
            "mcp_tool_code": {"description": "MCP transport status code."},
            "mcp_tool_msg": {"type": "string", "description": "MCP transport message."},
            "mcp_tool_data_schema": {
                "type": "object",
                "description": "Business data schema defined by tool.",
            },
            "mcp_tool_data": {"type": "string", "description": "Business data JSON string."},
            "mcp_tool_error_code": {"description": "Business error code."},
            "mcp_tool_error_msg": {"type": "string", "description": "Business error message."},
        },
        "required": [
            "mcp_tool_code",
            "mcp_tool_msg",
            "mcp_tool_data_schema",
            "mcp_tool_data",
            "mcp_tool_error_code",
            "mcp_tool_error_msg",
        ],
    }
    if data_schema is not None:
        schema["properties"]["mcp_tool_data_schema"] = data_schema
    return schema


def build_tool_result(
    data: Any,
    data_schema: dict[str, Any] | None = None,
    *,
    mcp_tool_code: int = 0,
    mcp_tool_msg: str = "success",
    mcp_tool_error_code: str | int | None = 0,
    mcp_tool_error_msg: str = "",
) -> list[dict[str, Any]]:
    """构造与 MCP 一致的 text block 返回值。"""
    payload = {
        "mcp_tool_code": mcp_tool_code,
        "mcp_tool_msg": mcp_tool_msg,
        "mcp_tool_data_schema": data_schema or {},
        "mcp_tool_data": json.dumps(data, ensure_ascii=False),
        "mcp_tool_error_code": mcp_tool_error_code,
        "mcp_tool_error_msg": mcp_tool_error_msg,
    }
    return [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]


def parse_tool_result(raw: Any) -> dict[str, Any]:
    """把 MCP 原始返回解析成主流程可消费的结果。"""
    payload = first_text_json_block(raw, required_key="mcp_tool_data")
    if payload is None:
        return {
            "ok": False,
            "data": None,
            "data_schema": None,
            "error_code": "TOOL_RESULT_PARSE_ERROR",
            "error_message": "Tool result is not the expected text JSON payload.",
            "payload": None,
            "raw": raw,
        }

    data_schema = payload.get("mcp_tool_data_schema")
    business = loads_json(payload.get("mcp_tool_data"))
    mcp_error_code = payload.get("mcp_tool_error_code")
    mcp_error_message = payload.get("mcp_tool_error_msg") or None

    if isinstance(business, dict) and _looks_like_business_envelope(business):
        ok = (
            _is_success_code(mcp_error_code)
            and business.get("state") == 0
            and not business.get("errorCode")
        )
        data = business.get("data")
        error_code = None if ok else business.get("errorCode") or mcp_error_code
        error_message = None if ok else business.get("errorMessage") or mcp_error_message
    else:
        ok = _is_success_code(mcp_error_code)
        data = business
        error_code = None if ok else mcp_error_code
        error_message = None if ok else mcp_error_message

    return {
        "ok": ok,
        "data": data,
        "data_schema": data_schema if isinstance(data_schema, dict) else None,
        "error_code": error_code,
        "error_message": error_message,
        "payload": payload,
        "raw": raw,
    }


def describe_tools(tools: Iterable[BaseTool]) -> list[dict[str, Any]]:
    """输出调用前可见的工具元数据。"""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": getattr(tool, "args", None),
            "source": _tool_extra(tool, "source"),
        }
        for tool in tools
    ]


def _tool_extra(tool: BaseTool, key: str) -> Any:
    return (getattr(tool, "extras", None) or {}).get(key)


def _looks_like_business_envelope(value: dict[str, Any]) -> bool:
    return {"data", "errorCode", "errorMessage", "state"}.issubset(value.keys())


def _is_success_code(value: Any) -> bool:
    return value in (None, 0, "0", "")

"""MCP 报文 (text content block) 解析工具。"""

from __future__ import annotations

import json
from typing import Any


def loads_json(value: Any) -> Any:
    """字符串就尝试 JSON 解析，解析失败或非字符串原样返回。"""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def first_text_json_block(raw: Any, required_key: str) -> dict[str, Any] | None:
    """从 MCP 返回的 content blocks 里找出第一个含 required_key 的 text JSON 块。

    MCP 工具返回是一组 content block，业务数据通常放在 type=text 的块里、
    内容是一段 JSON 字符串。这里负责把它解析出来。
    """
    if not isinstance(raw, list):
        return None
    for block in raw:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        payload = loads_json(block.get("text"))
        if isinstance(payload, dict) and required_key in payload:
            return payload
    return None

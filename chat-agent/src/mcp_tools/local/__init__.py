"""本地 MCP 工具集 (无需第三方地址)。

加载方式: 扫描本包下所有模块里用 @tool 定义的工具，全部收集。
新增本地工具 = 在本目录新建一个模块、用 @tool 定义即可，无需在别处登记。
"""

from __future__ import annotations

import importlib
import pkgutil

from langchain_core.tools import BaseTool


def scan_local_tools() -> list[BaseTool]:
    """扫描本包，收集所有 @tool 定义的工具。"""
    tools: dict[str, BaseTool] = {}
    for module_info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, BaseTool):
                tools[value.name] = value
    return list(tools.values())

"""为 ExecuteCommand 加载 wind MCP server 下的指定工具子集。

design/harness/05-harness-loop-overview.puml 中 ``ExecuteCurrentCommand``
可用的 MCP 工具包括查询、计算、渲染等专业工具；当前只接入查询类的两个工具：
``financial_query_data``（结构化数据查询）与 ``fin_doc_searchV3``（文档检索）。

与 ``mcp_tools.tool_loader.load_configured_tools`` 不同：那个函数会加载
``config.yaml`` 里 ``mcp.servers`` 下配置的全部 server（本地工具 + wind +
cosmos），工具集合由各 server 自己的 ``tool_names`` 白名单决定。本模块只精确
加载 ``mcp.servers.wind`` 这一个 server，并在此基础上再按名称收窄到
``financial_query_data``/``fin_doc_searchV3`` 这两个工具，不受
``config.yaml`` 里 ``wind.tool_names`` 配置的其它工具（如 ``search_sectors``/
``fin_ner``）影响，也不加载 cosmos/本地工具。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from config import get_logger
from config.mcp_settings import McpSettings
from mcp_tools.remote import RemoteMcpTool

log = get_logger(__name__)

#: ExecuteCommand 当前只绑定这两个查询类工具。
EXECUTE_COMMAND_TOOL_NAMES: tuple[str, ...] = (
    "financial_query_data",
    "fin_doc_searchV3",
)

_WIND_SERVER_NAME = "wind"


async def load_execute_command_tools(session_id: str | None = None) -> list[BaseTool]:
    """只加载 mcp.servers.wind 下的 financial_query_data / fin_doc_searchV3。

    Args:
        session_id: 用户会话标识，注入到 wind server 的 session_headers；
            server 配置了 session_headers 时必须提供，否则报错，行为与
            ``mcp_tools.tool_loader.load_configured_tools`` 一致。

    Returns:
        绑定给 agent 的工具列表（可能为空，若配置或远端未返回目标工具）。
    """
    server = McpSettings.load().servers.get(_WIND_SERVER_NAME)
    if server is None:
        log.warning(
            "harness_graph.execute_command_tools.wind_server_not_configured"
        )
        return []
    if server.session_headers and not session_id:
        raise ValueError(
            f"MCP server '{server.name}' requires a request session id"
        )

    client = await RemoteMcpTool.load(server, session_id=session_id)
    all_tools = {tool.name: tool for tool in client.list_tools()}

    tools = [
        all_tools[name] for name in EXECUTE_COMMAND_TOOL_NAMES if name in all_tools
    ]
    missing = [name for name in EXECUTE_COMMAND_TOOL_NAMES if name not in all_tools]
    if missing:
        log.warning(
            "harness_graph.execute_command_tools.missing_tools",
            missing=missing,
            available=list(all_tools.keys()),
        )
    return tools

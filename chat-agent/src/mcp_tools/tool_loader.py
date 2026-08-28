"""Load local tools and explicitly configured remote MCP tools."""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

from config.mcp_settings import McpSettings
from mcp_tools.local import scan_local_tools
from mcp_tools.remote import RemoteMcpTool

logger = logging.getLogger(__name__)


async def load_configured_tools(session_id: str | None = None) -> list[BaseTool]:
    tools = list(scan_local_tools())
    for server in McpSettings.load().servers.values():
        if server.session_headers and not session_id:
            raise ValueError(f"MCP server '{server.name}' requires a request session id")
        server_tools = (await RemoteMcpTool.load(server, session_id)).list_tools()
        logger.info("MCP tools loaded: server=%s tool_count=%s", server.name, len(server_tools))
        tools.extend(server_tools)
    return tools

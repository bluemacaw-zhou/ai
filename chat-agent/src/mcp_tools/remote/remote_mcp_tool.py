"""Generic streamable-HTTP remote MCP adapter."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from config.mcp_settings import McpServerSettings
from mcp_tools.tool_result import output_schema
from utils.mcp_http_client import create_mcp_http_client


class RemoteMcpTool:
    """Tools exposed by one configured remote MCP server."""

    def __init__(self, server: McpServerSettings, raw_tools: list[BaseTool]):
        self._server = server
        self._raw_tools = {tool.name: tool for tool in raw_tools}

    @classmethod
    async def load(cls, server: McpServerSettings, session_id: str | None = None) -> "RemoteMcpTool":
        return cls(server, await _fetch_raw_tools(server, session_id))

    def list_tools(self) -> list[BaseTool]:
        return [
            self._as_bindable_tool(tool)
            for tool in self._raw_tools.values()
            if not self._server.tool_names or tool.name in self._server.tool_names
        ]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._raw_tools.get(name)
        if tool is None:
            raise KeyError(f"Remote MCP tool does not exist: {name}")
        return await tool.ainvoke(arguments)

    def _as_bindable_tool(self, raw_tool: BaseTool) -> BaseTool:
        async def call(**arguments: Any) -> Any:
            return await self.execute_tool(raw_tool.name, arguments)

        return StructuredTool.from_function(
            coroutine=call,
            name=raw_tool.name,
            description=raw_tool.description,
            args_schema=raw_tool.args_schema,
            infer_schema=False,
            extras={
                **(getattr(raw_tool, "extras", None) or {}),
                "source": "remote",
                "output_schema": _raw_output_schema(raw_tool),
            },
        )


async def _fetch_raw_tools(server: McpServerSettings, session_id: str | None) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    headers = {header: session_id for header in server.session_headers if session_id}
    headers.update({"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    timeout = timedelta(seconds=server.http.timeout or 60)
    client = MultiServerMCPClient({server.name: {
        "transport": "streamable_http", "url": server.url, "headers": headers,
        "timeout": timeout, "sse_read_timeout": timeout, "terminate_on_close": False,
        "httpx_client_factory": create_mcp_http_client,
    }})
    return await client.get_tools()


def _raw_output_schema(raw_tool: BaseTool) -> dict[str, Any]:
    schema = (getattr(raw_tool, "extras", None) or {}).get("output_schema")
    return schema if isinstance(schema, dict) else output_schema()

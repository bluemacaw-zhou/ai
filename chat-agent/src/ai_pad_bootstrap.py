"""Application composition root for the HTTP and LangGraph layers."""

from __future__ import annotations

from config.chat_model import ChatModelRegistry
from graph.main_graph import MainGraph
from mcp_tools.tool_loader import load_configured_tools


class AiPadBootstrap:
    @staticmethod
    async def build_main_graph() -> MainGraph:
        registry = ChatModelRegistry.instance()
        tools = await load_configured_tools()
        return MainGraph(registry.model("default"), tools)

"""Pure LangGraph entry graph for one natural-language question."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

_SYSTEM_PROMPT = "You are a helpful assistant. Use an available tool when it improves accuracy."


class MainGraph:
    """The application-level LangGraph: agent -> tools -> agent."""

    def __init__(self, model: Any, tools: list[BaseTool]) -> None:
        self._model = model.bind_tools(tools)
        self._tools = tools
        self._graph = self._build_graph()

    def _build_graph(self):
        async def call_model(state: MessagesState) -> dict[str, list[Any]]:
            response = await self._model.ainvoke(
                [SystemMessage(_SYSTEM_PROMPT), *state["messages"]]
            )
            return {"messages": [response]}

        def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
            return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", route_after_model, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    async def run(self, question: str) -> str:
        """Answer one natural-language question and return the final text."""
        state = await self._graph.ainvoke(
            {"messages": [HumanMessage(question)]},
            {"recursion_limit": 32},
        )
        content = state["messages"][-1].content
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

"""Build a HarnessGraph (05 图) whose MCP tools belong to one A2A request.

MCP 工具需要按请求的 session_id 鉴权，因此 ``HarnessGraph`` 不能是进程级
复用的单例，需要每个请求单独创建一次，复用进程级共享的模型注册表。

``max_loop_iterations``（``CheckFailureThresholds`` 真正实现前的临时轮数
硬上限，见 05-minimal-loop-implementation-plan.md"细节 7"）支持从
``config.yaml`` 的 ``harness.max_loop_iterations`` 读取；未配置时使用
``harness_state.DEFAULT_MAX_LOOP_ITERATIONS``。

``observability``（进程级共享的 ``LangfuseObservability``）原样转发给每个
请求构造出的 ``HarnessGraph``，使 05 图的 Langfuse trace 与顶层 08 图
（``MainGraph``）用同一套 callback 配置逻辑。
"""

from __future__ import annotations

from typing import Any

from config.app_config import Config
from config.observability import LangfuseObservability
from graph.harness_graph.execute_command_tools import load_execute_command_tools
from graph.harness_graph.harness_graph import HarnessGraph
from graph.harness_graph.harness_state import DEFAULT_MAX_LOOP_ITERATIONS


class HarnessGraphFactory:
    """Creates request-scoped HarnessGraph while reusing the shared model registry."""

    def __init__(
        self, registry: Any, *, observability: LangfuseObservability | None = None
    ) -> None:
        self._registry = registry
        self._observability = observability

    async def create(self, session_id: str) -> HarnessGraph:
        """Create a HarnessGraph whose ExecuteCommand tools are bound to this request's user."""
        tools = await load_execute_command_tools(session_id=session_id)
        max_loop_iterations = Config().get(
            "harness.max_loop_iterations", DEFAULT_MAX_LOOP_ITERATIONS
        )
        return HarnessGraph(
            self._registry.model("default"),
            tools,
            max_loop_iterations=int(max_loop_iterations),
            observability=self._observability,
        )

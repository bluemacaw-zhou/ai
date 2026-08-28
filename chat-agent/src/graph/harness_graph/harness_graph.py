"""HarnessGraph：05 图（design/harness/05-harness-loop-overview.puml）最小闭环。

对应 design/harness/05-minimal-loop-implementation-plan.md 的目标闭环：

```
Initialize
  -> ExecuteCurrentCommand
     [达到 max_loop_iterations 硬上限] -> FinalizeLoop.finalize_loop_limit -> CompleteTask
     [未达到上限] -> RegisterCommandResult
       -> EvaluateCurrentCommand
          succeeded -> AssessPresentation
             action=render -> AppendRenderCommand -> ExecuteCurrentCommand
             action=skip   -> CheckGoalCompletion
                is_completed=true  -> FinalizeLoop.finalize_completed -> CompleteTask
                is_completed=false -> DecideNextGoal
          failed -> DecideNextGoal
       DecideNextGoal
          next_command    -> AppendNextCommand -> ExecuteCurrentCommand
          no_next_command -> FinalizeLoop.finalize_fallback -> CompleteTask
```

本次最小闭环不实现的分支（计划文档"本次暂不实现"一节）：
``FallbackStrategy``（未调用工具的独立兜底分支——本实现里由
``RegisterCommandResult`` 直接把"未调用工具"登记为
``failed``/``no_tool_matched``，继续走 ``EvaluateCurrentCommand`` ->
``DecideNextGoal`` 的失败分支收敛，不单独拆节点）、
``CheckFailureThresholds``/``CheckSameGoalRepetition``（用
``max_loop_iterations`` 硬编码轮数上限临时替代）、``UpdateExecutingStatus``、
完整 ``RenderCommandInput``。

因为 ``ExecuteCurrentCommand`` 绑定的 MCP 工具需要按请求的 session_id
鉴权，``HarnessGraph`` 不能是进程级复用的单例，需要每个请求单独构造一次，
交由 ``HarnessGraphFactory`` 负责（见同目录 ``harness_graph_factory.py``）。

Langfuse 整合：``HarnessGraph`` 接收进程级共享的 ``LangfuseObservability``
（由 ``HarnessGraphFactory`` 转发，最终来自 ``ai_pad_bootstrap.
AiPadBootstrap``），``run()`` 用 ``observability.configure_run()`` 给本次
``ainvoke`` 的 ``config`` 挂上 callback，并用 ``observability.trace_run()``
包裹整次执行，写入 trace 顶层 input（``original_question``）。挂好的
``config`` 由 LangGraph 自动逐节点注入，各 LLM 节点方法（
``ExecuteCommand``/``EvaluateCurrentCommand``/``AssessPresentation``/
``CheckGoalCompletion``/``DecideNextGoal``）需要把它原样转发给自己的
``model.ainvoke``，否则 callback 传播链会在节点内部断开。08 图的
``StartHarness`` 调用本图时会把自己收到的 ``config`` 转发进来（见
``start_harness.py``），因此本图的 trace 是嵌套在 ``main_graph`` trace 之下
的独立 observation，不是另一个顶层 trace。当前不写 trace 顶层 output。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from config.observability import LangfuseObservability
from graph.complete_task import CompleteTask
from graph.harness_graph.append_next_command import AppendNextCommand
from graph.harness_graph.append_render_command import AppendRenderCommand
from graph.harness_graph.assess_presentation import ACTION_RENDER, AssessPresentation
from graph.harness_graph.check_goal_completion import CheckGoalCompletion
from graph.harness_graph.decide_next_goal import DecideNextGoal
from graph.harness_graph.evaluate_current_command import (
    EVALUATION_SUCCEEDED,
    EvaluateCurrentCommand,
)
from graph.harness_graph.execute_command import ExecuteCommand
from graph.harness_graph.finalize_loop import FinalizeLoop
from graph.harness_graph.harness_state import DEFAULT_MAX_LOOP_ITERATIONS, HarnessState
from graph.harness_graph.initialize import Initialize
from graph.harness_graph.register_command_result import RegisterCommandResult

_NODE_INITIALIZE = "initialize"
_NODE_EXECUTE_CURRENT_COMMAND = "execute_current_command"
_NODE_REGISTER_COMMAND_RESULT = "register_command_result"
_NODE_EVALUATE_CURRENT_COMMAND = "evaluate_current_command"
_NODE_ASSESS_PRESENTATION = "assess_presentation"
_NODE_APPEND_RENDER_COMMAND = "append_render_command"
_NODE_CHECK_GOAL_COMPLETION = "check_goal_completion"
_NODE_DECIDE_NEXT_GOAL = "decide_next_goal"
_NODE_APPEND_NEXT_COMMAND = "append_next_command"
_NODE_FINALIZE_COMPLETED = "finalize_completed"
_NODE_FINALIZE_FALLBACK = "finalize_fallback"
_NODE_FINALIZE_LOOP_LIMIT = "finalize_loop_limit"
_NODE_COMPLETE_TASK = "complete_task"


class HarnessGraph:
    """组装图05核心循环最小闭环：Initialize 到 CompleteTask 之间的完整状态机。"""

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool],
        *,
        max_loop_iterations: int = DEFAULT_MAX_LOOP_ITERATIONS,
        observability: LangfuseObservability | None = None,
    ):
        """构造 05 图节点。

        Args:
            model: 未绑定工具的基础聊天模型；``ExecuteCurrentCommand`` 会
                按需 ``bind_tools``，其余 LLM 节点直接复用未绑定工具的实例。
            tools: 供 ``ExecuteCurrentCommand`` 绑定的查询类工具（当前固定
                为 financial_query_data / fin_doc_searchV3）；渲染工具
                ``create_markdown_surface`` 由 ``ExecuteCommand`` 自行构造，
                不需要在这里传入。
            max_loop_iterations: 核心循环硬上限，默认见
                ``harness_state.DEFAULT_MAX_LOOP_ITERATIONS``。
            observability: 进程级共享的 Langfuse observability（由
                ``HarnessGraphFactory`` 转发）；为 None 时（或未在
                config.yaml 里启用）本图不挂任何 callback。
        """
        self._max_loop_iterations = max_loop_iterations
        self._observability = observability
        self._initialize = Initialize(max_loop_iterations)
        self._execute_command = ExecuteCommand(model, tools)
        self._register_command_result = RegisterCommandResult()
        self._evaluate_current_command = EvaluateCurrentCommand(model)
        self._assess_presentation = AssessPresentation(model)
        self._append_render_command = AppendRenderCommand()
        self._check_goal_completion = CheckGoalCompletion(model)
        self._decide_next_goal = DecideNextGoal(model)
        self._append_next_command = AppendNextCommand()
        self._finalize_loop = FinalizeLoop()
        self._complete_task = CompleteTask()
        self._graph = self._build_graph()

    @property
    def graph(self):
        return self._graph

    def _build_graph(self):
        graph = StateGraph(HarnessState)
        graph.add_node(_NODE_INITIALIZE, self._initialize.initialize)
        graph.add_node(_NODE_EXECUTE_CURRENT_COMMAND, self._execute_command.execute)
        graph.add_node(_NODE_REGISTER_COMMAND_RESULT, self._register_command_result.register)
        graph.add_node(
            _NODE_EVALUATE_CURRENT_COMMAND, self._evaluate_current_command.evaluate
        )
        graph.add_node(_NODE_ASSESS_PRESENTATION, self._assess_presentation.assess)
        graph.add_node(_NODE_APPEND_RENDER_COMMAND, self._append_render_command.append)
        graph.add_node(_NODE_CHECK_GOAL_COMPLETION, self._check_goal_completion.check)
        graph.add_node(_NODE_DECIDE_NEXT_GOAL, self._decide_next_goal.decide)
        graph.add_node(_NODE_APPEND_NEXT_COMMAND, self._append_next_command.append)
        graph.add_node(_NODE_FINALIZE_COMPLETED, self._finalize_loop.finalize_completed)
        graph.add_node(_NODE_FINALIZE_FALLBACK, self._finalize_loop.finalize_fallback)
        graph.add_node(_NODE_FINALIZE_LOOP_LIMIT, self._finalize_loop.finalize_loop_limit)
        graph.add_node(_NODE_COMPLETE_TASK, self._complete_task.complete)

        graph.add_edge(START, _NODE_INITIALIZE)
        graph.add_edge(_NODE_INITIALIZE, _NODE_EXECUTE_CURRENT_COMMAND)

        graph.add_conditional_edges(
            _NODE_EXECUTE_CURRENT_COMMAND,
            self._route_after_execute,
            {
                _NODE_REGISTER_COMMAND_RESULT: _NODE_REGISTER_COMMAND_RESULT,
                _NODE_FINALIZE_LOOP_LIMIT: _NODE_FINALIZE_LOOP_LIMIT,
            },
        )
        graph.add_edge(_NODE_REGISTER_COMMAND_RESULT, _NODE_EVALUATE_CURRENT_COMMAND)

        graph.add_conditional_edges(
            _NODE_EVALUATE_CURRENT_COMMAND,
            self._route_after_evaluate,
            {
                _NODE_ASSESS_PRESENTATION: _NODE_ASSESS_PRESENTATION,
                _NODE_DECIDE_NEXT_GOAL: _NODE_DECIDE_NEXT_GOAL,
            },
        )

        graph.add_conditional_edges(
            _NODE_ASSESS_PRESENTATION,
            self._route_after_assess_presentation,
            {
                _NODE_APPEND_RENDER_COMMAND: _NODE_APPEND_RENDER_COMMAND,
                _NODE_CHECK_GOAL_COMPLETION: _NODE_CHECK_GOAL_COMPLETION,
            },
        )
        graph.add_edge(_NODE_APPEND_RENDER_COMMAND, _NODE_EXECUTE_CURRENT_COMMAND)

        graph.add_conditional_edges(
            _NODE_CHECK_GOAL_COMPLETION,
            self._route_after_check_goal_completion,
            {
                _NODE_FINALIZE_COMPLETED: _NODE_FINALIZE_COMPLETED,
                _NODE_DECIDE_NEXT_GOAL: _NODE_DECIDE_NEXT_GOAL,
            },
        )

        graph.add_conditional_edges(
            _NODE_DECIDE_NEXT_GOAL,
            self._route_after_decide_next_goal,
            {
                _NODE_APPEND_NEXT_COMMAND: _NODE_APPEND_NEXT_COMMAND,
                _NODE_FINALIZE_FALLBACK: _NODE_FINALIZE_FALLBACK,
            },
        )
        graph.add_edge(_NODE_APPEND_NEXT_COMMAND, _NODE_EXECUTE_CURRENT_COMMAND)

        graph.add_edge(_NODE_FINALIZE_COMPLETED, _NODE_COMPLETE_TASK)
        graph.add_edge(_NODE_FINALIZE_FALLBACK, _NODE_COMPLETE_TASK)
        graph.add_edge(_NODE_FINALIZE_LOOP_LIMIT, _NODE_COMPLETE_TASK)
        graph.add_edge(_NODE_COMPLETE_TASK, END)
        return graph.compile()

    def _route_after_execute(self, state: HarnessState) -> str:
        loop_iterations = int(state.get("loop_iterations") or 0)
        max_loop_iterations = int(state.get("max_loop_iterations") or self._max_loop_iterations)
        if loop_iterations >= max_loop_iterations:
            return _NODE_FINALIZE_LOOP_LIMIT
        return _NODE_REGISTER_COMMAND_RESULT

    @staticmethod
    def _route_after_evaluate(state: HarnessState) -> str:
        if state.get("current_evaluation") == EVALUATION_SUCCEEDED:
            return _NODE_ASSESS_PRESENTATION
        return _NODE_DECIDE_NEXT_GOAL

    @staticmethod
    def _route_after_assess_presentation(state: HarnessState) -> str:
        if state.get("presentation_action") == ACTION_RENDER:
            return _NODE_APPEND_RENDER_COMMAND
        return _NODE_CHECK_GOAL_COMPLETION

    @staticmethod
    def _route_after_check_goal_completion(state: HarnessState) -> str:
        if state.get("is_goal_completed"):
            return _NODE_FINALIZE_COMPLETED
        return _NODE_DECIDE_NEXT_GOAL

    @staticmethod
    def _route_after_decide_next_goal(state: HarnessState) -> str:
        if state.get("next_command_decision"):
            return _NODE_APPEND_NEXT_COMMAND
        return _NODE_FINALIZE_FALLBACK

    async def run(
        self,
        original_question: str,
        *,
        task_id: str,
        context_id: str,
        updater: Any = None,
    ) -> HarnessState:
        """执行 05 图，返回最终状态（含 route，供 StartHarness 消费）。

        用 ``observability.configure_run()``/``trace_run()`` 挂 Langfuse
        callback 并写入 trace 顶层 input（``original_question``），与
        ``MainGraph.run`` 用同一个 ``session_id``（``context_id or
        task_id``），使本图的 trace 在 Langfuse 里与外层 ``main_graph``
        trace 按 session 关联展示。挂好的 ``config`` 由 LangGraph 逐节点
        自动注入到每个节点方法的 ``config`` 形参。
        """
        input_state: HarnessState = {
            "original_question": original_question,
            "task_id": task_id,
            "context_id": context_id,
            "updater": updater,
        }

        config: dict[str, Any] = {}
        trace_context = nullcontext()
        if self._observability is not None:
            instance_ip = self._observability.instance_ip
            tags = (instance_ip,) if instance_ip else ()
            config = self._observability.configure_run(
                config,
                run_name="harness_graph",
                session_id=context_id or task_id,
                tags=tags,
                metadata={"task_id": task_id},
            )
            trace_context = self._observability.trace_run(
                run_name="harness_graph",
                session_id=context_id or task_id,
                input=original_question,
                tags=tags,
                metadata={"task_id": task_id},
            )

        with trace_context:
            return await self._graph.ainvoke(input_state, config)

"""HarnessGraph 子包：05 图（design/harness/05-harness-loop-overview.puml）的实现。

对应 design/harness/05-minimal-loop-implementation-plan.md 的最小闭环：
``Initialize`` 创建初始状态，``ExecuteCurrentCommand`` 执行
``CommandList[-1]``，``RegisterCommandResult`` 登记结果，
``EvaluateCurrentCommand``/``AssessPresentation``/``CheckGoalCompletion``/
``DecideNextGoal`` 四个 LLM 节点分别负责业务评审、呈现判断、完成度判断与
下一目标决策，``AppendRenderCommand``/``AppendNextCommand`` 追加新的
pending 命令回到循环，``FinalizeLoop`` 把循环终止结果转换成
``CompleteTask``（``graph.complete_task`` 下的通用收尾节点）可消费的字段。

本次不实现的部分（图05完整设计里的 ``FallbackStrategy``/
``CheckFailureThresholds``/``CheckSameGoalRepetition``/
``UpdateExecutingStatus``、完整 ``RenderCommandInput``）见
``harness_graph.py`` 模块docstring 与
05-minimal-loop-implementation-plan.md。

08图的 ``StartHarness`` 节点是本图唯一的调用方，通过 ``HarnessGraphFactory``
按请求的 session_id 创建实例。

对外暴露 :class:`HarnessGraph` 及其协作组件。每个类一个文件、文件名与类名
（或职责）对应（Java 风格）。``CompleteTask`` 是通用节点，定义在
``graph.complete_task``，本包不再重复导出，直接从 ``graph.complete_task``
导入即可。
"""

from graph.harness_graph.append_next_command import AppendNextCommand
from graph.harness_graph.append_render_command import AppendRenderCommand
from graph.harness_graph.assess_presentation import AssessPresentation
from graph.harness_graph.check_goal_completion import CheckGoalCompletion
from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.create_markdown_surface import (
    MarkdownSurfaceRecorder,
    create_markdown_surface_tool,
)
from graph.harness_graph.decide_next_goal import DecideNextGoal
from graph.harness_graph.evaluate_current_command import EvaluateCurrentCommand
from graph.harness_graph.execute_command import ExecuteCommand
from graph.harness_graph.execute_command_tools import load_execute_command_tools
from graph.harness_graph.finalize_loop import FinalizeLoop
from graph.harness_graph.harness_graph import HarnessGraph
from graph.harness_graph.harness_graph_factory import HarnessGraphFactory
from graph.harness_graph.harness_state import HarnessState
from graph.harness_graph.initialize import Initialize
from graph.harness_graph.register_command_result import RegisterCommandResult

__all__ = [
    "AppendNextCommand",
    "AppendRenderCommand",
    "AssessPresentation",
    "CheckGoalCompletion",
    "CommandRecord",
    "MarkdownSurfaceRecorder",
    "create_markdown_surface_tool",
    "DecideNextGoal",
    "EvaluateCurrentCommand",
    "ExecuteCommand",
    "FinalizeLoop",
    "HarnessGraph",
    "HarnessGraphFactory",
    "HarnessState",
    "Initialize",
    "RegisterCommandResult",
    "load_execute_command_tools",
]

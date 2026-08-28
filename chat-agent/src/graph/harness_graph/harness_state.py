"""HarnessState（05 图状态）：图05最小闭环共享的字段。

对应 design/harness/02-harness-state-model.puml 的 ``HarnessState``，以及
design/harness/05-minimal-loop-implementation-plan.md 描述的最小闭环范围。
本次实现是简化版：

- 不实现 ``render_registry``（图01/02定义的 SQL -> surfaceId 映射）：本次
  渲染是 markdown 文本、不走 SQL 标识，计划文档"需要确认/决策的细节 3"
  已明确本次可以跳过。
- ``failure_streak_threshold`` / ``CheckFailureThresholds`` /
  ``CheckSameGoalRepetition`` 均不实现；用 ``max_loop_iterations`` 硬编码
  轮数上限临时替代所有终止判定，作为真正实现前的兜底（计划文档"本次暂不
  实现"一节）。

字段说明：

- ``original_question``：08 图 ``RewriteQuestion`` 产出、``StartHarness``
  写入的根问题，全链路只用这一个变量名。
- ``task_id`` / ``context_id``：A2A 任务身份信息。
- ``updater``：``a2a.server.tasks.TaskUpdater`` 实例，供 ``CompleteTask``
  推送 artifact/completed 状态。
- ``command_list``：按 ``sequence_no`` 只追加的 ``CommandRecord`` 列表，
  05图核心循环的唯一状态载体。任意时刻最多只有一条 ``pending`` 记录
  （必须是列表最后一项）。
- ``latest_a2ui_surface``：最近一次成功发布给客户端的完整 A2UI surface
  报文；render 命令成功后由 ``RegisterCommandResult`` 回写。
- ``max_loop_iterations``：核心循环硬上限（临时兜底，替代
  ``CheckFailureThresholds`` 的判定1）；达到后强制收尾为
  ``fallback_required``。
- ``loop_iterations``：已经执行过 ``ExecuteCurrentCommand`` 的轮数计数。
- ``last_final_answer`` / ``render_surface``：收尾节点 ``CompleteTask``
  读取的最终结果；``render_surface`` 沿用 ``CompleteTask`` 现有契约（推送
  为 artifact），本次闭环把 ``latest_a2ui_surface`` 的内容在收尾前同步
  过来。
- ``route``：``CompleteTask`` 产出的路由结果，交还给 08 图的
  ``StartHarness``。
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from graph.harness_graph.command_record import CommandRecord

#: CheckFailureThresholds 真正实现前的临时轮数上限（计划文档"细节 7"）。
DEFAULT_MAX_LOOP_ITERATIONS = 12


class HarnessState(TypedDict, total=False):
    """05 图核心循环共享的状态（最小闭环版本）。"""

    original_question: str
    task_id: str
    context_id: str
    updater: Any

    command_list: list[CommandRecord]
    latest_a2ui_surface: Any
    max_loop_iterations: int
    loop_iterations: int

    #: ExecuteCurrentCommand 本轮执行结果，供 RegisterCommandResult 消费。
    execution_outcome: dict[str, Any]
    #: EvaluateCurrentCommand 的评审结果。
    current_evaluation: str
    current_evaluation_reason: str
    #: AssessPresentation 的呈现决策。
    presentation_action: str
    presentation_rationale: str
    #: CheckGoalCompletion 的完成度判断。
    is_goal_completed: bool
    goal_completion_reason: str
    #: DecideNextGoal 的下一目标决策（None 表示 no_next_command）。
    next_command_decision: dict[str, Any] | None
    no_next_command_reason: str

    last_final_answer: str
    render_surface: Any
    route: str

"""RegisterCommandResult：把 ExecuteCurrentCommand 本轮执行结果登记到 CommandList[-1]。

对应 design/harness/05-harness-loop-overview.puml 的 ``RegisterCommandResult``
（最小闭环简化版）：

    RegisterCommandResult : 输入：HarnessState + 工具刚返回的 ToolMessage
    RegisterCommandResult : 读取 CommandList[-1]（必须为 pending）
    RegisterCommandResult : 输出：更新最后一项的 input、status、output、diagnostics
    RegisterCommandResult : render 成功时，同时回写 HarnessState.latest_a2ui_surface

简化点（对应计划文档"细节 2"）：``depends_on`` 固定为空列表（已在
``CommandRecord`` 创建时写入，不在这里改写）；``diagnostics`` 只填
``reason_code``/``message``，不强制 ``details``。

技术成功/失败判定规则：

- 未调用任何工具（``ExecuteCurrentCommand`` 未产生 ToolMessage）：本节点
  不负责这种情况的收尾（那是图05完整设计里 ``FallbackStrategy`` 的职责，
  本次最小闭环暂不实现该分支，直接判 ``failed``、
  ``reason_code=no_tool_matched``，交由后续 ``EvaluateCurrentCommand``/
  ``DecideNextGoal`` 走失败分支收敛，不新增独立的 FallbackStrategy 节点）。
- 调用了工具但工具结果 ``ok=False``：``failed``，
  ``reason_code=tool_execution_failed``。
- 调用了渲染工具且产出了 surface：``succeeded``，同时回写
  ``latest_a2ui_surface``。
- 调用了查询工具且 ``ok=True``：``succeeded``。
"""

from __future__ import annotations

from typing import Any

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.create_markdown_surface import TOOL_NAME as MARKDOWN_SURFACE_TOOL_NAME
from graph.harness_graph.harness_state import HarnessState

REASON_NO_TOOL_MATCHED = "no_tool_matched"
REASON_TOOL_EXECUTION_FAILED = "tool_execution_failed"


class RegisterCommandResult:
    """代码节点：把执行结果登记为 CommandList[-1] 的终态。"""

    async def register(self, state: HarnessState) -> dict[str, Any]:
        command_list = list(state.get("command_list") or [])
        if not command_list:
            raise ValueError("RegisterCommandResult requires a non-empty command_list")

        outcome = dict(state.get("execution_outcome") or {})
        current_command: CommandRecord = dict(command_list[-1])  # type: ignore[assignment]

        updated_command, latest_surface = self._resolve(current_command, outcome)
        command_list[-1] = updated_command

        result: dict[str, Any] = {"command_list": command_list}
        if latest_surface is not None:
            result["latest_a2ui_surface"] = latest_surface
        return result

    @staticmethod
    def _resolve(
        command: CommandRecord, outcome: dict[str, Any]
    ) -> tuple[CommandRecord, Any | None]:
        tool_called = bool(outcome.get("tool_called"))
        tool_name = outcome.get("tool_name")
        tool_result = dict(outcome.get("tool_result") or {})

        if not tool_called:
            command["status"] = "failed"
            command["output"] = None
            command["diagnostics"] = {
                "reason_code": REASON_NO_TOOL_MATCHED,
                "message": "本轮未调用任何工具。",
            }
            return command, None

        command["input"] = {"tool_name": tool_name}

        if not tool_result.get("ok", False):
            command["status"] = "failed"
            command["output"] = tool_result.get("data")
            command["diagnostics"] = {
                "reason_code": REASON_TOOL_EXECUTION_FAILED,
                "message": str(tool_result.get("error_message") or "工具执行失败。"),
            }
            return command, None

        command["status"] = "succeeded"
        command["output"] = tool_result.get("data")
        command.pop("diagnostics", None)

        latest_surface = None
        if tool_name == MARKDOWN_SURFACE_TOOL_NAME:
            latest_surface = outcome.get("rendered_surface")

        return command, latest_surface

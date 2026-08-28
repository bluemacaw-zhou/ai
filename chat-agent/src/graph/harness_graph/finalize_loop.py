"""FinalizeLoop：把核心循环的终止结果转换成 CompleteTask 可消费的字段。

``graph.complete_task.CompleteTask``（05图复用的通用收尾节点）只依据
``last_final_answer`` 是否非空判定 route（有则 ``normal_completed``，否则
``fallback_required``），并在 ``render_surface`` 非空时推送 artifact。

05 图核心循环的多个终止分支（正常完成 / no_tool_matched / 达到轮数硬上限 /
goal_unreachable）各自产出不同的诊断信息，需要在进入 ``CompleteTask`` 前
统一转换成它认识的字段：

- 正常完成（``CheckGoalCompletion`` 判定 ``is_completed=true``）：如果
  ``latest_a2ui_surface`` 已有内容，优先把它作为 ``render_surface``
  推送（渲染场景下 completed 文本留空，由 ``CompleteTask`` 决定）；否则
  用最后一条成功命令的 output 文本作为 ``last_final_answer``。
- 降级（未调用工具 / 达到轮数硬上限 / 无法达成根目标）：把诊断信息拼成
  面向用户的说明文本，写入 ``last_final_answer``，不推送 render_surface
  （除非 ``latest_a2ui_surface`` 已有内容，此时仍一并带出，保留此前已呈现
  的部分结果）。
"""

from __future__ import annotations

from typing import Any
import json

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.harness_state import HarnessState

REASON_MAX_LOOP_ITERATIONS_EXCEEDED = "max_loop_iterations_exceeded"
REASON_GOAL_UNREACHABLE = "goal_unreachable"


class FinalizeLoop:
    """代码节点：把核心循环终止时的状态转换成 CompleteTask 的输入字段。"""

    async def finalize_completed(self, state: HarnessState) -> dict[str, Any]:
        """CheckGoalCompletion 判定 is_completed=true 时的收尾。"""
        latest_surface = state.get("latest_a2ui_surface")
        result: dict[str, Any] = {}
        if latest_surface is not None:
            result["render_surface"] = [latest_surface]
            result["last_final_answer"] = "已完成。"
        else:
            command_list = list(state.get("command_list") or [])
            result["last_final_answer"] = self._latest_output_text(command_list) or "已完成。"
        return result

    async def finalize_fallback(self, state: HarnessState) -> dict[str, Any]:
        """无法正常完成时的通用降级收尾：把诊断信息转成 last_final_answer。"""
        command_list = list(state.get("command_list") or [])
        latest_command = command_list[-1] if command_list else None
        diagnostics = (latest_command or {}).get("diagnostics") or {}
        reason_code = str(diagnostics.get("reason_code") or "")
        message = str(diagnostics.get("message") or "")

        no_next_command_reason = str(state.get("no_next_command_reason") or "")
        if not reason_code and no_next_command_reason:
            reason_code = REASON_GOAL_UNREACHABLE
            message = no_next_command_reason

        fallback_text = self._compose_fallback_text(reason_code, message)

        result: dict[str, Any] = {"last_final_answer": fallback_text}
        latest_surface = state.get("latest_a2ui_surface")
        if latest_surface is not None:
            result["render_surface"] = [latest_surface]
        return result

    async def finalize_loop_limit(self, state: HarnessState) -> dict[str, Any]:
        """达到 max_loop_iterations 硬上限时的强制降级收尾。"""
        fallback_text = self._compose_fallback_text(
            REASON_MAX_LOOP_ITERATIONS_EXCEEDED,
            "已达到核心循环轮数上限，任务被强制终止。",
        )
        result: dict[str, Any] = {"last_final_answer": fallback_text}
        latest_surface = state.get("latest_a2ui_surface")
        if latest_surface is not None:
            result["render_surface"] = [latest_surface]
        return result

    @staticmethod
    def _latest_output_text(command_list: list[CommandRecord]) -> str:
        for record in reversed(command_list):
            if record.get("status") == "succeeded" and record.get("output") is not None:
                output = record["output"]
                return output if isinstance(output, str) else json.dumps(
                    output, ensure_ascii=False
                )
        return ""

    @staticmethod
    def _compose_fallback_text(reason_code: str, message: str) -> str:
        if not reason_code and not message:
            return "任务未能完成，请稍后重试或换一种方式提问。"
        return f"任务未能完成（{reason_code or 'unknown'}）：{message}"

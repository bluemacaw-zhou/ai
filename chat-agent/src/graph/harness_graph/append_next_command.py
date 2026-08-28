"""AppendNextCommand：把 DecideNextGoal 产出的 next_command 追加为新的 pending 命令。

对应 design/harness/05-harness-loop-overview.puml 的 ``AppendNextCommand``：

    AppendNextCommand : 输入：next_command(kind, requirement, plan)
    AppendNextCommand : 输出：追加 CommandRecord(status=pending)，成为 CommandList[-1]
    AppendNextCommand : 补齐 task_id、original_question、sequence_no 等固定字段
"""

from __future__ import annotations

from typing import Any

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.harness_state import HarnessState


class AppendNextCommand:
    """代码节点：把 next_command_decision 转成新的 pending CommandRecord。"""

    async def append(self, state: HarnessState) -> dict[str, Any]:
        command_list = list(state.get("command_list") or [])
        next_command = state.get("next_command_decision")
        if not isinstance(next_command, dict):
            raise ValueError("AppendNextCommand requires a next_command_decision")

        task_id = str(state.get("task_id") or "")
        original_question = str(state.get("original_question") or "")

        record: CommandRecord = {
            "task_id": task_id,
            "original_question": original_question,
            "sequence_no": len(command_list),
            "kind": next_command.get("kind"),
            "requirement": next_command.get("requirement", ""),
            "input": {"plan": next_command.get("plan", "")},
            "depends_on": [],
            "status": "pending",
            "output": None,
        }

        return {"command_list": [*command_list, record]}

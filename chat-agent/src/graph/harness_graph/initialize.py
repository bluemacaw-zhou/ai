"""Initialize：05 图核心循环的入口代码节点。

对应 design/harness/05-harness-loop-overview.puml 的 ``Initialize``：

    Initialize : 代码节点
    Initialize : 输入：A2A task_id、用户 original_question
    Initialize : 输出：HarnessState + CommandRecord(kind=root, status=pending)

本次最小闭环实现的简化：``max_command_list_length``/
``failure_streak_threshold`` 对应的 ``CheckFailureThresholds`` 尚未落地，
改为写入 ``max_loop_iterations``（见 ``harness_state.py``
``DEFAULT_MAX_LOOP_ITERATIONS``）作为临时轮数硬上限。
"""

from __future__ import annotations

from typing import Any

from graph.harness_graph.command_record import CommandRecord
from graph.harness_graph.harness_state import DEFAULT_MAX_LOOP_ITERATIONS, HarnessState

ROOT_SEQUENCE_NO = 0


class Initialize:
    """代码节点：创建初始 HarnessState 字段与 root CommandRecord。"""

    def __init__(self, max_loop_iterations: int = DEFAULT_MAX_LOOP_ITERATIONS) -> None:
        self._max_loop_iterations = max_loop_iterations

    async def initialize(self, state: HarnessState) -> dict[str, Any]:
        """创建 root CommandRecord(kind=root, status=pending) 并初始化循环计数。"""
        original_question = str(state.get("original_question") or "")
        task_id = str(state.get("task_id") or "")

        root_command: CommandRecord = {
            "task_id": task_id,
            "original_question": original_question,
            "sequence_no": ROOT_SEQUENCE_NO,
            "kind": "root",
            "requirement": original_question,
            "input": None,
            "depends_on": [],
            "status": "pending",
            "output": None,
        }

        return {
            "command_list": [root_command],
            "latest_a2ui_surface": None,
            "max_loop_iterations": self._max_loop_iterations,
            "loop_iterations": 0,
        }

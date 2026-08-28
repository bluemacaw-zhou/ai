"""CommandRecord：05 图 CommandList 中单条命令的记录结构。

对应 design/harness/02-harness-state-model.puml 的 ``CommandRecord`` 类。当前
是本次最小闭环实现，只落地实现用得到的字段：``depends_on`` 先固定为空
列表，``diagnostics`` 先只填 ``reason_code``/``message``，不强制 ``details``
（计划文档"需要确认/决策的细节 2"里明确允许的简化）。

字段含义：

- ``task_id`` / ``original_question``：冗余自 HarnessState，确保单条命令
  离开列表上下文也能独立理解。
- ``sequence_no``：同一 task 内的命令唯一标识与回放顺序，从 0 开始。
- ``kind``：``root`` | ``query`` | ``calculate`` | ``render``。当前阶段只
  是语义记录，不作为 LangGraph 路由条件（``ExecuteCurrentCommand`` 统一
  处理最后一条 pending 命令）。
- ``requirement``：本条命令要完成的业务目标，供 ``ExecuteCurrentCommand``
  与各 LLM 评审节点读取。
- ``input``：工具实际入参；render 命令的 input 是简化版渲染内容（见
  ``AppendRenderCommand``），不是图01定义的完整 ``RenderCommandInput``。
- ``depends_on``：从 input 派生的依赖索引，本次实现固定为空列表。
- ``status``：``pending`` | ``succeeded`` | ``failed``。
- ``output``：命令正常执行后的输出；pending 时为空。
- ``diagnostics``：终态诊断信息，只有失败时才有意义。
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

CommandKind = Literal["root", "query", "calculate", "render"]
CommandStatus = Literal["pending", "succeeded", "failed"]


class CommandDiagnostics(TypedDict, total=False):
    """终态命令的诊断信息（简化版：先只要求 reason_code/message）。"""

    reason_code: str
    message: str
    details: Any


class CommandRecord(TypedDict, total=False):
    """CommandList 中的单条命令记录。"""

    task_id: str
    original_question: str
    sequence_no: int
    kind: CommandKind
    requirement: str
    input: Any
    depends_on: list[str]
    status: CommandStatus
    output: Any
    diagnostics: CommandDiagnostics

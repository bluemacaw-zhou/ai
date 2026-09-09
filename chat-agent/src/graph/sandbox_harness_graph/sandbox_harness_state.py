"""SandboxAgent 的 A2A 请求与 Harness 内部状态。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


DEFAULT_MAX_SANDBOX_ATTEMPTS = 4


class SandboxAgentRequest(TypedDict, total=False):
    """完整 A2A 请求；调用方只构造这一种结构。"""

    # 必填：本次局部计算要完成的目标。
    objective: str
    # 可选：希望如何呈现产物。
    expected_output: str
    # 可选：结果验收条件。
    success_criteria: list[str]
    # 可选：单位、时间范围、精度等约束。
    constraints: list[str]
    # 必填：输入 JSON 数据的 MinIO 地址，格式为 s3://bucket/object。
    input_uris: list[str]


class SandboxHarnessState(SandboxAgentRequest, total=False):
    """LangGraph 内部状态：请求字段加上本次 Harness 循环产生的内容。"""

    # 从输入 JSON 信封提取的 metadata，供模型理解数据；不保存 data。
    data_metadata: list[dict[str, Any]]
    # 本轮模型刚生成、尚未提交 Judge0 的代码。
    pending_code: str
    # pending_code 使用的语言；当前默认 python。
    pending_language: str
    # 模型上下文：初始约束，以及每版代码和对应的执行结果。
    messages: list[BaseMessage]
    # 每轮的代码、语言、Judge0 原始结果和解析后的输出/错误。
    attempts: list[dict[str, Any]]
    # 服务端限制的最大尝试次数，防止修复循环无限运行。
    max_attempts: int

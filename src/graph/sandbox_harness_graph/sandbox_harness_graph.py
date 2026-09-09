"""SandboxAgent：统一 JSON/MinIO 数据上的执行状态驱动 Harness。"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from config.judge0_client import Judge0Client
from config.minio_client import MinioClient
from graph.sandbox_harness_graph.execute_in_judge0 import ExecuteInJudge0
from graph.sandbox_harness_graph.generate_code import GenerateCode
from graph.sandbox_harness_graph.initialize import Initialize
from graph.sandbox_harness_graph.load_data_metadata import LoadDataMetadata
from graph.sandbox_harness_graph.publish_result import PublishResult
from graph.sandbox_harness_graph.sandbox_harness_state import (
    DEFAULT_MAX_SANDBOX_ATTEMPTS,
    SandboxAgentRequest,
    SandboxHarnessState,
)


class SandboxHarnessGraph:
    """负责数据计算的 Sandbox Harness 图。

    设计思路：
    1. 上游只传 ``SandboxAgentRequest``：局部计算目标和一组 MinIO JSON URI；
       不传聊天历史、数据内容、代码或预签名 URL。
    2. Agent 先从每个 JSON 信封提取 ``metadata``，仅将这些元数据和目标交给模型；
       ``data`` 始终留在 MinIO，避免占用模型上下文。
    3. 模型的职责仅是生成或修复代码。Judge0 每执行一次，就把代码版本、stdout、stderr
       和状态写入 ``attempts``；是否重试由程序根据执行状态决定，而非由模型输出 action/final。
    4. 成功产物或失败诊断都由 Agent 服务端封装为统一 JSON 并上传 MinIO；A2A 响应只返回
       ``result_uri``。Judge0 不持有 MinIO 凭证。
    """

    def __init__(
        self,
        model: Any,
        judge0: Judge0Client | None = None,
        minio: MinioClient | None = None,
        *,
        max_attempts: int = DEFAULT_MAX_SANDBOX_ATTEMPTS,
    ) -> None:
        self._max_attempts = max_attempts
        shared_minio = minio
        self._load_metadata = LoadDataMetadata(shared_minio)
        self._initialize = Initialize()
        self._generate_code = GenerateCode(model)
        self._execute = ExecuteInJudge0(judge0 or Judge0Client(), shared_minio)
        self._publish = PublishResult(shared_minio)
        self._graph = self._build_graph()

    @property
    def graph(self):
        return self._graph

    def _build_graph(self):
        workflow = StateGraph(SandboxHarnessState)
        # 1. 从每个 JSON 输入中提取 metadata，data 不进入提示词。
        workflow.add_node("load_data_metadata", self._load_metadata.load)
        # 2. 初始化代码生成所需的请求与 metadata 上下文。
        workflow.add_node("initialize", self._initialize.initialize)
        # 3. 模型只生成当前版本代码。
        workflow.add_node("generate_code", self._generate_code.generate)
        # 4. 执行代码，并把“代码版本 + Judge0 结果”追加到 attempts。
        workflow.add_node("execute_in_judge0", self._execute.execute)
        # 5. 服务端将成功结果或失败诊断写成统一 JSON 并上传 MinIO。
        workflow.add_node("publish_success", self._publish.publish_success)
        workflow.add_node("publish_failure", self._publish.publish_failure)

        workflow.add_edge(START, "load_data_metadata")
        workflow.add_edge("load_data_metadata", "initialize")
        workflow.add_edge("initialize", "generate_code")
        workflow.add_edge("generate_code", "execute_in_judge0")
        workflow.add_conditional_edges(
            "execute_in_judge0",
            self._route_after_execute,
            {"retry": "generate_code", "success": "publish_success", "failure": "publish_failure"},
        )
        workflow.add_edge("publish_success", END)
        workflow.add_edge("publish_failure", END)
        return workflow.compile()

    @staticmethod
    def _route_after_execute(state: SandboxHarnessState) -> Literal["retry", "success", "failure"]:
        attempts = state.get("attempts") or []
        if attempts and attempts[-1].get("outcome") == "succeeded":
            return "success"
        if len(attempts) >= int(state.get("max_attempts") or DEFAULT_MAX_SANDBOX_ATTEMPTS):
            return "failure"
        return "retry"

    async def run(self, request: SandboxAgentRequest) -> dict[str, str]:
        """执行请求，最终仅返回保存统一结果 JSON 的 MinIO URI。"""
        if not str(request.get("objective") or "").strip():
            raise ValueError("request.objective 不能为空")
        if not request.get("input_uris"):
            raise ValueError("request.input_uris 不能为空")
        state = await self._graph.ainvoke(
            {**request, "max_attempts": self._max_attempts},
            {"recursion_limit": self._max_attempts * 3 + 8},
        )
        attempts = state.get("attempts") or []
        result_uri = attempts[-1].get("result_uri") if attempts else None
        if not result_uri:
            raise RuntimeError("SandboxAgent 未能发布结果到 MinIO")
        return {"result_uri": str(result_uri)}

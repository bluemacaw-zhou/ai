import json

from langchain_core.messages import HumanMessage, SystemMessage

from graph.sandbox_harness_graph.sandbox_harness_state import SandboxHarnessState
from utils.prompt_loader import load_prompt


class Initialize:
    """节点 2：将请求和已加载的 metadata 组织为第一轮代码生成上下文。"""

    async def initialize(self, state: SandboxHarnessState) -> dict:
        request_summary = {
            "objective": state.get("objective"),
            "expected_output": state.get("expected_output"),
            "success_criteria": state.get("success_criteria") or [],
            "constraints": state.get("constraints") or [],
        }
        return {
            "messages": [
                SystemMessage(load_prompt("sandbox_harness_generate_code")),
                HumanMessage("委托请求：\n" + json.dumps(request_summary, ensure_ascii=False)),
                HumanMessage(
                    "输入数据 metadata（不含 data）：\n"
                    + json.dumps(state.get("data_metadata") or [], ensure_ascii=False)
                ),
            ],
            "attempts": [],
        }

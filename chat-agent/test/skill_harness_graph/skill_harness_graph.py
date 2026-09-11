"""仿照 Wind.Weaver.ChatAgent 的 agent -> tools -> agent Skill Harness 测试图。"""

from typing import Literal

from config.chat_model import ChatModelRegistry
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from skill_harness_graph.call_agent import CallAgent
from skill_harness_graph.finalize_loop import FinalizeLoop
from skill_harness_graph.initialize import Initialize
from skill_harness_graph.skill_harness_state import SkillHarnessState
from utils.skill_runtime import LocalSkillRuntime, inject_skills_meta


class SkillHarnessGraph:
    """模型从 SKILLS_META 选择并逐步加载 Skill，再按 Skill 指示完成问题。

    流程为：系统提示词提供元数据 -> 模型调用 get_skill -> 按需调用 get_script /
    get_reference -> ToolNode 返回 ToolMessage -> 模型根据新上下文继续推理。图只维持这个循环和安全上限，
    不预设选中哪个 Skill，也不自动执行 Skill 中的脚本。
    """

    def __init__(self, model=None, max_iterations: int = 8) -> None:
        self._max_iterations = max_iterations
        self._runtime = LocalSkillRuntime()
        self._model = model or ChatModelRegistry.instance().model("default")
        self._skill_tools = self._runtime.create_tool_metadata()
        self._initialize = Initialize()
        self._agent = CallAgent(self._model, self._skill_tools)
        self._finalize = FinalizeLoop()
        self.graph = self._build_graph()

    def _build_graph(self):
        """连接初始化、模型决策、工具执行和循环收敛节点。"""
        workflow = StateGraph(SkillHarnessState)
        workflow.add_node("initialize", self._initialize.run)  # 注入 SKILLS_META 和用户问题。
        workflow.add_node("agent", self._agent.run)  # 模型按元数据或已加载内容决定下一步。
        workflow.add_node("tools", ToolNode(self._skill_tools))  # 执行三个加载工具之一，并自动写回 ToolMessage。
        workflow.add_node("loop_limit", self._finalize.run)  # 达到轮数上限时输出兜底答案。
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "agent")
        workflow.add_conditional_edges("agent", self._route, {"tools": "tools", "loop_limit": "loop_limit", END: END})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("loop_limit", END)
        return workflow.compile()

    def _route(self, state: SkillHarnessState) -> Literal["tools", "loop_limit", "__end__"]:
        """模型无工具调用则结束；有调用且未超限则进入工具节点。"""
        if not state["messages"][-1].tool_calls:
            return END
        if int(state["loop_iterations"]) >= self._max_iterations:
            return "loop_limit"
        return "tools"

    async def ainvoke(self, question: str) -> dict:
        """先执行 middleware 的 before_agent，再异步运行 agent/tool 循环图。"""
        state = inject_skills_meta(self._runtime, {"question": question})
        state.update({"question": question, "max_loop_iterations": self._max_iterations})
        return await self.graph.ainvoke(state, {"recursion_limit": self._max_iterations * 3 + 4})

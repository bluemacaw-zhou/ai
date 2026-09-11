"""模型调用节点。"""

from typing import Any

from skill_harness_graph.skill_harness_state import SkillHarnessState


class CallAgent:
    """模型调用节点：绑定三个 Skill 加载工具，并由模型自行决定下一步。"""

    def __init__(self, model: Any, skill_tools: Any) -> None:
        self._model = model
        self._skill_tools = skill_tools

    async def run(self, state: SkillHarnessState) -> dict[str, Any]:
        """模型有 tool_calls 则交由工具节点；否则保存最终答案。

        ``bind_tools`` 绑定的是 get_skill、get_script、get_reference 的函数签名，
        不是任意一个 Skill。模型看到 SKILLS_META 后先决定是否调用 get_skill，
        再依据 SKILL.md 决定继续读取脚本、引用资料，还是直接回答。
        """
        model_with_tools = self._model.bind_tools(self._skill_tools)
        response = await model_with_tools.ainvoke(state["messages"])
        result: dict[str, Any] = {"messages": [response], "loop_iterations": int(state.get("loop_iterations", 0)) + 1}
        if not response.tool_calls:
            result["final_answer"] = str(response.content)
        return result

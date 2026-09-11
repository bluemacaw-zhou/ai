"""循环收敛节点。"""

from langchain_core.messages import AIMessage

from skill_harness_graph.skill_harness_state import SkillHarnessState


class FinalizeLoop:
    """循环收敛节点：超过最大模型调用次数后输出兜底结果。"""

    def run(self, _: SkillHarnessState) -> dict:
        """阻止异常工具链无限循环。"""
        answer = "Skill 调用轮数已达到上限。"
        return {"messages": [AIMessage(answer)], "final_answer": answer}

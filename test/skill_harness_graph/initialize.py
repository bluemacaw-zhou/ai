"""初始化节点。"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from skill_harness_graph.skill_harness_state import SkillHarnessState
from utils.prompt_loader import load_prompt


class Initialize:
    """初始化节点：将 middleware 注入的 SKILLS_META 渲染进系统提示词。

    系统提示词只承担 Skill 发现，不承载完整 SKILL.md、脚本和引用文件。
    """

    def run(self, state: SkillHarnessState) -> dict[str, Any]:
        """创建首轮消息；此时模型只看到元数据，看不到任一 Skill 正文。"""
        return {"messages": [SystemMessage(load_prompt("skill_harness_system", **state["system_prompt_params"])), HumanMessage(state["question"])], "loop_iterations": 0}

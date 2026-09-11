"""使用真实模型验证分层 Skill 的完整加载与推理链路。"""

import asyncio
import unittest

from langchain_core.messages import ToolMessage

from skill_harness_graph.skill_harness_graph import SkillHarnessGraph


REAL_USER_QUESTION = "请把下面的产品公告润色得更专业、简洁：\n 我们的 新版本  已经上线， 支持   批量导出和权限管理。欢迎大家使用！ "


class LayeredSkillHarnessTests(unittest.TestCase):
    """用一个真实、可复现的问题验证完整流程。"""

    def test_polish_announcement(self) -> None:
        """验证真实模型完成“公告工作流 → 文本润色 → 脚本规则”的分层推理。

        输入是包含多余空白、口语化表达的产品公告。模型预期先依据元数据选择
        announcement-workflow，再依照其正文加载 text-polisher，最后读取规范化脚本。
        最终答案应保留“新版本、批量导出、权限管理”等事实，但将原文加工为简洁、
        专业且可直接发布的公告文本；工具轨迹必须包含 get_skill 和 get_script。
        """
        result = asyncio.run(SkillHarnessGraph().ainvoke(REAL_USER_QUESTION))

        self.assertTrue(result["final_answer"])
        print(f"\n润色结果：\n{result['final_answer']}")
        tool_names = [message.name for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertIn("get_skill", tool_names)
        self.assertIn("get_script", tool_names)

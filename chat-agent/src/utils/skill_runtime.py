"""本地 Skill 的元数据发现、激活与按需加载。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from langchain_core.tools import StructuredTool


SKILLS_META_KEY = "SKILLS_META"
SKILL_FILE_NAME = "SKILL.md"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class SkillMeta:
    """首轮注入系统提示词的轻量 Skill 元数据。"""

    name: str
    description: str
    when_to_use: str

    def format_for_prompt(self) -> str:
        return f'"{self.name}": {self.description} - {self.when_to_use}'


class LocalSkillRuntime:
    """固定从 ``src/skills`` 发现和加载 Skill。

    对外只暴露两类能力：``load_skill_meta`` 用于发现阶段注入系统提示词，
    ``get_skill`` 用于模型按需读取正文或资源。``get_skill`` 的返回值会由
    LangGraph ToolNode 自动写成 ToolMessage，供下一轮模型推理使用。
    """

    def __init__(self) -> None:
        self._metas: list[SkillMeta] = []
        self._active_names: set[str] = set()

    def load_skill_meta(self) -> str:
        """加载 Skill frontmatter，并返回系统提示词使用的轻量元数据。

        此方法只读取名称、说明和触发条件，完整 SKILL.md、脚本与引用资料均不
        在发现阶段进入上下文。
        """
        # 每次新的图调用都从未激活状态开始，避免上一次会话的激活范围泄漏到下一次。
        self._active_names.clear()
        self._metas = self._load_metas()
        return "\n\n".join(meta.format_for_prompt() for meta in self._metas)

    def get_skill(self, name: str) -> str:
        """加载并激活指定 Skill 的 ``SKILL.md`` 正文。"""
        self._ensure_known(name)
        content = self._read_resource(name, SKILL_FILE_NAME)
        self._active_names.add(name)
        return content

    def get_script(self, name: str, file_path: str) -> str:
        """读取已激活 Skill 的 ``scripts/`` 下的一个脚本。"""
        return self._get_active_resource(name, "scripts", file_path)

    def get_reference(self, name: str, file_path: str) -> str:
        """读取已激活 Skill 的 ``references/`` 下的一份引用资料。"""
        return self._get_active_resource(name, "references", file_path)

    def create_tool_metadata(self) -> Sequence[StructuredTool]:
        """创建模型可调用的三项分层加载工具元数据及其执行函数。"""
        async def get_skill(name: str) -> str:
            return self.get_skill(name)

        async def get_script(name: str, file_path: str) -> str:
            return self.get_script(name, file_path)

        async def get_reference(name: str, file_path: str) -> str:
            return self.get_reference(name, file_path)

        return (
            StructuredTool.from_function(
                coroutine=get_skill,
                name="get_skill",
                description="加载并激活指定 Skill 的 SKILL.md 正文；必须先调用它，才能读取该 Skill 的脚本或引用资料。",
            ),
            StructuredTool.from_function(
                coroutine=get_script,
                name="get_script",
                description="读取已激活 Skill 的 scripts/ 目录下的一个脚本；file_path 是相对 scripts/ 的路径。",
            ),
            StructuredTool.from_function(
                coroutine=get_reference,
                name="get_reference",
                description="读取已激活 Skill 的 references/ 目录下的一份资料；file_path 是相对 references/ 的路径。",
            ),
        )

    def _get_active_resource(self, name: str, directory: str, file_path: str) -> str:
        self._ensure_known(name)
        if name not in self._active_names:
            raise ValueError(f"Skill 尚未激活，不能读取其资源：{name}")
        return self._read_resource(name, f"{directory}/{file_path}")

    def _ensure_known(self, name: str) -> None:
        if name not in {meta.name for meta in self._metas}:
            raise ValueError(f"未知 Skill：{name}")

    @staticmethod
    def _read_resource(name: str, relative_path: str) -> str:
        root = (SKILLS_DIR / name).resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents:
            raise ValueError("资源路径不能离开 Skill 目录。")
        content = target.read_text(encoding="utf-8")
        if target.relative_to(root).as_posix() == SKILL_FILE_NAME:
            content = LocalSkillRuntime._split(content)[1]
        return content

    @staticmethod
    def _load_metas() -> list[SkillMeta]:
        metas: list[SkillMeta] = []
        for file in sorted(SKILLS_DIR.glob(f"*/{SKILL_FILE_NAME}")):
            frontmatter, _ = LocalSkillRuntime._split(file.read_text(encoding="utf-8"))
            metas.append(SkillMeta(file.parent.name, str(frontmatter["description"]), str(frontmatter["when_to_use"])))
        return metas

    @staticmethod
    def _split(content: str) -> tuple[dict[str, Any], str]:
        _, frontmatter, body = content.split("---", 2)
        return yaml.safe_load(frontmatter), body.strip()


def inject_skills_meta(runtime: LocalSkillRuntime, state: Mapping[str, Any]) -> dict[str, Any]:
    """将 SKILLS_META 注入系统提示词；完整 Skill 仍等待模型主动加载。"""
    params = dict(state.get("system_prompt_params") or {})
    params[SKILLS_META_KEY] = runtime.load_skill_meta()
    return {"system_prompt_params": params}

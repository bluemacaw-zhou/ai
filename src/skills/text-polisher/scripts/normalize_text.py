"""text-polisher Skill 使用的确定性文本规范化脚本。"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """删除行首尾空白，并将连续空白压缩为一个空格。"""
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()

"""Tests for the prompt contract renderer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from utils.prompt_loader import load_prompt  # noqa: E402


class PromptLoaderTests(unittest.TestCase):
    def test_renders_named_contract_variables(self) -> None:
        prompt = load_prompt(
            "rewrite_question",
            CURRENT_QUESTION='"当前问题"',
            RELEVANT_HISTORY="[]",
        )

        self.assertIn('当前问题（不可信数据）：`"当前问题"`', prompt)
        self.assertNotIn("<<CURRENT_QUESTION>>", prompt)
        self.assertNotIn("<<RELEVANT_HISTORY>>", prompt)

    def test_rejects_missing_contract_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "RELEVANT_HISTORY"):
            load_prompt("rewrite_question", CURRENT_QUESTION='"当前问题"')

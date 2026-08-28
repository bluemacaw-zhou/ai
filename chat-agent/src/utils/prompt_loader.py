from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re


_PLACEHOLDER_PATTERN = re.compile(r"<<([A-Z][A-Z0-9_]*)>>")


@lru_cache(maxsize=None)
def load_prompt(name: str, **variables: object) -> str:
    """Load a prompt and render its explicit ``<<UPPER_CASE>>`` placeholders.

    Prompt files own the instruction contract; callers only provide the named
    runtime data required by that contract.  Failing on a missing placeholder
    prevents a node from silently sending an incomplete prompt to the model.
    """
    prompt_path = Path(__file__).resolve().parent.parent / 'prompts' / f'{name}.md'
    template = prompt_path.read_text(encoding='utf-8').strip()
    placeholders = set(_PLACEHOLDER_PATTERN.findall(template))
    missing = placeholders - variables.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Prompt '{name}' is missing variables: {names}")

    return _PLACEHOLDER_PATTERN.sub(
        lambda match: str(variables[match.group(1)]), template
    )

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


class PromptLoader:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = prompt_dir

    @lru_cache(maxsize=64)
    def _load_prompt_file(self, relative_path: str) -> str:
        prompt_path = self.prompt_dir / relative_path
        return prompt_path.read_text(encoding="utf-8")

    def resolve_prompt(self, prompt_ref: str, context: dict[str, Any]) -> tuple[str, str]:
        normalized = {key: self._stringify(value) for key, value in context.items()}
        if prompt_ref.startswith("file:"):
            relative_path = prompt_ref[5:].strip()
            template = self._load_prompt_file(relative_path)
            return prompt_ref, template.format_map(_SafeFormatDict(normalized)).strip()
        return "inline", prompt_ref.format_map(_SafeFormatDict(normalized)).strip()

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(value, indent=2, ensure_ascii=True)

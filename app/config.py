from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    project_name: str = "AI Discussion Studio"
    project_description: str = "Run participant-centric, turn-based AI discussions from editable JSON configuration."
    base_dir: Path
    prompt_dir: Path
    template_dir: Path
    static_dir: Path
    config_dir: Path
    session_file: Path
    recent_session_limit: int = Field(default=20, ge=1, le=200)
    request_timeout_seconds: float = Field(default=60.0, gt=1.0, le=300.0)
    default_max_turn_cycles: int = Field(default=2, ge=1, le=8)
    default_max_output_tokens: int = Field(default=700, ge=128, le=12000)
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base_dir = Path(__file__).resolve().parent.parent
        return cls(
            base_dir=base_dir,
            prompt_dir=base_dir / "prompts",
            template_dir=base_dir / "app" / "templates",
            static_dir=base_dir / "app" / "static",
            config_dir=base_dir / "config",
            session_file=base_dir / os.getenv("SESSION_FILE", "data/sessions.json"),
            recent_session_limit=int(os.getenv("RECENT_SESSION_LIMIT", "20")),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
            default_max_turn_cycles=int(os.getenv("DEFAULT_MAX_TURN_CYCLES", os.getenv("DEFAULT_MAX_ROUNDS", "2"))),
            default_max_output_tokens=int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "700")),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.3")),
        )

    def ensure_directories(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def base_template_context(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_description": self.project_description,
            "default_settings": {
                "max_turn_cycles": self.default_max_turn_cycles,
                "max_output_tokens": self.default_max_output_tokens,
                "temperature": self.default_temperature,
            },
        }

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class ProviderCredentialSet(BaseModel):
    api_keys: dict[str, str] = Field(default_factory=dict)
    base_urls: dict[str, str] = Field(default_factory=dict)


class ProviderPlatformConfig(BaseModel):
    provider_id: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)
    protocol_adapter: str = Field(min_length=1, max_length=100)
    api_style: str = Field(min_length=1, max_length=100)
    api_key_env_var: str | None = None
    api_key_label: str | None = None
    base_url: str | None = None
    base_url_env_var: str | None = None
    allow_base_url_override: bool = False
    endpoint_path: str | None = None
    enabled: bool = True
    enabled_by_default: bool = True
    request_headers: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("provider_id", "display_name", "protocol_adapter", "api_style")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

    @field_validator("endpoint_path")
    @classmethod
    def normalize_endpoint_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return value if value.startswith("/") else f"/{value}"


class ProviderRegistryFile(BaseModel):
    providers: list[ProviderPlatformConfig] = Field(default_factory=list)


class ModelOptionConfig(BaseModel):
    model: str = Field(min_length=3, max_length=300)
    label: str = Field(min_length=1, max_length=200)
    provider_id: str = Field(min_length=1, max_length=100)
    enabled: bool = True

    @field_validator("model", "label", "provider_id")
    @classmethod
    def strip_required_model_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value


class ModelRegistryFile(BaseModel):
    models: list[ModelOptionConfig] = Field(default_factory=list)


class ConfiguredParticipant(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=3, max_length=300)
    prompt: str = Field(min_length=1)
    role_label: str | None = None
    enabled: bool = True

    @field_validator("name", "model", "prompt")
    @classmethod
    def strip_required_participant_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

    @field_validator("role_label")
    @classmethod
    def strip_optional_role_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class DiscussionTemplateConfig(BaseModel):
    discussion_id: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    max_turn_cycles: int = Field(default=2, ge=1, le=8)
    participants: list[ConfiguredParticipant] = Field(default_factory=list)

    @field_validator("discussion_id", "display_name")
    @classmethod
    def strip_required_discussion_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value


class DiscussionConfigFile(BaseModel):
    default_discussion_id: str
    discussions: list[DiscussionTemplateConfig] = Field(default_factory=list)


class DiscussionParticipant(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=3, max_length=300)
    prompt: str = Field(min_length=1)
    prompt_source: str = Field(min_length=1)
    role_label: str | None = None
    enabled: bool = True
    sort_order: int = 0

    @field_validator("participant_id", "name", "model", "prompt", "prompt_source")
    @classmethod
    def strip_required_runtime_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

    @field_validator("role_label")
    @classmethod
    def strip_optional_runtime_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def provider_id(self) -> str:
        return self.model.partition("/")[0]

    @property
    def model_name(self) -> str:
        return self.model.partition("/")[2]

    @property
    def speaker_label(self) -> str:
        return self.name


class DiscussionSettings(BaseModel):
    max_turn_cycles: int = Field(default=2, ge=1, le=8)
    max_output_tokens: int = Field(default=700, ge=128, le=12000)
    temperature: float | None = Field(default=0.3, ge=0.0, le=2.0)


class DiscussionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    credentials: ProviderCredentialSet
    participants: list[DiscussionParticipant] = Field(default_factory=list)
    settings: DiscussionSettings
    discussion_id: str | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value

    @model_validator(mode="after")
    def validate_participants(self) -> "DiscussionRequest":
        if not self.participants:
            return self
        if not any(participant.enabled for participant in self.participants):
            raise ValueError("At least one participant must be enabled.")
        participant_ids = [participant.participant_id for participant in self.participants]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Participant IDs must be unique.")
        return self


class ProviderError(BaseModel):
    provider_id: str
    participant_id: str | None = None
    speaker: str | None = None
    model_name: str | None = None
    code: str = "provider_error"
    message: str
    details: str | None = None


class ProviderUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class SessionTurn(BaseModel):
    turn_id: str
    cycle_number: int
    turn_number: int
    participant_id: str
    speaker: str
    role_label: str | None = None
    model: str
    provider_id: str
    provider_display_name: str
    model_name: str
    prompt_source: str
    prompt_used: str
    status: TurnStatus = TurnStatus.PENDING
    content: str | None = None
    error: ProviderError | None = None
    request_id: str | None = None
    usage: ProviderUsage | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()


class SynthesisData(BaseModel):
    speaker: str
    final_answer: str
    unresolved_disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    continue_discussion: bool = False
    raw_output: str


class SessionRecord(BaseModel):
    session_id: str
    status: SessionStatus = SessionStatus.QUEUED
    question: str
    discussion_id: str | None = None
    discussion_display_name: str | None = None
    participants: list[DiscussionParticipant] = Field(default_factory=list)
    settings: DiscussionSettings
    turns: list[SessionTurn] = Field(default_factory=list)
    final_answer: str | None = None
    unresolved_disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    errors: list[ProviderError] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()


class SessionEnvelope(BaseModel):
    sessions: list[SessionRecord] = Field(default_factory=list)
    saved_at: datetime = Field(default_factory=utc_now)


class ProviderRequest(BaseModel):
    provider_id: str
    provider_display_name: str
    model_name: str
    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    temperature: float | None = None
    base_url: str | None = None
    endpoint_path: str | None = None
    request_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    provider_id: str
    provider_display_name: str
    model_name: str
    text: str
    request_id: str | None = None
    usage: ProviderUsage | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None

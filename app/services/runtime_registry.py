from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from app.schemas import (
    DiscussionConfigFile,
    DiscussionParticipant,
    DiscussionRequest,
    DiscussionTemplateConfig,
    ModelOptionConfig,
    ModelRegistryFile,
    ProviderCredentialSet,
    ProviderPlatformConfig,
    ProviderRegistryFile,
)


class RuntimeRegistry:
    def __init__(
        self,
        *,
        providers: dict[str, ProviderPlatformConfig],
        models: dict[str, ModelOptionConfig],
        discussions: dict[str, DiscussionTemplateConfig],
        default_discussion_id: str,
    ):
        self.providers = providers
        self.models = models
        self.discussions = discussions
        self.default_discussion_id = default_discussion_id

    @classmethod
    def load(cls, config_dir: Path) -> "RuntimeRegistry":
        provider_file = ProviderRegistryFile.model_validate(
            json.loads((config_dir / "providers.json").read_text(encoding="utf-8"))
        )
        model_file = ModelRegistryFile.model_validate(
            json.loads((config_dir / "models.json").read_text(encoding="utf-8"))
        )
        discussion_file = DiscussionConfigFile.model_validate(
            json.loads((config_dir / "discussions.json").read_text(encoding="utf-8"))
        )

        provider_map = {
            provider.provider_id: provider
            for provider in provider_file.providers
            if provider.enabled
        }
        if len(provider_map) != len([provider for provider in provider_file.providers if provider.enabled]):
            raise ValueError("Provider IDs must be unique in providers.json.")

        model_map = {
            model.model: model
            for model in model_file.models
            if model.enabled
        }
        if len(model_map) != len([model for model in model_file.models if model.enabled]):
            raise ValueError("Model values must be unique in models.json.")

        discussion_map = {
            discussion.discussion_id: discussion
            for discussion in discussion_file.discussions
        }
        if discussion_file.default_discussion_id not in discussion_map:
            raise ValueError("Default discussion ID does not exist in discussions.json.")
        if not discussion_map:
            raise ValueError("At least one discussion template must exist in discussions.json.")

        registry = cls(
            providers=provider_map,
            models=model_map,
            discussions=discussion_map,
            default_discussion_id=discussion_file.default_discussion_id,
        )
        registry._validate_models()
        registry._validate_discussions()
        return registry

    def _validate_models(self) -> None:
        for model in self.models.values():
            provider_id, model_name = self.split_model_ref(model.model)
            if provider_id != model.provider_id:
                raise ValueError(f"Model '{model.model}' must use the same provider_id in both fields.")
            if provider_id not in self.providers:
                raise ValueError(f"Model '{model.model}' references unknown provider '{provider_id}'.")
            if not model_name:
                raise ValueError(f"Model '{model.model}' must be formatted as 'provider/model'.")

    def _validate_discussions(self) -> None:
        for discussion in self.discussions.values():
            participants = self.materialize_participants(discussion.discussion_id)
            enabled_participants = [participant for participant in participants if participant.enabled]
            if not enabled_participants:
                raise ValueError(f"Discussion '{discussion.discussion_id}' must include at least one enabled participant.")
            if not any(self.is_synthesizer(participant) for participant in enabled_participants):
                raise ValueError(
                    f"Discussion '{discussion.discussion_id}' must include at least one participant with role_label 'Synthesizer'."
                )
            for participant in participants:
                if participant.model not in self.models:
                    raise ValueError(
                        f"Discussion '{discussion.discussion_id}' references model '{participant.model}' missing from models.json."
                    )

    def get_provider(self, provider_id: str) -> ProviderPlatformConfig | None:
        return self.providers.get(provider_id)

    def model_options(self, provider_id: str | None = None) -> list[ModelOptionConfig]:
        models = list(self.models.values())
        if provider_id:
            models = [model for model in models if model.provider_id == provider_id]
        models.sort(key=lambda item: (self.providers[item.provider_id].display_name.lower(), item.label.lower(), item.model.lower()))
        return models

    def get_discussion(self, discussion_id: str | None = None) -> DiscussionTemplateConfig:
        resolved_id = discussion_id or self.default_discussion_id
        discussion = self.discussions.get(resolved_id)
        if discussion is None:
            raise ValueError(f"Unknown discussion template '{resolved_id}'.")
        return discussion

    def materialize_participants(self, discussion_id: str | None = None) -> list[DiscussionParticipant]:
        discussion = self.get_discussion(discussion_id)
        participants: list[DiscussionParticipant] = []
        for index, configured in enumerate(discussion.participants):
            provider_id, model_name = self.split_model_ref(configured.model)
            if provider_id not in self.providers:
                raise ValueError(
                    f"Discussion '{discussion.discussion_id}' references unknown provider '{provider_id}'."
                )
            participants.append(
                DiscussionParticipant(
                    participant_id=self._participant_id(configured.name, index),
                    name=configured.name,
                    model=f"{provider_id}/{model_name}",
                    prompt=configured.prompt,
                    prompt_source=configured.prompt,
                    role_label=configured.role_label,
                    enabled=configured.enabled,
                    sort_order=index,
                )
            )
        return participants

    def hydrate_request(self, request: DiscussionRequest) -> DiscussionRequest:
        discussion_id = request.discussion_id or self.default_discussion_id
        participants = request.participants or self.materialize_participants(discussion_id)
        return request.model_copy(
            update={
                "discussion_id": discussion_id,
                "participants": copy.deepcopy(participants),
            }
        )

    def validate_participant(self, participant: DiscussionParticipant) -> None:
        provider_id, model_name = self.split_model_ref(participant.model)
        if not model_name:
            raise ValueError(f"Participant '{participant.name}' must define model as 'provider/model'.")
        provider = self.get_provider(provider_id)
        if provider is None:
            raise ValueError(f"Unknown or disabled provider '{provider_id}'.")

    def validate_request(self, request: DiscussionRequest) -> None:
        hydrated = self.hydrate_request(request)
        enabled_participants = [participant for participant in hydrated.participants if participant.enabled]
        if not enabled_participants:
            raise ValueError("At least one participant must be enabled.")
        for participant in enabled_participants:
            self.validate_participant(participant)
        if not any(self.is_synthesizer(participant) for participant in enabled_participants):
            raise ValueError("At least one enabled participant with role_label 'Synthesizer' is required.")

    @staticmethod
    def split_model_ref(model_ref: str) -> tuple[str, str]:
        provider_id, separator, model_name = model_ref.partition("/")
        provider_id = provider_id.strip()
        model_name = model_name.strip()
        if not separator or not provider_id or not model_name:
            return "", ""
        return provider_id, model_name

    @staticmethod
    def _participant_id(name: str, index: int) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "participant"
        return f"{base}_{index + 1}"

    @staticmethod
    def is_synthesizer(participant: DiscussionParticipant) -> bool:
        label = (participant.role_label or "").strip().lower()
        return label in {"synthesizer", "final", "summarizer", "moderator"}

    @staticmethod
    def provider_protocol_label(provider: ProviderPlatformConfig) -> str:
        if provider.protocol_adapter == "anthropic_messages":
            return "Anthropic-Compatible"
        if provider.protocol_adapter in {"openai_compatible_chat", "openai_responses"}:
            return "OpenAI-Compatible"
        if provider.protocol_adapter == "gemini_generate_content":
            return "Gemini Native"
        return provider.protocol_adapter

    def default_credentials_from_env(self) -> ProviderCredentialSet:
        api_keys: dict[str, str] = {}
        base_urls: dict[str, str] = {}
        for provider in self.providers.values():
            if provider.api_key_env_var:
                value = os.getenv(provider.api_key_env_var)
                if value:
                    api_keys[provider.provider_id] = value
            base_url = provider.base_url
            if provider.base_url_env_var and os.getenv(provider.base_url_env_var):
                base_url = os.getenv(provider.base_url_env_var)
            if base_url:
                base_urls[provider.provider_id] = base_url
        return ProviderCredentialSet(api_keys=api_keys, base_urls=base_urls)

    def client_payload(self) -> dict[str, object]:
        return {
            "default_discussion_id": self.default_discussion_id,
            "providers": [provider.model_dump(mode="json") for provider in self.providers.values()],
            "models": [
                {
                    **model.model_dump(mode="json"),
                    "model_name": self.split_model_ref(model.model)[1],
                    "provider_display_name": self.providers[model.provider_id].display_name,
                }
                for model in self.model_options()
            ],
            "discussions": [
                {
                    **discussion.model_dump(mode="json"),
                    "participants": [
                        {
                            **participant.model_dump(mode="json"),
                            "participant_id": self._participant_id(participant.name, index),
                            "sort_order": index,
                            "prompt_source": participant.prompt,
                            "provider_id": self.split_model_ref(participant.model)[0],
                            "model_name": self.split_model_ref(participant.model)[1],
                        }
                        for index, participant in enumerate(discussion.participants)
                    ],
                }
                for discussion in self.discussions.values()
            ],
        }

    def provider_credentials_for_form(self, credentials: ProviderCredentialSet | None = None) -> list[dict[str, object]]:
        merged = credentials or self.default_credentials_from_env()
        rows: list[dict[str, object]] = []
        for provider in self.providers.values():
            base_url = merged.base_urls.get(provider.provider_id, provider.base_url)
            rows.append(
                {
                    "provider_id": provider.provider_id,
                    "display_name": provider.display_name,
                    "api_key_label": provider.api_key_label or f"{provider.display_name} API Key",
                    "api_key_value": merged.api_keys.get(provider.provider_id, ""),
                    "base_url_value": base_url or "",
                    "allow_base_url_override": provider.allow_base_url_override,
                    "notes": provider.notes,
                    "enabled_by_default": provider.enabled_by_default,
                    "protocol_label": self.provider_protocol_label(provider),
                }
            )
        rows.sort(key=lambda row: (not row["enabled_by_default"], str(row["display_name"]).lower()))
        return rows

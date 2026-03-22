from __future__ import annotations

import httpx

from app.config import Settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import BaseLLMProvider
from app.providers.compatible_provider import OpenAICompatibleProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.schemas import ProviderCredentialSet, ProviderPlatformConfig


def build_provider(
    provider: ProviderPlatformConfig,
    credentials: ProviderCredentialSet,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BaseLLMProvider:
    api_key = credentials.api_keys.get(provider.provider_id)
    adapter_id = provider.protocol_adapter

    if adapter_id == "openai_responses":
        return OpenAIProvider(api_key=api_key, timeout=settings.request_timeout_seconds, transport=transport)
    if adapter_id == "openai_compatible_chat":
        return OpenAICompatibleProvider(api_key=api_key, timeout=settings.request_timeout_seconds, transport=transport)
    if adapter_id == "anthropic_messages":
        return AnthropicProvider(api_key=api_key, timeout=settings.request_timeout_seconds, transport=transport)
    if adapter_id == "gemini_generate_content":
        return GeminiProvider(api_key=api_key, timeout=settings.request_timeout_seconds, transport=transport)
    raise ValueError(f"Unsupported protocol adapter: {adapter_id}")

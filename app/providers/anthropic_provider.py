from __future__ import annotations

from typing import Any

from app.providers.base import BaseLLMProvider
from app.schemas import ProviderRequest, ProviderResponse, ProviderUsage


class AnthropicProvider(BaseLLMProvider):
    adapter_name = "anthropic_messages"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._require_api_key(request.provider_id, request.model_name)
        base_url = (request.base_url or "https://api.anthropic.com/v1").rstrip("/")
        payload: dict[str, Any] = {
            "model": request.model_name,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        response_json = await self._post_json(
            f"{base_url}/messages",
            provider_id=request.provider_id,
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                **request.request_headers,
            },
            payload=payload,
            model_name=request.model_name,
        )

        parts = response_json.get("content", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        text = "\n".join(part.strip() for part in texts if part and part.strip()).strip()
        if not text:
            raise self._build_error(
                provider_id=request.provider_id,
                message=f"{request.provider_id} response did not contain any text content.",
                model_name=request.model_name,
                code="malformed_response",
                details=str(response_json)[:2000],
            )

        usage_block = response_json.get("usage", {})
        usage = ProviderUsage(
            input_tokens=usage_block.get("input_tokens"),
            output_tokens=usage_block.get("output_tokens"),
            total_tokens=None,
        )
        return ProviderResponse(
            provider_id=request.provider_id,
            provider_display_name=request.provider_display_name,
            model_name=request.model_name,
            text=text,
            request_id=response_json.get("id"),
            usage=usage,
            finish_reason=response_json.get("stop_reason"),
            raw_response=response_json,
        )

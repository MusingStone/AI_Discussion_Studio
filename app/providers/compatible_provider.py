from __future__ import annotations

from typing import Any

from app.providers.base import BaseLLMProvider
from app.schemas import ProviderRequest, ProviderResponse, ProviderUsage


class OpenAICompatibleProvider(BaseLLMProvider):
    adapter_name = "openai_compatible_chat"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._require_api_key(request.provider_id, request.model_name)
        base_url = (request.base_url or "").rstrip("/")
        endpoint_path = request.endpoint_path or "/chat/completions"
        if not base_url:
            raise self._build_error(
                provider_id=request.provider_id,
                message=f"Missing base URL for provider '{request.provider_id}'.",
                model_name=request.model_name,
                code="missing_base_url",
            )

        payload: dict[str, Any] = {
            "model": request.model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "max_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        response_json = await self._post_json(
            f"{base_url}{endpoint_path}",
            provider_id=request.provider_id,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **request.request_headers,
            },
            payload=payload,
            model_name=request.model_name,
        )
        text = self._extract_text(response_json)
        if not text:
            raise self._build_error(
                provider_id=request.provider_id,
                message=f"{request.provider_id} response did not contain any message text.",
                model_name=request.model_name,
                code="malformed_response",
                details=str(response_json)[:2000],
            )

        usage_block = response_json.get("usage", {})
        usage = ProviderUsage(
            input_tokens=usage_block.get("prompt_tokens"),
            output_tokens=usage_block.get("completion_tokens"),
            total_tokens=usage_block.get("total_tokens"),
        )
        finish_reason = None
        choices = response_json.get("choices", [])
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")

        return ProviderResponse(
            provider_id=request.provider_id,
            provider_display_name=request.provider_display_name,
            model_name=request.model_name,
            text=text,
            request_id=response_json.get("id"),
            usage=usage,
            finish_reason=finish_reason,
            raw_response=response_json,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""

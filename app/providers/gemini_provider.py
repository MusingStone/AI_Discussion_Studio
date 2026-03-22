from __future__ import annotations

from typing import Any

from app.providers.base import BaseLLMProvider
from app.schemas import ProviderRequest, ProviderResponse, ProviderUsage


class GeminiProvider(BaseLLMProvider):
    adapter_name = "gemini_generate_content"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._require_api_key(request.provider_id, request.model_name)
        base_url = (request.base_url or "https://generativelanguage.googleapis.com/v1beta/models").rstrip("/")
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{request.system_prompt}\n\n{request.user_prompt}"}],
                }
            ],
            "generationConfig": {"maxOutputTokens": request.max_output_tokens},
        }
        if request.temperature is not None:
            payload["generationConfig"]["temperature"] = request.temperature

        response_json = await self._post_json(
            f"{base_url}/{request.model_name}:generateContent",
            provider_id=request.provider_id,
            headers={
                "x-goog-api-key": self.api_key or "",
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
                message=f"{request.provider_id} response did not contain any text content.",
                model_name=request.model_name,
                code="malformed_response",
                details=str(response_json)[:2000],
            )

        usage_block = response_json.get("usageMetadata", {})
        usage = ProviderUsage(
            input_tokens=usage_block.get("promptTokenCount"),
            output_tokens=usage_block.get("candidatesTokenCount"),
            total_tokens=usage_block.get("totalTokenCount"),
        )
        return ProviderResponse(
            provider_id=request.provider_id,
            provider_display_name=request.provider_display_name,
            model_name=request.model_name,
            text=text,
            usage=usage,
            raw_response=response_json,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "\n".join(part.strip() for part in texts if part and part.strip()).strip()

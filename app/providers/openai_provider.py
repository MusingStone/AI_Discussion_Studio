from __future__ import annotations

from typing import Any

from app.providers.base import BaseLLMProvider
from app.schemas import ProviderRequest, ProviderResponse, ProviderUsage


class OpenAIProvider(BaseLLMProvider):
    adapter_name = "openai_responses"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._require_api_key(request.provider_id, request.model_name)
        base_url = (request.base_url or "https://api.openai.com/v1").rstrip("/")
        payload: dict[str, Any] = {
            "model": request.model_name,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.user_prompt}],
                },
            ],
            "max_output_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        response_json = await self._post_json(
            f"{base_url}/responses",
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
                message=f"{request.provider_id} response did not contain any output text.",
                model_name=request.model_name,
                code="malformed_response",
                details=str(response_json)[:2000],
            )

        usage_block = response_json.get("usage", {})
        usage = ProviderUsage(
            input_tokens=usage_block.get("input_tokens"),
            output_tokens=usage_block.get("output_tokens"),
            total_tokens=usage_block.get("total_tokens"),
        )
        return ProviderResponse(
            provider_id=request.provider_id,
            provider_display_name=request.provider_display_name,
            model_name=request.model_name,
            text=text,
            request_id=response_json.get("id"),
            usage=usage,
            finish_reason=response_json.get("status"),
            raw_response=response_json,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        texts: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        return "\n".join(part.strip() for part in texts if part and part.strip()).strip()

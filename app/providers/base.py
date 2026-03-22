from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.schemas import ProviderError, ProviderRequest, ProviderResponse


class ProviderClientException(Exception):
    def __init__(self, error: ProviderError):
        super().__init__(error.message)
        self.error = error


class BaseLLMProvider(ABC):
    adapter_name: str

    def __init__(self, api_key: str | None, timeout: float, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    def _build_error(
        self,
        *,
        provider_id: str,
        message: str,
        model_name: str | None = None,
        code: str = "provider_error",
        details: str | None = None,
    ) -> ProviderClientException:
        return ProviderClientException(
            ProviderError(
                provider_id=provider_id,
                model_name=model_name,
                code=code,
                message=message,
                details=details,
            )
        )

    def _require_api_key(self, provider_id: str, model_name: str) -> None:
        if not self.api_key:
            raise self._build_error(
                provider_id=provider_id,
                message=f"Missing API key for provider '{provider_id}'.",
                model_name=model_name,
                code="missing_api_key",
            )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self.transport)

    async def _post_json(
        self,
        url: str,
        *,
        provider_id: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise self._build_error(
                provider_id=provider_id,
                message=f"{provider_id} request timed out.",
                model_name=model_name,
                code="timeout",
                details=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise self._build_error(
                provider_id=provider_id,
                message=f"{provider_id} request failed before receiving a response.",
                model_name=model_name,
                code="http_error",
                details=str(exc),
            ) from exc

        if response.status_code >= 400:
            raise self._build_error(
                provider_id=provider_id,
                message=f"{provider_id} returned HTTP {response.status_code}.",
                model_name=model_name,
                code="http_status_error",
                details=response.text[:2000],
            )

        try:
            return response.json()
        except ValueError as exc:
            raise self._build_error(
                provider_id=provider_id,
                message=f"{provider_id} returned malformed JSON.",
                model_name=model_name,
                code="malformed_response",
                details=response.text[:2000],
            ) from exc

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app.schemas import DiscussionParticipant, DiscussionRequest, DiscussionSettings, ProviderCredentialSet
from app.services.runtime_registry import RuntimeRegistry

router = APIRouter(tags=["web"])


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _registry(request: Request) -> RuntimeRegistry:
    registry = RuntimeRegistry.load(request.app.state.settings.config_dir)
    request.app.state.registry = registry
    request.app.state.orchestrator.registry = registry
    return registry


def _participant_id(name: str, index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "participant"
    return f"{base}_{index + 1}"


def default_form_state(request: Request, discussion_id: str | None = None) -> dict[str, Any]:
    settings = request.app.state.settings
    registry = _registry(request)
    discussion = registry.get_discussion(discussion_id)
    return {
        "question": "",
        "discussion_id": discussion.discussion_id,
        "credentials": registry.default_credentials_from_env(),
        "participants": registry.materialize_participants(discussion.discussion_id),
        "settings": {
            "max_turn_cycles": discussion.max_turn_cycles or settings.default_max_turn_cycles,
            "max_output_tokens": settings.default_max_output_tokens,
            "temperature": settings.default_temperature,
        },
    }


def _parse_form_payload(form: Any, request: Request, *, fallback_to_template: bool) -> dict[str, Any]:
    settings = request.app.state.settings
    registry = _registry(request)
    env_credentials = registry.default_credentials_from_env()
    api_keys = dict(env_credentials.api_keys)
    base_urls = dict(env_credentials.base_urls)

    for provider in registry.providers.values():
        api_key_value = (form.get(f"api_key__{provider.provider_id}") or "").strip()
        if api_key_value:
            api_keys[provider.provider_id] = api_key_value

        base_url_value = (form.get(f"base_url__{provider.provider_id}") or "").strip()
        if base_url_value:
            base_urls[provider.provider_id] = base_url_value

    participants: list[DiscussionParticipant] = []
    participant_count = int(form.get("participant_count") or 0)
    for index in range(participant_count):
        name = (form.get(f"participants-{index}-name") or "").strip()
        provider_id = (form.get(f"participants-{index}-provider_id") or "").strip()
        model_name = (form.get(f"participants-{index}-model_name") or "").strip()
        model = (form.get(f"participants-{index}-model") or "").strip()
        prompt = (form.get(f"participants-{index}-prompt") or "").strip()
        role_label = (form.get(f"participants-{index}-role_label") or "").strip() or None
        participant_id = (form.get(f"participants-{index}-participant_id") or "").strip() or _participant_id(name or "participant", index)
        if provider_id and model_name:
            model = f"{provider_id}/{model_name}"
        if not any([name, model, prompt, role_label]):
            continue

        participants.append(
            DiscussionParticipant(
                participant_id=participant_id,
                name=name or f"Participant {index + 1}",
                model=model,
                prompt=prompt,
                prompt_source=prompt,
                role_label=role_label,
                enabled=_truthy(form.get(f"participants-{index}-enabled")),
                sort_order=int(form.get(f"participants-{index}-sort_order") or index),
            )
        )

    discussion_id = (form.get("discussion_id") or "").strip() or registry.default_discussion_id
    if fallback_to_template and not participants:
        participants = registry.materialize_participants(discussion_id)

    discussion = registry.get_discussion(discussion_id)
    return {
        "question": form.get("question", ""),
        "discussion_id": discussion_id,
        "credentials": ProviderCredentialSet(api_keys=api_keys, base_urls=base_urls),
        "participants": participants,
        "settings": {
            "max_turn_cycles": int(form.get("max_turn_cycles") or discussion.max_turn_cycles or settings.default_max_turn_cycles),
            "max_output_tokens": int(form.get("max_output_tokens") or settings.default_max_output_tokens),
            "temperature": float(form.get("temperature") or settings.default_temperature),
        },
    }


def form_state_from_form_input(form: Any, request: Request) -> dict[str, Any]:
    return _parse_form_payload(form, request, fallback_to_template=True)


def build_request_from_form(form: Any, request: Request) -> DiscussionRequest:
    registry = _registry(request)
    payload = _parse_form_payload(form, request, fallback_to_template=False)
    discussion_request = registry.hydrate_request(
        DiscussionRequest(
            question=payload["question"],
            discussion_id=payload["discussion_id"],
            credentials=payload["credentials"],
            participants=payload["participants"],
            settings=DiscussionSettings.model_validate(payload["settings"]),
        )
    )
    registry.validate_request(discussion_request)
    return discussion_request


def build_page_context(
    request: Request,
    *,
    session_id: str | None = None,
    form_state: dict[str, Any] | None = None,
    form_errors: list[str] | None = None,
) -> dict[str, Any]:
    settings = request.app.state.settings
    registry = _registry(request)
    client_payload = registry.client_payload()
    session = request.app.state.session_store.get_session(session_id) if session_id else None
    context = settings.base_template_context()
    context.update(
        {
            "request": request,
            "recent_sessions": request.app.state.session_store.recent_sessions(),
            "session": session,
            "form_state": form_state or default_form_state(request),
            "form_errors": form_errors or [],
            "provider_credentials": registry.provider_credentials_for_form(
                form_state["credentials"] if form_state else registry.default_credentials_from_env()
            ),
            "discussion_options": client_payload["discussions"],
            "provider_options": client_payload["providers"],
            "model_options": client_payload["models"],
            "registry_json": json.dumps(client_payload, ensure_ascii=True),
        }
    )
    return context


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, session_id: str | None = None) -> HTMLResponse:
    context = build_page_context(request, session_id=session_id)
    return request.app.state.templates.TemplateResponse(request, "index.html", context)


@router.post("/discuss", response_class=HTMLResponse)
async def submit_discussion(request: Request) -> HTMLResponse:
    form = await request.form()
    try:
        discussion_request = build_request_from_form(form, request)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            errors = [error["msg"] for error in exc.errors()]
        else:
            errors = [str(exc)]
        form_state = form_state_from_form_input(form, request)
        context = build_page_context(request, form_state=form_state, form_errors=errors)
        return request.app.state.templates.TemplateResponse(request, "index.html", context, status_code=400)

    orchestrator = request.app.state.orchestrator
    session = orchestrator.create_session(discussion_request)
    asyncio.create_task(orchestrator.run_session(session.session_id, discussion_request))
    return RedirectResponse(url=f"/?session_id={session.session_id}#results-shell", status_code=303)


@router.post("/sessions/{session_id}/delete", response_class=HTMLResponse)
async def delete_session(request: Request, session_id: str) -> RedirectResponse:
    request.app.state.session_store.delete_session(session_id)
    referer = request.headers.get("referer") or "/"
    if "#" not in referer:
        referer = f"{referer}#recent-sessions"
    if f"session_id={session_id}" in referer:
        referer = "/#recent-sessions"
    return RedirectResponse(url=referer, status_code=303)

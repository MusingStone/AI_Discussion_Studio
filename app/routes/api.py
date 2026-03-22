from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.schemas import DiscussionRequest
from app.services.runtime_registry import RuntimeRegistry

router = APIRouter(prefix="/api", tags=["api"])


def _registry(request: Request) -> RuntimeRegistry:
    registry = RuntimeRegistry.load(request.app.state.settings.config_dir)
    request.app.state.registry = registry
    request.app.state.orchestrator.registry = registry
    return registry


@router.get("/runtime-config")
async def runtime_config(request: Request) -> dict:
    return _registry(request).client_payload()


@router.post("/discuss")
async def start_discussion(request: Request, payload: DiscussionRequest) -> dict[str, str]:
    registry = _registry(request)
    payload = registry.hydrate_request(payload)
    registry.validate_request(payload)
    orchestrator = request.app.state.orchestrator
    session = orchestrator.create_session(payload)
    asyncio.create_task(orchestrator.run_session(session.session_id, payload))
    return {"session_id": session.session_id}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> dict:
    session = request.app.state.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session.model_dump(mode="json")


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict[str, bool]:
    deleted = request.app.state.session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True}

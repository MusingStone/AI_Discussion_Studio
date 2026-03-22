from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.logging_config import configure_logging
from app.routes.api import router as api_router
from app.routes.web import router as web_router
from app.services.orchestrator import DiscussionOrchestrator
from app.services.prompt_loader import PromptLoader
from app.services.runtime_registry import RuntimeRegistry
from app.services.session_store import SessionStore


def create_app(settings_override: Optional[Settings] = None) -> FastAPI:
    configure_logging()
    settings = settings_override or Settings.load()
    settings.ensure_directories()

    app = FastAPI(title=settings.project_name)
    templates = Jinja2Templates(directory=str(settings.template_dir))
    session_store = SessionStore(settings.session_file, settings.recent_session_limit)
    prompt_loader = PromptLoader(settings.prompt_dir)
    registry = RuntimeRegistry.load(settings.config_dir)
    orchestrator = DiscussionOrchestrator(settings, prompt_loader, session_store, registry)

    app.state.settings = settings
    app.state.templates = templates
    app.state.session_store = session_store
    app.state.prompt_loader = prompt_loader
    app.state.registry = registry
    app.state.orchestrator = orchestrator

    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.include_router(web_router)
    app.include_router(api_router)
    return app


app = create_app()

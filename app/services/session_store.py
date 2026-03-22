from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.logging_config import get_logger
from app.schemas import SessionEnvelope, SessionRecord

logger = get_logger(__name__)


class SessionStore:
    def __init__(self, session_file: Path, recent_limit: int):
        self.session_file = session_file
        self.recent_limit = recent_limit
        self._active: Dict[str, SessionRecord] = {}
        self._persisted: Dict[str, SessionRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.session_file.exists():
            return
        try:
            payload = json.loads(self.session_file.read_text(encoding="utf-8"))
            envelope = SessionEnvelope.model_validate(payload)
        except Exception as exc:
            logger.warning("Failed to load session history from %s: %s", self.session_file, exc)
            return
        self._persisted = {session.session_id: session for session in envelope.sessions}

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        session = self._active.get(session_id) or self._persisted.get(session_id)
        if session is None:
            return None
        return self._copy_session(session)

    def upsert_session(self, session: SessionRecord) -> None:
        session.touch()
        self._active[session.session_id] = self._copy_session(session)
        self._persist()

    def recent_sessions(self) -> List[SessionRecord]:
        sessions = list(self._merged_sessions().values())
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return [self._copy_session(session) for session in sessions[: self.recent_limit]]

    def delete_session(self, session_id: str) -> bool:
        existed = session_id in self._active or session_id in self._persisted
        self._active.pop(session_id, None)
        self._persisted.pop(session_id, None)
        if existed:
            self._persist()
        return existed

    @staticmethod
    def _copy_session(session: SessionRecord) -> SessionRecord:
        return SessionRecord.model_validate(session.model_dump(mode="json"))

    def _merged_sessions(self) -> Dict[str, SessionRecord]:
        merged = dict(self._persisted)
        merged.update(self._active)
        return merged

    def _persist(self) -> None:
        sessions = list(self._merged_sessions().values())
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        trimmed = sessions[: self.recent_limit]
        self._persisted = {session.session_id: session for session in trimmed}

        envelope = SessionEnvelope(sessions=trimmed)
        serialized = json.dumps(envelope.model_dump(mode="json"), indent=2)
        temp_path = self.session_file.with_suffix(f"{self.session_file.suffix}.tmp")
        temp_path.write_text(serialized, encoding="utf-8")
        os.replace(temp_path, self.session_file)

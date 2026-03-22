from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.logging_config import get_logger
from app.providers.base import ProviderClientException
from app.providers.factory import build_provider
from app.schemas import (
    DiscussionParticipant,
    DiscussionRequest,
    ProviderError,
    ProviderRequest,
    SessionRecord,
    SessionStatus,
    SessionTurn,
    SynthesisData,
    TurnStatus,
)
from app.services.prompt_loader import PromptLoader
from app.services.runtime_registry import RuntimeRegistry
from app.services.session_store import SessionStore

logger = get_logger(__name__)


@dataclass
class ResolvedParticipant:
    participant: DiscussionParticipant
    provider_config: Any


class DiscussionOrchestrator:
    def __init__(
        self,
        settings: Settings,
        prompt_loader: PromptLoader,
        session_store: SessionStore,
        registry: RuntimeRegistry,
    ):
        self.settings = settings
        self.prompt_loader = prompt_loader
        self.session_store = session_store
        self.registry = registry

    def create_session(self, request: DiscussionRequest) -> SessionRecord:
        self.registry = RuntimeRegistry.load(self.settings.config_dir)
        request = self.registry.hydrate_request(request)
        self.registry.validate_request(request)
        discussion = self.registry.get_discussion(request.discussion_id)
        session = SessionRecord(
            session_id=str(uuid.uuid4()),
            question=request.question,
            discussion_id=request.discussion_id,
            discussion_display_name=discussion.display_name,
            participants=request.participants,
            settings=request.settings,
        )
        self.session_store.upsert_session(session)
        return session

    async def run_session(self, session_id: str, request: DiscussionRequest) -> None:
        self.registry = RuntimeRegistry.load(self.settings.config_dir)
        request = self.registry.hydrate_request(request)
        session = self.session_store.get_session(session_id)
        if session is None:
            return

        resolved_participants = self._resolve_participants(request.participants)
        session.status = SessionStatus.RUNNING
        self.session_store.upsert_session(session)
        latest_synthesis: SynthesisData | None = None

        try:
            turn_number = len(session.turns)
            for cycle_number in range(1, request.settings.max_turn_cycles + 1):
                for resolved in resolved_participants:
                    turn_number += 1
                    turn = await self._run_turn(
                        session=session,
                        request=request,
                        resolved=resolved,
                        cycle_number=cycle_number,
                        turn_number=turn_number,
                        latest_synthesis=latest_synthesis,
                    )
                    if turn.status == TurnStatus.COMPLETED and self.registry.is_synthesizer(resolved.participant):
                        latest_synthesis = self._parse_synthesizer_output(turn)
                        session.final_answer = latest_synthesis.final_answer
                        session.unresolved_disagreements = latest_synthesis.unresolved_disagreements
                        session.open_questions = latest_synthesis.open_questions
                        self.session_store.upsert_session(session)

            if latest_synthesis is None:
                session.status = SessionStatus.FAILED
                session.errors.append(
                    ProviderError(
                        provider_id="system",
                        code="missing_synthesis",
                        message="No synthesizer turn produced a final synthesis.",
                    )
                )
            else:
                session.status = SessionStatus.COMPLETED
            self.session_store.upsert_session(session)
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Session %s failed: %s", session_id, exc)
            session.status = SessionStatus.FAILED
            session.errors.append(
                ProviderError(
                    provider_id="system",
                    code="orchestrator_error",
                    message="The discussion failed before completion.",
                    details=str(exc),
                )
            )
            self.session_store.upsert_session(session)

    def _resolve_participants(self, participants: list[DiscussionParticipant]) -> list[ResolvedParticipant]:
        enabled = [
            participant
            for participant in sorted(participants, key=lambda item: (item.sort_order, item.participant_id))
            if participant.enabled
        ]
        return [
            ResolvedParticipant(
                participant=participant,
                provider_config=self.registry.providers[participant.provider_id],
            )
            for participant in enabled
        ]

    async def _run_turn(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        turn_number: int,
        latest_synthesis: SynthesisData | None,
    ) -> SessionTurn:
        provider = build_provider(resolved.provider_config, request.credentials, self.settings)
        participant = resolved.participant
        provider_config = resolved.provider_config

        prompt_source, system_prompt = self.prompt_loader.resolve_prompt(
            participant.prompt,
            self._build_prompt_context(session, participant, cycle_number, latest_synthesis),
        )
        user_prompt = self._build_user_prompt(session, participant, cycle_number, latest_synthesis)

        turn = SessionTurn(
            turn_id=f"turn_{turn_number}",
            cycle_number=cycle_number,
            turn_number=turn_number,
            participant_id=participant.participant_id,
            speaker=participant.name,
            role_label=participant.role_label,
            model=participant.model,
            provider_id=provider_config.provider_id,
            provider_display_name=provider_config.display_name,
            model_name=participant.model_name,
            prompt_source=prompt_source,
            prompt_used=system_prompt,
            status=TurnStatus.RUNNING,
        )
        session.turns.append(turn)
        self.session_store.upsert_session(session)

        try:
            response = await provider.generate(
                ProviderRequest(
                    provider_id=provider_config.provider_id,
                    provider_display_name=provider_config.display_name,
                    model_name=participant.model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=request.settings.max_output_tokens,
                    temperature=request.settings.temperature,
                    base_url=request.credentials.base_urls.get(provider_config.provider_id, provider_config.base_url),
                    endpoint_path=provider_config.endpoint_path,
                    request_headers=provider_config.request_headers,
                    metadata={
                        "session_id": session.session_id,
                        "participant_id": participant.participant_id,
                        "cycle_number": cycle_number,
                        "turn_number": turn_number,
                    },
                )
            )
        except ProviderClientException as exc:
            turn.status = TurnStatus.ERROR
            turn.error = exc.error.model_copy(
                update={
                    "participant_id": participant.participant_id,
                    "speaker": participant.name,
                    "provider_id": provider_config.provider_id,
                    "model_name": participant.model_name,
                }
            )
            session.errors.append(turn.error)
            turn.touch()
            self.session_store.upsert_session(session)
            return turn

        turn.status = TurnStatus.COMPLETED
        turn.content = response.text
        turn.request_id = response.request_id
        turn.usage = response.usage
        turn.touch()
        self.session_store.upsert_session(session)
        return turn

    def _build_prompt_context(
        self,
        session: SessionRecord,
        participant: DiscussionParticipant,
        cycle_number: int,
        latest_synthesis: SynthesisData | None,
    ) -> dict[str, Any]:
        return {
            "question": session.question,
            "discussion_id": session.discussion_id or "",
            "discussion_display_name": session.discussion_display_name or "",
            "cycle_number": cycle_number,
            "max_turn_cycles": session.settings.max_turn_cycles,
            "speaker": participant.name,
            "role_label": participant.role_label or "",
            "model": participant.model,
            "participants": [item.model_dump(mode="json") for item in session.participants],
            "turns": [turn.model_dump(mode="json") for turn in session.turns],
            "transcript": self._build_transcript(session.turns),
            "latest_synthesis": latest_synthesis.model_dump(mode="json") if latest_synthesis else None,
        }

    def _build_user_prompt(
        self,
        session: SessionRecord,
        participant: DiscussionParticipant,
        cycle_number: int,
        latest_synthesis: SynthesisData | None,
    ) -> str:
        roster = "\n".join(
            f"{index + 1}. {item.name} | {item.model} | {item.role_label or 'Participant'}"
            for index, item in enumerate(sorted(session.participants, key=lambda entry: (entry.sort_order, entry.participant_id)))
            if item.enabled
        )
        transcript = self._build_transcript(session.turns)
        latest = (
            json.dumps(latest_synthesis.model_dump(mode="json"), indent=2, ensure_ascii=True)
            if latest_synthesis
            else "None"
        )
        return "\n\n".join(
            [
                f"Question:\n{session.question}",
                f"Current cycle: {cycle_number} / {session.settings.max_turn_cycles}",
                f"Participant order:\n{roster}",
                f"Latest synthesis:\n{latest}",
                "Discussion transcript so far:\n" + (transcript or "No prior turns yet."),
                (
                    f"You are {participant.name}. Speak as the next participant in the dialog. "
                    "React to earlier messages where useful: support them, challenge them, correct them, or propose alternatives."
                ),
            ]
        )

    def _build_transcript(self, turns: list[SessionTurn]) -> str:
        lines: list[str] = []
        for turn in turns:
            if turn.status != TurnStatus.COMPLETED or not turn.content:
                continue
            label = turn.role_label or "Participant"
            lines.append(
                f"Turn {turn.turn_number} | Cycle {turn.cycle_number} | {turn.speaker} | {label} | {turn.model}\n{turn.content}"
            )
        return "\n\n".join(lines).strip()

    def _parse_synthesizer_output(self, turn: SessionTurn) -> SynthesisData:
        raw_output = turn.content or ""
        parsed_json = self._extract_json_object(raw_output)
        if parsed_json is not None:
            final_answer = str(parsed_json.get("final_answer", "")).strip() or raw_output.strip()
            return SynthesisData(
                speaker=turn.speaker,
                final_answer=final_answer,
                unresolved_disagreements=self._coerce_list(parsed_json.get("unresolved_disagreements")),
                open_questions=self._coerce_list(parsed_json.get("open_questions")),
                continue_discussion=bool(parsed_json.get("continue_discussion", False)),
                raw_output=raw_output,
            )

        partial_json = self._extract_partial_json_fields(raw_output)
        if partial_json is not None:
            return SynthesisData(
                speaker=turn.speaker,
                final_answer=partial_json["final_answer"] or raw_output.strip(),
                unresolved_disagreements=partial_json["unresolved_disagreements"],
                open_questions=partial_json["open_questions"],
                continue_discussion=partial_json["continue_discussion"],
                raw_output=raw_output,
            )

        fallback = self._parse_plaintext_sections(raw_output)
        return SynthesisData(
            speaker=turn.speaker,
            final_answer=fallback["final_answer"] or raw_output.strip(),
            unresolved_disagreements=fallback["unresolved_disagreements"],
            open_questions=fallback["open_questions"],
            continue_discussion=fallback["continue_discussion"],
            raw_output=raw_output,
        )

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        fenced_start = text.find("```json")
        if fenced_start != -1:
            fenced_end = text.find("```", fenced_start + 7)
            if fenced_end != -1:
                candidate = text[fenced_start + 7 : fenced_end].strip()
                try:
                    value = json.loads(candidate)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass

        start_index = text.find("{")
        while start_index != -1:
            depth = 0
            for index in range(start_index, len(text)):
                char = text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_index : index + 1]
                        try:
                            value = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if isinstance(value, dict):
                            return value
                        break
            start_index = text.find("{", start_index + 1)
        return None

    def _extract_partial_json_fields(self, text: str) -> dict[str, Any] | None:
        if '"final_answer"' not in text:
            return None

        final_answer = self._extract_json_string_field(text, "final_answer")
        unresolved = self._extract_json_string_array_field(text, "unresolved_disagreements")
        open_questions = self._extract_json_string_array_field(text, "open_questions")
        continue_discussion = self._extract_json_bool_field(text, "continue_discussion")

        if not any([final_answer, unresolved, open_questions, continue_discussion]):
            return None

        return {
            "final_answer": final_answer,
            "unresolved_disagreements": unresolved,
            "open_questions": open_questions,
            "continue_discussion": continue_discussion,
        }

    def _extract_json_string_field(self, text: str, field_name: str) -> str:
        marker = f'"{field_name}"'
        start = text.find(marker)
        if start == -1:
            return ""
        colon = text.find(":", start + len(marker))
        if colon == -1:
            return ""
        value_start = text.find('"', colon)
        if value_start == -1:
            return ""

        escaped = False
        chars: list[str] = []
        for index in range(value_start + 1, len(text)):
            char = text[index]
            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                return "".join(chars).strip()
            chars.append(char)
        return "".join(chars).strip()

    def _extract_json_string_array_field(self, text: str, field_name: str) -> list[str]:
        marker = f'"{field_name}"'
        start = text.find(marker)
        if start == -1:
            return []
        colon = text.find(":", start + len(marker))
        if colon == -1:
            return []
        array_start = text.find("[", colon)
        if array_start == -1:
            return []

        items: list[str] = []
        current: list[str] = []
        in_string = False
        escaped = False
        for index in range(array_start + 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    current.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    item = "".join(current).strip()
                    if item:
                        items.append(item)
                    current = []
                    in_string = False
                    continue
                current.append(char)
                continue
            if char == "]":
                break
            if char == '"':
                in_string = True

        if in_string and current:
            item = "".join(current).strip()
            if item:
                items.append(item)
        return items

    def _extract_json_bool_field(self, text: str, field_name: str) -> bool:
        marker = f'"{field_name}"'
        start = text.find(marker)
        if start == -1:
            return False
        colon = text.find(":", start + len(marker))
        if colon == -1:
            return False
        remainder = text[colon + 1 :].lstrip()
        if remainder.startswith("true"):
            return True
        if remainder.startswith("false"):
            return False
        return False

    def _parse_plaintext_sections(self, text: str) -> dict[str, Any]:
        sections: dict[str, Any] = {
            "final_answer": "",
            "unresolved_disagreements": [],
            "open_questions": [],
            "continue_discussion": False,
        }
        current_key: str | None = None
        labels = {
            "FINAL_ANSWER:": "final_answer",
            "UNRESOLVED_DISAGREEMENTS:": "unresolved_disagreements",
            "OPEN_QUESTIONS:": "open_questions",
            "CONTINUE_DISCUSSION:": "continue_discussion",
        }

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line and current_key in {"unresolved_disagreements", "open_questions"}:
                continue
            matched = False
            for label, key in labels.items():
                if line.upper().startswith(label):
                    current_key = key
                    value = raw_line.split(":", 1)[1].strip() if ":" in raw_line else ""
                    if key in {"unresolved_disagreements", "open_questions"}:
                        if value:
                            sections[key].append(value.lstrip("- ").strip())
                    elif key == "continue_discussion":
                        sections[key] = value.lower() in {"true", "yes", "1"}
                    else:
                        sections[key] = value
                    matched = True
                    break
            if matched or current_key is None:
                continue
            if current_key in {"unresolved_disagreements", "open_questions"}:
                sections[current_key].append(line.lstrip("- ").strip())
            elif current_key == "final_answer":
                sections[current_key] = (sections[current_key] + "\n" + raw_line).strip()
            elif current_key == "continue_discussion":
                sections[current_key] = line.lower() in {"true", "yes", "1"}

        sections["unresolved_disagreements"] = [item for item in sections["unresolved_disagreements"] if item]
        sections["open_questions"] = [item for item in sections["open_questions"] if item]
        return sections

    def _coerce_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

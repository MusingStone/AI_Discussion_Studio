from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
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

WEREWOLF_DISCUSSION_ID = "werewolf_table"
WEREWOLF_SYSTEM_ROLES = {"narrator", "vote_counter"}
LANGUAGE_ZH = "zh"
LANGUAGE_EN = "en"


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

        session.status = SessionStatus.RUNNING
        self.session_store.upsert_session(session)

        try:
            if self._is_werewolf_discussion(request.discussion_id):
                await self._run_werewolf_session(session, request)
                return
            await self._run_standard_session(session, request)
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

    async def _run_standard_session(self, session: SessionRecord, request: DiscussionRequest) -> None:
        resolved_participants = self._resolve_participants(request.participants)
        latest_synthesis: SynthesisData | None = None
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

    async def _run_werewolf_session(self, session: SessionRecord, request: DiscussionRequest) -> None:
        resolved_participants = self._resolve_participants(request.participants)
        participant_index = {resolved.participant.participant_id: resolved for resolved in resolved_participants}
        role_index = self._build_werewolf_role_index(resolved_participants)
        narrator = self._first_or_none(role_index.get("narrator", []))
        if narrator is None:
            raise ValueError("Werewolf mode requires a narrator participant.")

        state = self._initialize_werewolf_state(resolved_participants)
        state["question"] = session.question
        await self._confirm_werewolf_identities(
            session=session,
            request=request,
            resolved_participants=resolved_participants,
            state=state,
        )
        participant_index = {resolved.participant.participant_id: resolved for resolved in resolved_participants}
        role_index = self._build_werewolf_role_index(resolved_participants)
        session.engine_state = state
        self._sync_werewolf_public_state(session)
        self.session_store.upsert_session(session)

        for cycle_number in range(1, request.settings.max_turn_cycles + 1):
            if self._winner_from_state(state):
                break

            state["round_number"] = cycle_number
            state["phase"] = "night"
            self._sync_werewolf_public_state(session)
            self.session_store.upsert_session(session)

            night_result = await self._run_werewolf_night(
                session=session,
                request=request,
                cycle_number=cycle_number,
                role_index=role_index,
                participant_index=participant_index,
            )
            self._append_engine_turn(
                session=session,
                resolved=narrator,
                cycle_number=cycle_number,
                phase="night_result",
                content=night_result["public_summary"],
            )
            if self._winner_from_state(state):
                break

            state["phase"] = "day_discussion"
            self._sync_werewolf_public_state(session)
            self.session_store.upsert_session(session)

            self._append_engine_turn(
                session=session,
                resolved=narrator,
                cycle_number=cycle_number,
                phase="day_discussion",
                content=self._render_werewolf_discussion_opening(state, cycle_number),
            )

            for resolved in self._living_werewolf_players(state, participant_index):
                await self._run_werewolf_public_speech(
                    session=session,
                    request=request,
                    resolved=resolved,
                    cycle_number=cycle_number,
                    state=state,
                )

            if self._winner_from_state(state):
                break

            state["phase"] = "day_vote"
            self._sync_werewolf_public_state(session)
            self.session_store.upsert_session(session)

            self._append_engine_turn(
                session=session,
                resolved=narrator,
                cycle_number=cycle_number,
                phase="day_vote",
                content=self._t(
                    session.question,
                    f"第 {cycle_number} 天开始投票。\n所有存活玩家现在必须公开投票给一名存活玩家。",
                    f"Day {cycle_number} voting begins.\nEach living player must now cast one public vote against a living player.",
                ),
            )

            day_result = await self._run_werewolf_vote(
                session=session,
                request=request,
                cycle_number=cycle_number,
                role_index=role_index,
                participant_index=participant_index,
            )
            self._append_engine_turn(
                session=session,
                resolved=narrator,
                cycle_number=cycle_number,
                phase="day_resolution",
                content=day_result["public_summary"],
            )

            if self._winner_from_state(state):
                break

        self._append_engine_turn(
            session=session,
            resolved=narrator,
            cycle_number=max(1, state.get("round_number", 1)),
            phase="game_over",
            content=self._render_werewolf_verdict(state),
        )
        await self._run_werewolf_postgame_reflections(
            session=session,
            request=request,
            resolved_participants=resolved_participants,
            narrator=narrator,
            cycle_number=max(1, state.get("round_number", 1)),
        )
        self._finalize_werewolf_session(session)
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
        participant = resolved.participant
        prompt_source, system_prompt = self.prompt_loader.resolve_prompt(
            participant.prompt,
            self._build_prompt_context(session, participant, cycle_number, latest_synthesis),
        )
        user_prompt = self._build_user_prompt(session, participant, cycle_number, latest_synthesis)
        return await self._run_ai_turn(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            turn_number=turn_number,
            phase="discussion",
            prompt_source=prompt_source,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def _run_ai_turn(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        turn_number: int,
        phase: str,
        prompt_source: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
    ) -> SessionTurn:
        participant = resolved.participant
        provider_config = resolved.provider_config
        provider = build_provider(provider_config, request.credentials, self.settings)

        turn = SessionTurn(
            turn_id=f"turn_{turn_number}",
            cycle_number=cycle_number,
            turn_number=turn_number,
            phase=phase,
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
                    max_output_tokens=max_output_tokens or request.settings.max_output_tokens,
                    temperature=request.settings.temperature,
                    base_url=request.credentials.base_urls.get(provider_config.provider_id, provider_config.base_url),
                    endpoint_path=provider_config.endpoint_path,
                    request_headers=provider_config.request_headers,
                    metadata={
                        "session_id": session.session_id,
                        "participant_id": participant.participant_id,
                        "cycle_number": cycle_number,
                        "turn_number": turn_number,
                        "phase": phase,
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

    async def _generate_private_response(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        phase: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        record_errors: bool = True,
    ) -> str | None:
        provider_config = resolved.provider_config
        participant = resolved.participant
        provider = build_provider(provider_config, request.credentials, self.settings)
        try:
            response = await provider.generate(
                ProviderRequest(
                    provider_id=provider_config.provider_id,
                    provider_display_name=provider_config.display_name,
                    model_name=participant.model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens or request.settings.max_output_tokens,
                    temperature=request.settings.temperature,
                    base_url=request.credentials.base_urls.get(provider_config.provider_id, provider_config.base_url),
                    endpoint_path=provider_config.endpoint_path,
                    request_headers=provider_config.request_headers,
                    metadata={
                        "session_id": session.session_id,
                        "participant_id": participant.participant_id,
                        "cycle_number": cycle_number,
                        "phase": phase,
                        "visibility": "private",
                    },
                )
            )
        except ProviderClientException as exc:
            error = exc.error.model_copy(
                update={
                    "participant_id": participant.participant_id,
                    "speaker": participant.name,
                    "provider_id": provider_config.provider_id,
                    "model_name": participant.model_name,
                }
            )
            if record_errors:
                session.errors.append(error)
                self.session_store.upsert_session(session)
            return None
        return response.text

    def _append_engine_turn(
        self,
        *,
        session: SessionRecord,
        resolved: ResolvedParticipant,
        cycle_number: int,
        phase: str,
        content: str,
    ) -> SessionTurn:
        participant = resolved.participant
        provider_config = resolved.provider_config
        turn = SessionTurn(
            turn_id=f"turn_{len(session.turns) + 1}",
            cycle_number=cycle_number,
            turn_number=len(session.turns) + 1,
            phase=phase,
            participant_id=participant.participant_id,
            speaker=participant.name,
            role_label=participant.role_label,
            model=participant.model,
            provider_id=provider_config.provider_id,
            provider_display_name=provider_config.display_name,
            model_name=participant.model_name,
            prompt_source="engine:werewolf",
            prompt_used="Authoritative werewolf game event generated by the server.",
            status=TurnStatus.COMPLETED,
            content=content,
        )
        session.turns.append(turn)
        self.session_store.upsert_session(session)
        return turn

    def _append_participant_turn(
        self,
        *,
        session: SessionRecord,
        resolved: ResolvedParticipant,
        cycle_number: int,
        phase: str,
        content: str,
        prompt_source: str,
        prompt_used: str,
    ) -> SessionTurn:
        participant = resolved.participant
        provider_config = resolved.provider_config
        turn = SessionTurn(
            turn_id=f"turn_{len(session.turns) + 1}",
            cycle_number=cycle_number,
            turn_number=len(session.turns) + 1,
            phase=phase,
            participant_id=participant.participant_id,
            speaker=participant.name,
            role_label=participant.role_label,
            model=participant.model,
            provider_id=provider_config.provider_id,
            provider_display_name=provider_config.display_name,
            model_name=participant.model_name,
            prompt_source=prompt_source,
            prompt_used=prompt_used,
            status=TurnStatus.COMPLETED,
            content=content,
        )
        session.turns.append(turn)
        self.session_store.upsert_session(session)
        return turn

    async def _run_werewolf_public_speech(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        state: dict[str, Any],
    ) -> SessionTurn:
        participant = resolved.participant
        role = self._werewolf_role_for_participant(participant)
        context = self._build_prompt_context(session, participant, cycle_number, None)
        prompt_source, base_prompt = self.prompt_loader.resolve_prompt(participant.prompt, context)
        system_prompt = "\n\n".join(
            [
                base_prompt,
                "You are in a real werewolf game engine, not a free-form roleplay.",
                f"Current public phase: Day {cycle_number} discussion.",
                "Speak in plain text only. Do not output JSON.",
                self._language_instruction(session.question, public=True),
                "Do not invent secret moderator information you were not privately told.",
                "If you are already dead, say one short final line at most.",
                "Private briefing:",
                self._render_werewolf_private_briefing(state, participant.participant_id),
            ]
        ).strip()
        user_prompt = "\n\n".join(
            [
                f"Original setup question:\n{session.question}",
                f"Public board state:\n{self._render_werewolf_public_state_text(state)}",
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=10) or "No public turns yet."),
                "Personal memory window:\n" + self._render_werewolf_personal_memory_window(session, state, participant.participant_id),
                (
                    f"Speak as {participant.name} during the current day discussion. "
                    f"You are privately assigned the role '{role}'. Make a concrete case, suspicion, defense, or strategic read. "
                    f"{self._language_instruction(session.question, public=True)}"
                ),
            ]
        )
        return await self._run_ai_turn(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            turn_number=len(session.turns) + 1,
            phase="day_discussion",
            prompt_source=prompt_source,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=request.settings.max_output_tokens,
        )

    async def _run_werewolf_night(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        cycle_number: int,
        role_index: dict[str, list[ResolvedParticipant]],
        participant_index: dict[str, ResolvedParticipant],
    ) -> dict[str, Any]:
        state = session.engine_state
        narrator_label = self._role_display_name("narrator", session.question)
        alive_ids = list(state["alive_ids"])
        private_dialogue: list[dict[str, str]] = []
        wolf_ids = [resolved.participant.participant_id for resolved in role_index.get("werewolf", []) if resolved.participant.participant_id in alive_ids]
        eligible_wolf_targets = [player_id for player_id in alive_ids if player_id not in wolf_ids]
        living_wolves = [
            resolved
            for resolved in role_index.get("werewolf", [])
            if resolved.participant.participant_id in alive_ids
        ]
        wolf_questions = {
            resolved.participant.participant_id: self._night_narrator_question(
                state=state,
                role="werewolf",
                participant_name=resolved.participant.name,
            )
            for resolved in living_wolves
        }
        wolf_tasks = [
            self._request_wolf_night_action(
                session=session,
                request=request,
                resolved=resolved,
                cycle_number=cycle_number,
                eligible_target_ids=eligible_wolf_targets,
                living_wolf_ids=wolf_ids,
                narrator_question=wolf_questions[resolved.participant.participant_id],
            )
            for resolved in living_wolves
        ]
        seer_resolved = self._first_or_none(
            [
                resolved
                for resolved in role_index.get("seer", [])
                if resolved.participant.participant_id in alive_ids
            ]
        )
        seer_question = None
        seer_task = None
        if seer_resolved is not None:
            seer_question = self._night_narrator_question(
                state=state,
                role="seer",
                participant_name=seer_resolved.participant.name,
            )
            seer_task = self._request_seer_night_action(
                session=session,
                request=request,
                resolved=seer_resolved,
                cycle_number=cycle_number,
                eligible_target_ids=[player_id for player_id in alive_ids if player_id != seer_resolved.participant.participant_id],
                narrator_question=seer_question,
            )

        gathered: list[Any] = []
        if wolf_tasks and seer_task is not None:
            gathered = list(await asyncio.gather(*wolf_tasks, seer_task))
        elif wolf_tasks:
            gathered = list(await asyncio.gather(*wolf_tasks))
        elif seer_task is not None:
            gathered = [await seer_task]

        wolf_actions = gathered[: len(wolf_tasks)] if wolf_tasks else []
        wolf_votes: list[str] = []
        wolf_notes: list[dict[str, str]] = []
        for resolved, action in zip(living_wolves, wolf_actions):
            participant_id = resolved.participant.participant_id
            self._append_private_dialogue(private_dialogue, narrator_label, wolf_questions[participant_id])
            self._append_private_dialogue(private_dialogue, resolved.participant.name, action["reply"])
            if action["target_id"]:
                wolf_votes.append(action["target_id"])
            wolf_notes.append(
                {
                    "wolf": resolved.participant.name,
                    "target": self._player_name(state, action["target_id"]) if action["target_id"] else "no valid target",
                    "reply": action["reply"],
                }
            )
        wolf_target_id = self._choose_weighted_target(
            proposed_ids=wolf_votes,
            eligible_ids=eligible_wolf_targets,
            ordered_ids=eligible_wolf_targets,
        )

        seer_result: dict[str, Any] | None = None
        if seer_resolved is not None and seer_task is not None:
            participant_id = seer_resolved.participant.participant_id
            seer_result = gathered[-1]
            self._append_private_dialogue(private_dialogue, narrator_label, seer_question or "")
            self._append_private_dialogue(private_dialogue, seer_resolved.participant.name, seer_result["reply"])
            if seer_result["target_id"]:
                alignment = "werewolf" if state["roles"][seer_result["target_id"]]["team"] == "wolves" else "not_werewolf"
                seer_record = {
                    "round_number": cycle_number,
                    "seer_id": participant_id,
                    "target_id": seer_result["target_id"],
                    "target_name": self._player_name(state, seer_result["target_id"]),
                    "alignment": alignment,
                    "reply": seer_result.get("reply", ""),
                }
                state.setdefault("seer_results", []).append(seer_record)
                seer_result["seer_id"] = participant_id

        witch_action = {"save_target": None, "poison_target": None, "reply": ""}
        for resolved in role_index.get("witch", []):
            participant_id = resolved.participant.participant_id
            if participant_id not in alive_ids:
                continue
            witch_question = self._night_narrator_question(
                state=state,
                role="witch",
                participant_name=resolved.participant.name,
                wolf_target_id=wolf_target_id,
            )
            witch_action = await self._request_witch_night_action(
                session=session,
                request=request,
                resolved=resolved,
                cycle_number=cycle_number,
                wolf_target_id=wolf_target_id,
                eligible_poison_ids=[player_id for player_id in alive_ids if player_id != participant_id],
                narrator_question=witch_question,
            )
            self._append_private_dialogue(private_dialogue, narrator_label, witch_question)
            self._append_private_dialogue(private_dialogue, resolved.participant.name, witch_action["reply"])
            break

        witch_state = state["witch_state"]
        saved_target_id = None
        if witch_action["save_target"] == wolf_target_id and witch_state["antidote_available"]:
            saved_target_id = wolf_target_id
            witch_state["antidote_available"] = False

        poison_target_id = None
        if witch_action["poison_target"] in alive_ids and witch_state["poison_available"]:
            poison_target_id = witch_action["poison_target"]
            witch_state["poison_available"] = False

        death_causes: dict[str, list[str]] = {}
        if wolf_target_id and wolf_target_id != saved_target_id:
            death_causes.setdefault(wolf_target_id, []).append("werewolf_attack")
        if poison_target_id:
            death_causes.setdefault(poison_target_id, []).append("witch_poison")

        hunter_chain = await self._resolve_hunter_shots(
            session=session,
            request=request,
            cycle_number=cycle_number,
            dead_player_ids=list(death_causes.keys()),
            participant_index=participant_index,
            cause_map=death_causes,
            private_dialogue=private_dialogue,
        )

        self._apply_deaths(state, death_causes.keys())
        public_summary = self._render_werewolf_night_summary(
            state=state,
            cycle_number=cycle_number,
            dead_player_ids=list(death_causes.keys()),
            saved_target_id=saved_target_id,
            hunter_chain=hunter_chain,
        )
        night_record = {
            "round_number": cycle_number,
            "wolf_votes": wolf_notes,
            "wolf_target_id": wolf_target_id,
            "seer_result": seer_result,
            "witch_actor": self._first_or_none(role_index.get("witch", [])).participant.name if self._first_or_none(role_index.get("witch", [])) else None,
            "witch_reply": witch_action.get("reply", ""),
            "saved_target_id": saved_target_id,
            "poison_target_id": poison_target_id,
            "dead_player_ids": list(death_causes.keys()),
            "death_causes": death_causes,
            "hunter_chain": hunter_chain,
            "private_dialogue": private_dialogue,
            "public_summary": public_summary,
        }
        private_summary = self._render_werewolf_private_night_summary(state, night_record)
        state.setdefault("night_history", []).append(night_record)
        state.setdefault("night_private_history", []).append(
            {
                "round_number": cycle_number,
                "summary": private_summary,
            }
        )
        state["last_night_summary"] = public_summary
        state["latest_night_private"] = private_summary
        state["winner"] = self._determine_werewolf_winner(state)
        self._sync_werewolf_public_state(session)
        self.session_store.upsert_session(session)
        return night_record

    async def _run_werewolf_vote(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        cycle_number: int,
        role_index: dict[str, list[ResolvedParticipant]],
        participant_index: dict[str, ResolvedParticipant],
    ) -> dict[str, Any]:
        state = session.engine_state

        votes: dict[str, str] = {}
        for resolved in self._living_werewolf_players(state, participant_index):
            voter_id = resolved.participant.participant_id
            eligible_ids = [player_id for player_id in state["alive_ids"] if player_id != voter_id]
            action = await self._request_public_vote(
                session=session,
                request=request,
                resolved=resolved,
                cycle_number=cycle_number,
                eligible_target_ids=eligible_ids,
            )
            target_id = action["target_id"] or self._fallback_vote_target(state, voter_id)
            if target_id is None:
                continue
            votes[voter_id] = target_id
            reason = action["reason"] or self._t(
                session.question,
                "这是我今天最应该出的目标。",
                "This target is the strongest elimination candidate for today.",
            )
            self._append_engine_turn(
                session=session,
                resolved=resolved,
                cycle_number=cycle_number,
                phase="day_vote",
                content=self._t(
                    session.question,
                    f"我投票淘汰 {self._player_name(state, target_id)}。{reason}",
                    f"I vote to eliminate {self._player_name(state, target_id)}. {reason}",
                ),
            )

        tally = Counter(votes.values())
        ordered_alive = [player_id for player_id in state["player_order"] if player_id in state["alive_ids"]]
        eliminated_id = None
        tie_names: list[str] = []
        if tally:
            top_votes = max(tally.values())
            leaders = [player_id for player_id in ordered_alive if tally.get(player_id, 0) == top_votes]
            if len(leaders) == 1:
                eliminated_id = leaders[0]
            else:
                tie_names = [self._player_name(state, player_id) for player_id in leaders]

        death_causes: dict[str, list[str]] = {}
        final_words: SessionTurn | None = None
        if eliminated_id:
            death_causes[eliminated_id] = ["day_vote"]
            eliminated_resolved = participant_index.get(eliminated_id)
            if eliminated_resolved is not None:
                final_words = await self._run_werewolf_final_words_turn(
                    session=session,
                    request=request,
                    resolved=eliminated_resolved,
                    cycle_number=cycle_number,
                    state=state,
                )

        hunter_chain = await self._resolve_hunter_shots(
            session=session,
            request=request,
            cycle_number=cycle_number,
            dead_player_ids=list(death_causes.keys()),
            participant_index=participant_index,
            cause_map=death_causes,
        )
        self._apply_deaths(state, death_causes.keys())

        public_summary = self._render_werewolf_vote_summary(
            state=state,
            cycle_number=cycle_number,
            votes=votes,
            tally=tally,
            eliminated_id=eliminated_id,
            tie_names=tie_names,
            hunter_chain=hunter_chain,
        )
        day_record = {
            "round_number": cycle_number,
            "votes": votes,
            "tally": dict(tally),
            "eliminated_id": eliminated_id,
            "final_words": final_words.content if final_words is not None else "",
            "tie_names": tie_names,
            "death_causes": death_causes,
            "hunter_chain": hunter_chain,
            "public_summary": public_summary,
        }
        state.setdefault("day_history", []).append(day_record)
        state["last_vote_summary"] = public_summary
        state["winner"] = self._determine_werewolf_winner(state)
        self._sync_werewolf_public_state(session)
        self.session_store.upsert_session(session)

        return day_record

    async def _run_werewolf_final_words_turn(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        state: dict[str, Any],
    ) -> SessionTurn:
        participant = resolved.participant
        system_prompt = "\n\n".join(
            [
                f"You are {participant.name} in a real werewolf game.",
                "You were just eliminated by the public day vote and may leave one short final statement.",
                "Speak in plain text only. Do not output JSON.",
                "Keep it brief: one or two short sentences.",
                "Do not directly reveal your exact hidden role, even if you want to defend yourself.",
                self._language_instruction(session.question, public=True),
            ]
        ).strip()
        user_prompt = "\n\n".join(
            [
                f"Original setup question:\n{session.question}",
                f"Public board state:\n{self._render_werewolf_public_state_text(state)}",
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=10) or "No public turns yet."),
                "Personal memory window:\n" + self._render_werewolf_personal_memory_window(session, state, participant.participant_id),
                self._t(
                    session.question,
                    f"你刚刚被白天公投出局。现在请留下简短遗言，不能直接公开自己的确切身份。",
                    f"You have just been eliminated by the day vote. Leave brief final words without directly revealing your exact role.",
                ),
            ]
        )
        response_text = await self._generate_private_response(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="day_final_words",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=min(request.settings.max_output_tokens, 120),
            record_errors=False,
        )
        content = (response_text or "").strip() or self._t(
            session.question,
            "我的遗言很简单：请继续看清票型和发言，不要被表面节奏带走。",
            "My final words are simple: keep reading the votes and speeches, and do not be dragged by surface momentum.",
        )
        return self._append_participant_turn(
            session=session,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="day_resolution",
            content=content,
            prompt_source="engine:werewolf-final-words",
            prompt_used=system_prompt,
        )

    async def _resolve_hunter_shots(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        cycle_number: int,
        dead_player_ids: list[str],
        participant_index: dict[str, ResolvedParticipant],
        cause_map: dict[str, list[str]],
        private_dialogue: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        state = session.engine_state
        narrator_label = self._role_display_name("narrator", state.get("question", ""))
        chain: list[dict[str, str]] = []
        if state.get("hunter_shot_used"):
            return chain

        for dead_id in list(dead_player_ids):
            role = state["roles"].get(dead_id, {}).get("role")
            if role != "hunter":
                continue
            resolved = participant_index.get(dead_id)
            if resolved is None:
                continue
            eligible_ids = [
                player_id
                for player_id in state["alive_ids"]
                if player_id != dead_id and player_id not in cause_map
            ]
            if not eligible_ids:
                state["hunter_shot_used"] = True
                break
            narrator_question = self._night_narrator_question(
                state=state,
                role="hunter",
                participant_name=resolved.participant.name,
            )
            action = await self._request_hunter_shot(
                session=session,
                request=request,
                resolved=resolved,
                cycle_number=cycle_number,
                eligible_target_ids=eligible_ids,
                narrator_question=narrator_question,
            )
            if private_dialogue is not None:
                self._append_private_dialogue(private_dialogue, narrator_label, narrator_question)
                self._append_private_dialogue(private_dialogue, resolved.participant.name, action["reply"])
            target_id = action["target_id"] or eligible_ids[0]
            cause_map.setdefault(target_id, []).append("hunter_shot")
            chain.append(
                {
                    "hunter_name": self._player_name(state, dead_id),
                    "target_name": self._player_name(state, target_id),
                    "target_id": target_id,
                    "reply": action["reply"],
                }
            )
            state["hunter_shot_used"] = True
            break
        return chain

    async def _request_wolf_night_action(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        eligible_target_ids: list[str],
        living_wolf_ids: list[str],
        narrator_question: str,
    ) -> dict[str, str | None]:
        state = session.engine_state
        target_lines = self._render_player_choices(state, eligible_target_ids)
        partner_lines = self._render_player_choices(
            state,
            [player_id for player_id in living_wolf_ids if player_id != resolved.participant.participant_id],
        )
        system_prompt = "\n".join(
            [
                f"You are {resolved.participant.name}, a werewolf acting during the secret night phase.",
                "Return JSON only in this shape:",
                '{"target_player": "participant_id", "reply": "very short spoken answer"}',
                "Choose exactly one living non-wolf player to attack.",
                "Assume the all-knowing Narrator just asked you what to do tonight.",
                "The reply must be extremely short and sound like a direct answer to the Narrator.",
                self._language_instruction(session.question, public=False),
                "Do not include markdown fences.",
            ]
        )
        user_prompt = "\n\n".join(
            [
                f"Night {cycle_number}",
                f"Private narrator message:\n{narrator_question}",
                "Living non-wolf targets:\n" + (target_lines or "None"),
                "Living wolf partners:\n" + (partner_lines or "None"),
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=8) or "No public turns yet."),
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="night_wolf_action",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=48,
        )
        return {
            "target_id": self._coerce_player_id(payload.get("target_player"), eligible_target_ids, state),
            "reply": str(payload.get("reply", "")).strip() or self._night_reply_fallback("werewolf", state, payload.get("target_player")),
        }

    async def _request_seer_night_action(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        eligible_target_ids: list[str],
        narrator_question: str,
    ) -> dict[str, str | None]:
        state = session.engine_state
        system_prompt = "\n".join(
            [
                f"You are {resolved.participant.name}, the Seer, acting during the secret night phase.",
                "Return JSON only in this shape:",
                '{"target_player": "participant_id", "reply": "very short spoken answer"}',
                "Choose one living player to inspect tonight.",
                "Assume the all-knowing Narrator just asked who you want to inspect.",
                "The reply must be extremely short and sound like a direct answer to the Narrator.",
                self._language_instruction(session.question, public=False),
                "Do not include markdown fences.",
            ]
        )
        user_prompt = "\n\n".join(
            [
                f"Night {cycle_number}",
                f"Private narrator message:\n{narrator_question}",
                "Eligible inspection targets:\n" + self._render_player_choices(state, eligible_target_ids),
                "Your previous private checks:\n" + self._render_seer_results(state),
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=8) or "No public turns yet."),
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="night_seer_action",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=48,
        )
        return {
            "target_id": self._coerce_player_id(payload.get("target_player"), eligible_target_ids, state),
            "reply": str(payload.get("reply", "")).strip() or self._night_reply_fallback("seer", state, payload.get("target_player")),
        }

    async def _request_witch_night_action(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        wolf_target_id: str | None,
        eligible_poison_ids: list[str],
        narrator_question: str,
    ) -> dict[str, str | None]:
        state = session.engine_state
        witch_state = state["witch_state"]
        save_target_text = self._player_name(state, wolf_target_id) if wolf_target_id else "none"
        system_prompt = "\n".join(
            [
                f"You are {resolved.participant.name}, the Witch, acting during the secret night phase.",
                "Return JSON only in this shape:",
                '{"save_target": "participant_id or null", "poison_target": "participant_id or null", "reply": "very short spoken answer"}',
                "Use save_target only if you want to spend the antidote on the wolf victim.",
                "Use poison_target only if you want to poison one living player.",
                "Assume the all-knowing Narrator just told you tonight's wolf victim and asked for your decision.",
                "The reply must be extremely short and sound like a direct answer to the Narrator.",
                self._language_instruction(session.question, public=False),
                "Do not include markdown fences.",
            ]
        )
        user_prompt = "\n\n".join(
            [
                f"Night {cycle_number}",
                f"Private narrator message:\n{narrator_question}",
                f"Current wolf victim: {save_target_text}",
                f"Antidote available: {'yes' if witch_state['antidote_available'] else 'no'}",
                f"Poison available: {'yes' if witch_state['poison_available'] else 'no'}",
                "Eligible poison targets:\n" + self._render_player_choices(state, eligible_poison_ids),
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=8) or "No public turns yet."),
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="night_witch_action",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=64,
        )
        return {
            "save_target": self._coerce_player_id([payload.get("save_target"), payload.get("save_player")], [wolf_target_id] if wolf_target_id else [], state),
            "poison_target": self._coerce_player_id(payload.get("poison_target"), eligible_poison_ids, state),
            "reply": str(payload.get("reply", "")).strip()
            or self._night_reply_fallback(
                "witch",
                state,
                [payload.get("save_target"), payload.get("poison_target")],
            ),
        }

    async def _request_hunter_shot(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        eligible_target_ids: list[str],
        narrator_question: str,
    ) -> dict[str, str | None]:
        state = session.engine_state
        system_prompt = "\n".join(
            [
                f"You are {resolved.participant.name}, the Hunter, and you have just died.",
                "Return JSON only in this shape:",
                '{"target_player": "participant_id or null", "reply": "very short spoken answer"}',
                "Choose one living player to shoot as your final action, or null if no valid shot exists.",
                "Assume the all-knowing Narrator just asked for your final shot.",
                "The reply must be extremely short and sound like a direct answer to the Narrator.",
                self._language_instruction(session.question, public=False),
                "Do not include markdown fences.",
            ]
        )
        user_prompt = "\n\n".join(
            [
                f"Round {cycle_number}",
                f"Private narrator message:\n{narrator_question}",
                "Eligible targets:\n" + self._render_player_choices(state, eligible_target_ids),
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=8) or "No public turns yet."),
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="hunter_last_shot",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=48,
        )
        return {
            "target_id": self._coerce_player_id(payload.get("target_player"), eligible_target_ids, state),
            "reply": str(payload.get("reply", "")).strip() or self._night_reply_fallback("hunter", state, payload.get("target_player")),
        }

    async def _request_public_vote(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        eligible_target_ids: list[str],
    ) -> dict[str, str | None]:
        state = session.engine_state
        system_prompt = "\n".join(
            [
                f"You are {resolved.participant.name}, casting your public day vote in a werewolf game.",
                "Return JSON only in this shape:",
                '{"vote_target": "participant_id", "reason": "one short public justification"}',
                "Choose exactly one living player other than yourself.",
                self._language_instruction(session.question, public=True),
                "Do not include markdown fences.",
            ]
        )
        user_prompt = "\n\n".join(
            [
                f"Day {cycle_number} vote phase.",
                "Eligible vote targets:\n" + self._render_player_choices(state, eligible_target_ids),
                "Your private briefing:\n" + self._render_werewolf_private_briefing(state, resolved.participant.participant_id),
                "Personal memory window:\n" + self._render_werewolf_personal_memory_window(session, state, resolved.participant.participant_id),
                "Recent public transcript:\n" + (self._build_recent_transcript(session.turns, limit=10) or "No public turns yet."),
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="day_vote_choice",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=request.settings.max_output_tokens,
        )
        return {
            "target_id": self._coerce_player_id(payload.get("vote_target"), eligible_target_ids, state),
            "reason": str(payload.get("reason", "")).strip(),
        }

    async def _request_json_payload(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
        phase: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        response_text = await self._generate_private_response(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase=phase,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )
        if not response_text:
            return {}
        payload = self._extract_json_object(response_text)
        return payload if payload is not None else {}

    async def _confirm_werewolf_identities(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved_participants: list[ResolvedParticipant],
        state: dict[str, Any],
    ) -> None:
        state["phase"] = "identity_setup"
        checks = [
            self._request_identity_confirmation(
                session=session,
                request=request,
                resolved=resolved,
                state=state,
            )
            for resolved in resolved_participants
            if self._werewolf_role_for_participant(resolved.participant) not in WEREWOLF_SYSTEM_ROLES
        ]
        results = await asyncio.gather(*checks) if checks else []

        confirmations: list[dict[str, Any]] = []
        for result in results:
            participant_id = result["participant_id"]
            assigned_role = state["roles"][participant_id]["role"]
            confirmed_role = result["confirmed_role"] or assigned_role
            state["roles"][participant_id]["role"] = confirmed_role
            state["roles"][participant_id]["team"] = self._werewolf_team_for_role(confirmed_role)
            confirmations.append(
                {
                    "participant_id": participant_id,
                    "name": result["name"],
                    "assigned_role": assigned_role,
                    "confirmed_role": confirmed_role,
                    "matched": assigned_role == confirmed_role,
                    "question": result["question"],
                    "reply": result["reply"],
                    "raw_role": result["raw_role"],
                }
            )
        state["identity_confirmations"] = confirmations

    async def _request_identity_confirmation(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        participant = resolved.participant
        context = self._build_prompt_context(session, participant, 0, None)
        prompt_source, base_prompt = self.prompt_loader.resolve_prompt(participant.prompt, context)
        narrator_question = self._identity_confirmation_question(session.question, participant.name)
        system_prompt = "\n\n".join(
            [
                base_prompt,
                "This is a private pre-game identity check from the Narrator.",
                'Return JSON only in this shape: {"role": "hidden role", "reply": "one very short spoken reply"}',
                "State only your own hidden role in this game.",
                "Do not choose actions yet.",
                self._language_instruction(session.question, public=False),
                "Do not include markdown fences.",
            ]
        ).strip()
        user_prompt = "\n\n".join(
            [
                f"Original setup question:\n{session.question}",
                f"Narrator private question:\n{narrator_question}",
                "Reply with your hidden role and one very short spoken confirmation.",
            ]
        )
        payload = await self._request_json_payload(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=0,
            phase="identity_confirmation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=48,
        )
        raw_role = str(
            payload.get("role")
            or payload.get("identity")
            or payload.get("claimed_role")
            or ""
        ).strip()
        reply = str(payload.get("reply") or "").strip()
        confirmed_role = self._canonicalize_werewolf_role(raw_role or reply)
        if not reply:
            fallback_role = confirmed_role or state["roles"][participant.participant_id]["role"]
            reply = self._identity_confirmation_reply(session.question, participant.name, fallback_role)
        return {
            "participant_id": participant.participant_id,
            "name": participant.name,
            "question": narrator_question,
            "reply": reply,
            "raw_role": raw_role,
            "confirmed_role": confirmed_role,
            "prompt_source": prompt_source,
        }

    async def _run_werewolf_postgame_reflections(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved_participants: list[ResolvedParticipant],
        narrator: ResolvedParticipant,
        cycle_number: int,
    ) -> None:
        state = session.engine_state
        question = state.get("question", session.question)
        state["phase"] = "post_game_reflection"
        self._reveal_all_werewolf_roles(state)
        self._sync_werewolf_public_state(session)
        self.session_store.upsert_session(session)

        votes: list[dict[str, str]] = []
        for resolved in resolved_participants:
            turn, vote = await self._run_werewolf_postgame_reflection_turn(
                session=session,
                request=request,
                resolved=resolved,
                cycle_number=cycle_number,
            )
            if vote:
                votes.append(vote)

        state["postgame_votes"] = votes
        summary = self._render_werewolf_postgame_mvp_summary(state, votes)
        state["postgame_mvp_summary"] = summary
        self._append_engine_turn(
            session=session,
            resolved=narrator,
            cycle_number=cycle_number,
            phase="post_game_mvp",
            content=summary,
        )
        self._sync_werewolf_public_state(session)
        self.session_store.upsert_session(session)

    async def _run_werewolf_postgame_reflection_turn(
        self,
        *,
        session: SessionRecord,
        request: DiscussionRequest,
        resolved: ResolvedParticipant,
        cycle_number: int,
    ) -> tuple[SessionTurn | None, dict[str, str] | None]:
        state = session.engine_state
        participant = resolved.participant
        eligible_ids = [
            player_id
            for player_id in state["player_order"]
            if player_id != participant.participant_id
        ]
        if not eligible_ids:
            eligible_ids = list(state["player_order"])
        role_name = self._role_display_name(state["roles"][participant.participant_id]["role"], session.question)
        system_prompt = "\n\n".join(
            [
                "The werewolf game is over. All roles are now public.",
                'Return JSON only in this shape: {"reflection": "short post-game feeling", "mvp_player": "participant_id", "reason": "speech-based evaluation"}',
                "Give one short post-game reaction, then vote for the single player who played best.",
                "Your MVP reason must explicitly evaluate that player's public speaking, logic, influence, pressure handling, or vote impact in this game.",
                "Keep it concise and decisive.",
                "Choose one real player, not yourself if avoidable.",
                self._language_instruction(session.question, public=True),
                "Do not include markdown fences.",
            ]
        ).strip()
        user_prompt = "\n\n".join(
            [
                f"You are {participant.name}. Your revealed role is {role_name}.",
                f"Winner:\n{state.get('winner') or 'No decisive winner'}",
                "Compact game recap:\n" + self._render_werewolf_postgame_recap(state),
                "Public speech highlights:\n" + self._render_werewolf_postgame_speech_highlights(session, state),
                "Eligible MVP choices:\n" + self._render_player_choices(state, eligible_ids),
                "Publish your short post-game feeling and cast one MVP vote based on the actual speeches and influence shown above.",
            ]
        )
        response_text = await self._generate_private_response(
            session=session,
            request=request,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="post_game_reflection",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=120,
            record_errors=False,
        )
        if response_text is None:
            return None, None
        payload = self._extract_json_object(response_text or "") or {}
        reflection = str(payload.get("reflection", "")).strip()
        mvp_id = self._coerce_player_id(payload.get("mvp_player"), eligible_ids, state)
        reason = str(payload.get("reason", "")).strip()
        if not reflection:
            reflection = self._fallback_postgame_reflection(state, participant.participant_id)
        if not mvp_id:
            return None, None
        if not reason:
            reason = self._fallback_postgame_mvp_reason(state, mvp_id)
        if mvp_id:
            content = self._t(
                session.question,
                f"赛后感受：{reflection}\n我投 {self._player_name(state, mvp_id)} 是本局玩得最好的人。{reason}",
                f"Post-game feeling: {reflection}\nMy MVP vote goes to {self._player_name(state, mvp_id)}. {reason}",
            ).strip()
            vote = {
                "speaker": participant.name,
                "voter_id": participant.participant_id,
                "mvp_player_id": mvp_id,
                "reason": reason,
            }
        else:
            content = self._t(
                session.question,
                f"赛后感受：{reflection}",
                f"Post-game feeling: {reflection}",
            ).strip()
            vote = None
        turn = self._append_participant_turn(
            session=session,
            resolved=resolved,
            cycle_number=cycle_number,
            phase="post_game_reflection",
            content=content,
            prompt_source="engine:werewolf-postgame",
            prompt_used=system_prompt,
        )
        return turn, vote

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
            phase = f" | {turn.phase}" if turn.phase else ""
            lines.append(
                f"Turn {turn.turn_number} | Cycle {turn.cycle_number}{phase} | {turn.speaker} | {label} | {turn.model}\n{turn.content}"
            )
        return "\n\n".join(lines).strip()

    def _build_recent_transcript(self, turns: list[SessionTurn], limit: int = 8) -> str:
        completed = [turn for turn in turns if turn.status == TurnStatus.COMPLETED and turn.content]
        return self._build_transcript(completed[-limit:])

    def _build_werewolf_role_index(self, resolved_participants: list[ResolvedParticipant]) -> dict[str, list[ResolvedParticipant]]:
        index: dict[str, list[ResolvedParticipant]] = {}
        for resolved in resolved_participants:
            role = self._werewolf_role_for_participant(resolved.participant)
            index.setdefault(role, []).append(resolved)
        return index

    def _initialize_werewolf_state(self, resolved_participants: list[ResolvedParticipant]) -> dict[str, Any]:
        roles: dict[str, dict[str, str]] = {}
        player_order: list[str] = []
        for resolved in resolved_participants:
            participant = resolved.participant
            role = self._werewolf_role_for_participant(participant)
            roles[participant.participant_id] = {
                "name": participant.name,
                "role": role,
                "team": self._werewolf_team_for_role(role),
            }
            if role not in WEREWOLF_SYSTEM_ROLES:
                player_order.append(participant.participant_id)
        return {
            "mode": "werewolf",
            "phase": "setup",
            "round_number": 0,
            "player_order": player_order,
            "alive_ids": list(player_order),
            "dead_ids": [],
            "roles": roles,
            "seer_results": [],
            "witch_state": {
                "antidote_available": True,
                "poison_available": True,
            },
            "hunter_shot_used": False,
            "night_history": [],
            "night_private_history": [],
            "day_history": [],
            "public_reveals": {},
            "identity_confirmations": [],
            "postgame_votes": [],
            "postgame_mvp_summary": "",
            "last_night_summary": "Game not started yet.",
            "latest_night_private": "No night actions have been resolved yet.",
            "last_vote_summary": "No vote has happened yet.",
            "winner": None,
        }

    def _sync_werewolf_public_state(self, session: SessionRecord) -> None:
        state = session.engine_state
        roles = state.get("roles", {})
        alive_ids = state.get("alive_ids", [])
        dead_ids = state.get("dead_ids", [])
        public_reveals = state.get("public_reveals", {})
        session.game_state = {
            "mode": "werewolf",
            "phase": state.get("phase", "setup"),
            "round_number": state.get("round_number", 0),
            "alive_players": [
                {"participant_id": player_id, "name": roles[player_id]["name"]}
                for player_id in alive_ids
                if player_id in roles
            ],
            "dead_players": [
                {
                    "participant_id": player_id,
                    "name": roles[player_id]["name"],
                    "revealed_role": public_reveals.get(player_id),
                }
                for player_id in dead_ids
                if player_id in roles
            ],
            "last_night_summary": state.get("last_night_summary", ""),
            "last_vote_summary": state.get("last_vote_summary", ""),
            "winner": state.get("winner"),
            "moderator_view": {
                "role_map": [
                    {
                        "participant_id": player_id,
                        "name": details["name"],
                        "role": self._role_display_name(details["role"], state.get("question", session.question)),
                        "team": details["team"],
                        "status": "alive" if player_id in alive_ids else "dead",
                    }
                    for player_id, details in roles.items()
                    if details["role"] not in WEREWOLF_SYSTEM_ROLES
                ],
                "latest_night_private": state.get("latest_night_private", ""),
                "night_history": state.get("night_private_history", []),
                "identity_confirmations": state.get("identity_confirmations", []),
            },
        }

    def _apply_deaths(self, state: dict[str, Any], player_ids: Any) -> None:
        for player_id in player_ids:
            if not player_id or player_id not in state["roles"]:
                continue
            if player_id in state["alive_ids"]:
                state["alive_ids"].remove(player_id)
            if player_id not in state["dead_ids"]:
                state["dead_ids"].append(player_id)

    def _reveal_all_werewolf_roles(self, state: dict[str, Any]) -> None:
        question = state.get("question", "")
        for player_id, details in state["roles"].items():
            if details["role"] in WEREWOLF_SYSTEM_ROLES:
                continue
            if player_id in state["dead_ids"]:
                state["public_reveals"][player_id] = self._role_display_name(details["role"], question)

    def _determine_werewolf_winner(self, state: dict[str, Any]) -> str | None:
        alive_ids = state["alive_ids"]
        wolf_alive = [player_id for player_id in alive_ids if state["roles"][player_id]["team"] == "wolves"]
        village_alive = [player_id for player_id in alive_ids if state["roles"][player_id]["team"] == "village"]
        if not wolf_alive:
            return "Village"
        if len(wolf_alive) >= len(village_alive):
            return "Werewolves"
        return None

    def _winner_from_state(self, state: dict[str, Any]) -> str | None:
        winner = state.get("winner")
        if winner:
            return winner
        winner = self._determine_werewolf_winner(state)
        state["winner"] = winner
        return winner

    def _finalize_werewolf_session(self, session: SessionRecord) -> None:
        state = session.engine_state
        winner = self._winner_from_state(state)
        state["phase"] = "game_over"
        self._reveal_all_werewolf_roles(state)
        session.status = SessionStatus.COMPLETED
        question = state.get("question", session.question)
        if winner:
            closing = self._t(question, f"获胜方：{winner}。", f"Winner: {winner}.")
        else:
            closing = self._t(question, "在设定回合上限内，没有任何一方取得决定性胜利。", "No side reached a decisive win before the configured cycle limit.")
        role_table = "\n".join(
            self._t(
                question,
                f"- {details['name']}：{self._role_display_name(details['role'], question)}（{'存活' if player_id in state['alive_ids'] else '死亡'}）",
                f"- {details['name']}: {self._role_display_name(details['role'], question)} ({'alive' if player_id in state['alive_ids'] else 'dead'})",
            )
            for player_id, details in state["roles"].items()
            if details["role"] not in WEREWOLF_SYSTEM_ROLES
        )
        night_lines = [
            self._t(
                question,
                f"- 第 {record['round_number']} 夜：{record['public_summary'].replace(chr(10), ' ')}",
                f"- Night {record['round_number']}: {record['public_summary'].replace(chr(10), ' ')}",
            )
            for record in state.get("night_history", [])
        ]
        day_lines = [
            self._t(
                question,
                f"- 第 {record['round_number']} 天：{record['public_summary'].replace(chr(10), ' ')}",
                f"- Day {record['round_number']}: {record['public_summary'].replace(chr(10), ' ')}",
            )
            for record in state.get("day_history", [])
        ]
        summary_parts = [
            self._t(question, "狼人杀对局总结", "Werewolf game summary"),
            closing,
            "",
            self._t(question, "身份揭示：", "Role reveal:"),
            role_table or self._t(question, "- 没有记录到玩家身份。", "- No player roles recorded."),
        ]
        if night_lines:
            summary_parts.extend(["", self._t(question, "夜晚记录：", "Night log:")] + night_lines)
        if day_lines:
            summary_parts.extend(["", self._t(question, "白天记录：", "Day log:")] + day_lines)
        if state.get("postgame_mvp_summary"):
            summary_parts.extend(["", self._t(question, "赛后投票：", "Post-game voting:"), state["postgame_mvp_summary"]])
        session.final_answer = "\n".join(summary_parts).strip()
        session.unresolved_disagreements = []
        session.open_questions = [] if winner else [self._t(question, "游戏达到回合上限后仍未完全分出胜负。", "The game hit the cycle limit before one side fully closed the game.")]
        self._sync_werewolf_public_state(session)

    def _render_werewolf_verdict(self, state: dict[str, Any]) -> str:
        question = state.get("question", "")
        winner = state.get("winner")
        if winner == "Village":
            return self._t(
                question,
                "游戏结束。\n全部狼人已经出局。\n好人阵营获胜。",
                "Game over.\nBoth werewolves have been eliminated.\nThe Village wins.",
            )
        if winner == "Werewolves":
            return self._t(
                question,
                "游戏结束。\n狼人阵营人数已经追平剩余好人。\n狼人阵营获胜。",
                "Game over.\nThe werewolves have reached parity with the remaining village.\nThe Werewolves win.",
            )
        return self._t(
            question,
            "游戏结束。\n已达到设定的回合上限。\n没有任何一方取得决定性胜利。",
            "Game over.\nThe configured round limit has been reached.\nNo side secured a decisive win.",
        )

    def _living_werewolf_players(
        self,
        state: dict[str, Any],
        participant_index: dict[str, ResolvedParticipant],
    ) -> list[ResolvedParticipant]:
        living: list[ResolvedParticipant] = []
        for player_id in state["player_order"]:
            if player_id not in state["alive_ids"]:
                continue
            resolved = participant_index.get(player_id)
            if resolved is not None:
                living.append(resolved)
        return living

    def _render_werewolf_discussion_opening(self, state: dict[str, Any], cycle_number: int) -> str:
        alive_names = ", ".join(self._player_name(state, player_id) for player_id in state["alive_ids"]) or "none"
        question = state.get("question", "")
        return self._t(
            question,
            f"第 {cycle_number} 天讨论开始。\n当前存活玩家：{alive_names}。\n请依次发言，回应昨夜结果，并准备进入正式投票。",
            f"Day {cycle_number} discussion opens.\nLiving players: {alive_names}.\nSpeak one by one, react to the night result, and prepare for a formal vote.",
        )

    def _render_werewolf_night_summary(
        self,
        *,
        state: dict[str, Any],
        cycle_number: int,
        dead_player_ids: list[str],
        saved_target_id: str | None,
        hunter_chain: list[dict[str, str]],
    ) -> str:
        question = state.get("question", "")
        lines = [self._t(question, f"第 {cycle_number} 夜结束。", f"Night {cycle_number} has ended.")]
        if dead_player_ids:
            deaths = ", ".join(self._player_name(state, player_id) for player_id in dead_player_ids)
            lines.append(self._t(question, f"夜间死亡：{deaths}。", f"Overnight deaths: {deaths}."))
        else:
            lines.append(self._t(question, "这是一个平安夜。没有玩家死亡。", "It was a peaceful night. No player died."))
        if saved_target_id and saved_target_id not in dead_player_ids:
            lines.append(self._t(question, "黑夜中有人被成功救下，但桌面并不知道被救的是谁。", "A successful rescue happened in the dark, but the table does not know who was saved."))
        for item in hunter_chain:
            lines.append(self._t(question, f"{item['hunter_name']} 在临死前带走了 {item['target_name']}。", f"As a final act, {item['hunter_name']} shot {item['target_name']}."))
        lines.append(
            self._t(
                question,
                f"当前存活玩家：{', '.join(self._player_name(state, player_id) for player_id in state['alive_ids']) or '无'}。",
                f"Remaining living players: {', '.join(self._player_name(state, player_id) for player_id in state['alive_ids']) or 'none'}.",
            )
        )
        return "\n".join(lines)

    def _render_werewolf_private_night_summary(self, state: dict[str, Any], night_record: dict[str, Any]) -> str:
        question = state.get("question", "")
        lines = [self._t(question, f"上帝视角：第 {night_record['round_number']} 夜", f"Moderator Night {night_record['round_number']}")]
        role_map = "，".join(
            self._t(
                question,
                f"{details['name']}是{self._role_display_name(details['role'], question)}",
                f"{details['name']} is {self._role_display_name(details['role'], question)}",
            )
            for details in state["roles"].values()
            if details["role"] not in WEREWOLF_SYSTEM_ROLES
        )
        lines.append(self._t(question, f"旁白已知全部身份：{role_map}。", f"The Narrator knows every identity: {role_map}."))
        private_dialogue = night_record.get("private_dialogue", [])
        if private_dialogue:
            lines.append(self._t(question, "夜间私密交流：", "Private night exchanges:"))
            for item in private_dialogue:
                lines.append(f"{item['speaker']}：{item['content']}")

        wolf_votes = night_record.get("wolf_votes", [])
        if wolf_votes and not private_dialogue:
            for item in wolf_votes:
                lines.append(
                    self._t(
                        question,
                        f"旁白：{item['wolf']}，今晚你要袭击谁？",
                        f"Narrator: {item['wolf']}, who do you attack tonight?",
                    )
                )
                lines.append(f"{item['wolf']}：{item['reply'] or self._night_reply_fallback('werewolf', state, item['target'])}")
        elif not wolf_votes:
            lines.append(self._t(question, "旁白：今夜没有记录到有效的狼人行动。", "Narrator: no valid werewolf action was recorded tonight."))

        seer_result = night_record.get("seer_result")
        if seer_result and seer_result.get("target_id"):
            target_id = seer_result["target_id"]
            alignment = self._t(
                question,
                "狼人" if state["roles"][target_id]["team"] == "wolves" else "非狼人",
                "Werewolf" if state["roles"][target_id]["team"] == "wolves" else "Not Werewolf",
            )
            lines.append(self._t(question, f"旁白：查验结果，{self._player_name(state, target_id)} 是{alignment}。", f"Narrator: inspection result, {self._player_name(state, target_id)} is {alignment}."))
        else:
            lines.append(self._t(question, "旁白：今夜没有进行预言家查验。", "Narrator: no Seer inspection happened tonight."))

        saved_target_id = night_record.get("saved_target_id")
        poison_target_id = night_record.get("poison_target_id")
        if saved_target_id:
            lines.append(self._t(question, f"旁白：记录，女巫救下了 {self._player_name(state, saved_target_id)}。", f"Narrator: recorded, the Witch saved {self._player_name(state, saved_target_id)}."))
        else:
            lines.append(self._t(question, "旁白：记录，女巫今夜没有救人。", "Narrator: recorded, the Witch did not save anyone tonight."))
        if poison_target_id:
            lines.append(self._t(question, f"旁白：记录，女巫毒死了 {self._player_name(state, poison_target_id)}。", f"Narrator: recorded, the Witch poisoned {self._player_name(state, poison_target_id)}."))
        else:
            lines.append(self._t(question, "旁白：记录，女巫今夜没有下毒。", "Narrator: recorded, the Witch did not poison anyone tonight."))

        for item in night_record.get("hunter_chain", []):
            lines.append(self._t(question, f"旁白：记录，{item['hunter_name']} 带走了 {item['target_name']}。", f"Narrator: recorded, {item['hunter_name']} shot {item['target_name']}."))

        dead_names = [
            self._player_name(state, player_id)
            for player_id in night_record.get("dead_player_ids", [])
        ]
        lines.append(self._t(question, f"旁白：天亮了，实际死亡为 {', '.join(dead_names) or '无'}。", f"Narrator: dawn breaks, deaths applied are {', '.join(dead_names) or 'none'}."))
        return "\n".join(lines)

    def _append_private_dialogue(self, private_dialogue: list[dict[str, str]], speaker: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        private_dialogue.append({"speaker": speaker, "content": text})

    def _night_narrator_question(
        self,
        *,
        state: dict[str, Any],
        role: str,
        participant_name: str,
        wolf_target_id: str | None = None,
    ) -> str:
        question = state.get("question", "")
        if role == "werewolf":
            return self._t(
                question,
                f"【私密】{participant_name}，你的身份是狼人。今晚你要袭击谁？只用一句很短的话回答。",
                f"[Private] {participant_name}, you are a Werewolf. Who do you attack tonight? Reply in one very short line.",
            )
        if role == "seer":
            return self._t(
                question,
                f"【私密】{participant_name}，你的身份是预言家。今晚你要查验谁？只用一句很短的话回答。",
                f"[Private] {participant_name}, you are the Seer. Who do you inspect tonight? Reply in one very short line.",
            )
        if role == "witch":
            victim_name = self._player_name(state, wolf_target_id)
            return self._t(
                question,
                f"【私密】{participant_name}，你的身份是女巫。今晚死的人是 {victim_name}。你要救人还是下毒？只用一句很短的话回答。",
                f"[Private] {participant_name}, you are the Witch. Tonight's victim is {victim_name}. Will you save or poison? Reply in one very short line.",
            )
        if role == "hunter":
            return self._t(
                question,
                f"【私密】{participant_name}，你的身份是猎人。你现在可以带走一人。你要带走谁？只用一句很短的话回答。",
                f"[Private] {participant_name}, you are the Hunter. You may shoot one player now. Who is your target? Reply in one very short line.",
            )
        return self._t(question, "【私密】请行动。只用一句很短的话回答。", "[Private] Act now. Reply in one very short line.")

    def _identity_confirmation_question(self, question: str, participant_name: str) -> str:
        return self._t(
            question,
            f"【私密】{participant_name}，请先告诉我你在这局游戏中的隐藏身份。只用一句很短的话回答。",
            f"[Private] {participant_name}, first tell me your hidden role in this game. Reply in one very short line.",
        )

    def _identity_confirmation_reply(self, question: str, participant_name: str, role: str) -> str:
        return self._t(
            question,
            f"我是{self._role_display_name(role, question)}。",
            f"I am the {self._role_display_name(role, question)}.",
        )

    def _canonicalize_werewolf_role(self, text: str) -> str | None:
        value = (text or "").strip().lower()
        if not value:
            return None
        normalized = value.replace("-", " ").replace("_", " ")
        role_patterns = [
            ("werewolf", ["werewolf", "wolf", "狼人", "狼"]),
            ("seer", ["seer", "预言家"]),
            ("witch", ["witch", "女巫"]),
            ("hunter", ["hunter", "猎人"]),
            ("villager", ["villager", "villager role", "civilian", "平民", "村民", "好人"]),
            ("narrator", ["narrator", "moderator", "上帝", "法官", "旁白"]),
            ("vote_counter", ["vote counter", "counter", "计票员"]),
        ]
        for role, patterns in role_patterns:
            if any(pattern in normalized for pattern in patterns):
                return role
        return None

    def _night_reply_fallback(self, role: str, state: dict[str, Any], target: Any) -> str:
        question = state.get("question", "")
        if role == "werewolf":
            return self._t(question, f"刀 {self._player_name(state, self._coerce_single_target(target))}。", f"Attack {self._player_name(state, self._coerce_single_target(target))}.")
        if role == "seer":
            return self._t(question, f"查 {self._player_name(state, self._coerce_single_target(target))}。", f"Check {self._player_name(state, self._coerce_single_target(target))}.")
        if role == "witch":
            if isinstance(target, list):
                save_target = self._coerce_single_target(target[0] if target else None)
                poison_target = self._coerce_single_target(target[1] if len(target) > 1 else None)
                if save_target and poison_target:
                    return self._t(question, f"救 {self._player_name(state, save_target)}，毒 {self._player_name(state, poison_target)}。", f"Save {self._player_name(state, save_target)}, poison {self._player_name(state, poison_target)}.")
                if save_target:
                    return self._t(question, f"救 {self._player_name(state, save_target)}，不毒。", f"Save {self._player_name(state, save_target)}, no poison.")
                if poison_target:
                    return self._t(question, f"不救，毒 {self._player_name(state, poison_target)}。", f"No save, poison {self._player_name(state, poison_target)}.")
            return self._t(question, "不救，不毒。", "No save, no poison.")
        if role == "hunter":
            return self._t(question, f"带走 {self._player_name(state, self._coerce_single_target(target))}。", f"Shoot {self._player_name(state, self._coerce_single_target(target))}.")
        return self._t(question, "收到。", "Understood.")

    def _coerce_single_target(self, target: Any) -> str | None:
        if isinstance(target, list):
            return str(target[0]).strip() if target and target[0] is not None else None
        if target is None:
            return None
        value = str(target).strip()
        return value or None

    def _render_werewolf_vote_summary(
        self,
        *,
        state: dict[str, Any],
        cycle_number: int,
        votes: dict[str, str],
        tally: Counter[str],
        eliminated_id: str | None,
        tie_names: list[str],
        hunter_chain: list[dict[str, str]],
    ) -> str:
        question = state.get("question", "")
        lines = [self._t(question, f"第 {cycle_number} 天投票结束。", f"Day {cycle_number} voting is complete.")]
        if votes:
            for target_id in [player_id for player_id in state["player_order"] if player_id in tally]:
                lines.append(self._t(question, f"- {self._player_name(state, target_id)}：{tally[target_id]} 票", f"- {self._player_name(state, target_id)}: {tally[target_id]} vote(s)"))
        else:
            lines.append(self._t(question, "没有记录到有效投票。", "No valid votes were recorded."))
        if eliminated_id:
            lines.append(self._t(question, f"被投票淘汰：{self._player_name(state, eliminated_id)}。", f"Eliminated by vote: {self._player_name(state, eliminated_id)}."))
        elif tie_names:
            lines.append(self._t(question, f"{', '.join(tie_names)} 平票。无人出局。", f"The vote tied between {', '.join(tie_names)}. No one is eliminated."))
        else:
            lines.append(self._t(question, "今天无人出局。", "No one is eliminated today."))
        for item in hunter_chain:
            lines.append(self._t(question, f"{item['hunter_name']} 临走前带走了 {item['target_name']}。", f"Before leaving, {item['hunter_name']} shot {item['target_name']}."))
        return "\n".join(lines)

    def _render_werewolf_postgame_mvp_summary(self, state: dict[str, Any], votes: list[dict[str, str]]) -> str:
        question = state.get("question", "")
        if not votes:
            return self._t(question, "没有记录到赛后 MVP 投票。", "No post-game MVP votes were recorded.")
        tally = Counter(vote["mvp_player_id"] for vote in votes if vote.get("mvp_player_id"))
        lines = [self._t(question, "赛后最佳表现投票结果：", "Post-game MVP vote results:")]
        for player_id in [player_id for player_id in state["player_order"] if player_id in tally]:
            lines.append(self._t(question, f"- {self._player_name(state, player_id)}：{tally[player_id]} 票", f"- {self._player_name(state, player_id)}: {tally[player_id]} vote(s)"))
        top_votes = max(tally.values()) if tally else 0
        winners = [
            self._player_name(state, player_id)
            for player_id in state["player_order"]
            if tally.get(player_id, 0) == top_votes
        ]
        if len(winners) == 1:
            lines.append(self._t(question, f"本局公认发挥最好的是：{winners[0]}。", f"Consensus best player: {winners[0]}."))
        elif winners:
            lines.append(self._t(question, f"本局最佳表现平票：{', '.join(winners)}。", f"Tied best players: {', '.join(winners)}."))
        return "\n".join(lines)

    def _render_werewolf_postgame_recap(self, state: dict[str, Any]) -> str:
        question = state.get("question", "")
        parts = [
            self._t(question, f"胜者：{state.get('winner') or '未分胜负'}", f"Winner: {state.get('winner') or 'No decisive winner'}"),
        ]
        last_night = state.get("last_night_summary")
        last_vote = state.get("last_vote_summary")
        if last_night:
            parts.append(self._t(question, f"最后夜晚：{last_night.replace(chr(10), ' ')}", f"Last night: {last_night.replace(chr(10), ' ')}"))
        if last_vote:
            parts.append(self._t(question, f"最后投票：{last_vote.replace(chr(10), ' ')}", f"Last vote: {last_vote.replace(chr(10), ' ')}"))
        dead_names = ", ".join(self._player_name(state, player_id) for player_id in state.get("dead_ids", [])) or self._t(question, "无", "none")
        parts.append(self._t(question, f"死亡玩家：{dead_names}", f"Dead players: {dead_names}"))
        return "\n".join(parts)

    def _render_werewolf_postgame_speech_highlights(self, session: SessionRecord, state: dict[str, Any]) -> str:
        highlights: list[str] = []
        public_phases = {"day_discussion", "day_vote", "day_resolution", "night_result"}
        turns_by_player: dict[str, list[SessionTurn]] = {}
        for turn in session.turns:
            if turn.status != TurnStatus.COMPLETED or not turn.content:
                continue
            if turn.phase not in public_phases:
                continue
            if turn.participant_id not in state.get("player_order", []):
                continue
            turns_by_player.setdefault(turn.participant_id, []).append(turn)

        for player_id in state.get("player_order", []):
            turns = turns_by_player.get(player_id, [])
            if not turns:
                continue
            player_name = self._player_name(state, player_id)
            for turn in turns[-2:]:
                snippet = " ".join((turn.content or "").split())
                if len(snippet) > 140:
                    snippet = snippet[:137].rstrip() + "..."
                highlights.append(f"- {player_name} | {turn.phase}: {snippet}")
        return "\n".join(highlights) if highlights else self._t(
            state.get("question", ""),
            "没有可用的公开发言摘要。",
            "No usable public speech highlights.",
        )

    def _fallback_postgame_mvp_id(self, state: dict[str, Any], voter_id: str) -> str | None:
        dead_ids = list(state.get("dead_ids", []))
        for player_id in dead_ids:
            if player_id != voter_id:
                return player_id
        winner = state.get("winner")
        if winner == "Werewolves":
            for player_id in state.get("player_order", []):
                if player_id != voter_id and state["roles"][player_id]["team"] == "wolves":
                    return player_id
        if winner == "Village":
            for player_id in state.get("player_order", []):
                if player_id != voter_id and state["roles"][player_id]["team"] == "village":
                    return player_id
        for player_id in state.get("player_order", []):
            if player_id != voter_id:
                return player_id
        return None

    def _fallback_postgame_reflection(self, state: dict[str, Any], participant_id: str) -> str:
        question = state.get("question", "")
        winner = state.get("winner")
        role = state["roles"][participant_id]["role"]
        team = state["roles"][participant_id]["team"]
        won = (winner == "Werewolves" and team == "wolves") or (winner == "Village" and team == "village")
        if role == "narrator":
            return self._t(question, "这局结束得很快，关键票型直接决定了胜负。", "The game ended quickly, and the decisive vote locked the result.")
        if role == "vote_counter":
            return self._t(question, "这局的票型和临场选择都很关键。", "The vote pattern and clutch decisions defined this game.")
        if won:
            return self._t(question, "赢下来了，关键回合的选择很重要。", "We got the win, and the key turns mattered.")
        return self._t(question, "这局输了，但关键转折很值得复盘。", "We lost this one, but the turning points are worth reviewing.")

    def _fallback_postgame_mvp_reason(self, state: dict[str, Any], mvp_id: str | None) -> str:
        question = state.get("question", "")
        if not mvp_id:
            return self._t(question, "他的发挥最有影响力。", "Their play had the biggest impact.")
        return self._t(
            question,
            f"{self._player_name(state, mvp_id)} 的公开发言和对局势的影响最强。",
            f"{self._player_name(state, mvp_id)} had the strongest public speaking impact on the game.",
        )

    def _render_werewolf_public_state_text(self, state: dict[str, Any]) -> str:
        question = state.get("question", "")
        alive_names = ", ".join(self._player_name(state, player_id) for player_id in state["alive_ids"]) or "none"
        dead_names = ", ".join(
            f"{self._player_name(state, player_id)} ({state['public_reveals'].get(player_id, 'role hidden')})"
            for player_id in state["dead_ids"]
        ) or "none"
        return "\n".join(
            [
                self._t(question, f"阶段：{state.get('phase', 'setup')}", f"Phase: {state.get('phase', 'setup')}"),
                self._t(question, f"回合：{state.get('round_number', 0)}", f"Round: {state.get('round_number', 0)}"),
                self._t(question, f"存活：{alive_names}", f"Alive: {alive_names}"),
                self._t(question, f"死亡：{dead_names}", f"Dead: {dead_names}"),
                self._t(question, f"最新夜晚结果：{state.get('last_night_summary', '无')}", f"Latest night result: {state.get('last_night_summary', 'none')}"),
                self._t(question, f"最新投票结果：{state.get('last_vote_summary', '无')}", f"Latest vote result: {state.get('last_vote_summary', 'none')}"),
            ]
        )

    def _render_werewolf_private_briefing(self, state: dict[str, Any], participant_id: str) -> str:
        question = state.get("question", "")
        role = state["roles"][participant_id]["role"]
        if role == "werewolf":
            partner_names = [
                self._player_name(state, player_id)
                for player_id in state["alive_ids"]
                if player_id != participant_id and state["roles"][player_id]["role"] == "werewolf"
            ]
            return self._t(question, f"你是狼人。\n存活狼队友：{', '.join(partner_names) or '无'}。\n保护狼队、转移怀疑并活过白天投票。", f"You are a Werewolf.\nLiving wolf partners: {', '.join(partner_names) or 'none'}.\nProtect the wolf team, misdirect suspicion, and survive the vote.")
        if role == "seer":
            return self._t(question, f"你是预言家。\n目前私有查验结果：\n{self._render_seer_results(state)}\n请谨慎使用信息，不必立刻全部公开。", f"You are the Seer.\nPrivate checks so far:\n{self._render_seer_results(state)}\nUse your information carefully. You do not need to reveal everything immediately.")
        if role == "witch":
            witch_state = state["witch_state"]
            return self._t(question, f"你是女巫。\n解药可用：{'是' if witch_state['antidote_available'] else '否'}。\n毒药可用：{'是' if witch_state['poison_available'] else '否'}。\n平衡生存、节奏和信息收益。", f"You are the Witch.\nAntidote available: {'yes' if witch_state['antidote_available'] else 'no'}.\nPoison available: {'yes' if witch_state['poison_available'] else 'no'}.\nBalance survival, tempo, and information advantage.")
        if role == "hunter":
            return self._t(question, "你是猎人。如果你死亡，引擎会让你决定是否开枪带人。", "You are the Hunter. If you die, the engine will ask you for a final shot.")
        if role == "villager":
            return self._t(question, "你是平民。你没有夜晚技能，请根据局势分析并认真投票。", "You are a Villager. You have no night power. Read the board and vote well.")
        return self._t(question, "你是本模式的系统主持角色。", "You are the moderator-side system role for this mode.")

    def _render_werewolf_personal_memory_window(
        self,
        session: SessionRecord,
        state: dict[str, Any],
        participant_id: str,
    ) -> str:
        question = state.get("question", "")
        role = state["roles"][participant_id]["role"]
        parts = [
            self._t(question, "你自己的近期公开发言：", "Your recent public statements:"),
            self._render_recent_public_turn_snippets(session, [participant_id], limit_per_player=2),
        ]
        if role == "werewolf":
            partner_ids = [
                player_id
                for player_id in state["alive_ids"]
                if player_id != participant_id and state["roles"][player_id]["role"] == "werewolf"
            ]
            parts.extend(
                [
                    self._t(question, "狼队友的近期公开发言：", "Recent public statements from your wolf partners:"),
                    self._render_recent_public_turn_snippets(session, partner_ids, limit_per_player=2),
                ]
            )
        elif role == "seer":
            parts.extend(
                [
                    self._t(question, "你的私有查验记忆：", "Your private inspection memory:"),
                    self._render_seer_results(state),
                ]
            )
        elif role == "witch":
            parts.extend(
                [
                    self._t(question, "你的药剂与夜晚行动记忆：", "Your potion and night-action memory:"),
                    self._render_witch_action_memory(state),
                ]
            )
        elif role == "hunter":
            parts.append(
                self._t(
                    question,
                    "你的个人提醒：若你死亡，引擎会立刻私密询问你是否开枪带走一人。",
                    "Your reminder: if you die, the engine will immediately ask you privately whether to fire your final shot.",
                )
            )
        elif role == "villager":
            parts.append(
                self._t(
                    question,
                    "你的个人提醒：你没有夜晚技能，请重点根据白天发言、票型和死亡结果建立判断。",
                    "Your reminder: you have no night power, so build your read from speeches, voting patterns, and deaths.",
                )
            )
        return "\n".join(part for part in parts if part).strip()

    def _render_seer_results(self, state: dict[str, Any]) -> str:
        question = state.get("question", "")
        rows = []
        for result in state.get("seer_results", []):
            rows.append(self._t(question, f"- 第 {result['round_number']} 夜：{result['target_name']} -> {'狼人' if result['alignment'] == 'werewolf' else '非狼人'}", f"- Night {result['round_number']}: {result['target_name']} -> {result['alignment']}"))
        return "\n".join(rows) if rows else self._t(question, "还没有私有查验结果。", "No private checks yet.")

    def _render_witch_action_memory(self, state: dict[str, Any]) -> str:
        question = state.get("question", "")
        rows = []
        for record in state.get("night_history", []):
            actions: list[str] = []
            if record.get("saved_target_id"):
                actions.append(
                    self._t(
                        question,
                        f"救下了 {self._player_name(state, record['saved_target_id'])}",
                        f"saved {self._player_name(state, record['saved_target_id'])}",
                    )
                )
            if record.get("poison_target_id"):
                actions.append(
                    self._t(
                        question,
                        f"毒死了 {self._player_name(state, record['poison_target_id'])}",
                        f"poisoned {self._player_name(state, record['poison_target_id'])}",
                    )
                )
            if not actions:
                actions.append(self._t(question, "没有使用药剂", "used no potion"))
            rows.append(
                self._t(
                    question,
                    f"- 第 {record['round_number']} 夜：{'，'.join(actions)}",
                    f"- Night {record['round_number']}: {', '.join(actions)}",
                )
            )
        if not rows:
            return self._t(question, "你还没有留下药剂行动记录。", "You have not recorded any potion actions yet.")
        return "\n".join(rows)

    def _render_recent_public_turn_snippets(
        self,
        session: SessionRecord,
        player_ids: list[str],
        *,
        limit_per_player: int = 2,
    ) -> str:
        if not player_ids:
            return self._t(session.question, "没有可用记录。", "No records available.")
        public_phases = {"day_discussion", "day_vote", "day_resolution", "night_result"}
        snippets: list[str] = []
        for player_id in player_ids:
            player_turns = [
                turn
                for turn in session.turns
                if turn.status == TurnStatus.COMPLETED
                and turn.content
                and turn.participant_id == player_id
                and turn.phase in public_phases
            ]
            for turn in player_turns[-limit_per_player:]:
                snippet = " ".join((turn.content or "").split())
                if len(snippet) > 140:
                    snippet = snippet[:137].rstrip() + "..."
                snippets.append(f"- {turn.speaker} | {turn.phase}: {snippet}")
        return "\n".join(snippets) if snippets else self._t(session.question, "没有可用记录。", "No records available.")

    def _render_player_choices(self, state: dict[str, Any], player_ids: list[str]) -> str:
        return "\n".join(f"- {player_id}: {self._player_name(state, player_id)}" for player_id in player_ids if player_id)

    def _choose_weighted_target(
        self,
        *,
        proposed_ids: list[str],
        eligible_ids: list[str],
        ordered_ids: list[str],
    ) -> str | None:
        counts = Counter(player_id for player_id in proposed_ids if player_id in eligible_ids)
        if not counts:
            return eligible_ids[0] if eligible_ids else None
        top = max(counts.values())
        for player_id in ordered_ids:
            if counts.get(player_id, 0) == top:
                return player_id
        return eligible_ids[0] if eligible_ids else None

    def _fallback_vote_target(self, state: dict[str, Any], voter_id: str) -> str | None:
        for player_id in state["player_order"]:
            if player_id in state["alive_ids"] and player_id != voter_id:
                return player_id
        return None

    def _coerce_player_id(self, value: Any, eligible_ids: list[str], state: dict[str, Any]) -> str | None:
        if not eligible_ids:
            return None
        candidates: list[str] = []
        if isinstance(value, list):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        elif value is not None:
            candidates = [str(value).strip()]
        lowered: dict[str, str] = {}
        for player_id in eligible_ids:
            lowered[player_id.lower()] = player_id
            lowered[self._player_name(state, player_id).lower()] = player_id
        for candidate in candidates:
            key = candidate.lower()
            if key in lowered:
                return lowered[key]
            for lowered_key, player_id in lowered.items():
                if lowered_key in key or key in lowered_key:
                    return player_id
        return None

    def _player_name(self, state: dict[str, Any], player_id: str | None) -> str:
        if not player_id:
            return "Unknown player"
        return state["roles"].get(player_id, {}).get("name", player_id)

    def _werewolf_role_for_participant(self, participant: DiscussionParticipant) -> str:
        prompt_ref = (participant.prompt_source or participant.prompt).lower()
        name = participant.name.lower()
        if "werewolf_narrator" in prompt_ref or name == "narrator":
            return "narrator"
        if "werewolf_vote_counter" in prompt_ref or "vote counter" in name:
            return "vote_counter"
        if "werewolf_player_wolf" in prompt_ref:
            return "werewolf"
        if "werewolf_seer" in prompt_ref:
            return "seer"
        if "werewolf_player_witch" in prompt_ref:
            return "witch"
        if "werewolf_hunter" in prompt_ref:
            return "hunter"
        if "werewolf_player_villager" in prompt_ref:
            return "villager"
        return "villager"

    def _werewolf_team_for_role(self, role: str) -> str:
        if role == "werewolf":
            return "wolves"
        if role in WEREWOLF_SYSTEM_ROLES:
            return "system"
        return "village"

    def _role_display_name(self, role: str, question: str = "") -> str:
        if self._preferred_language(question) == LANGUAGE_ZH:
            labels = {
                "narrator": "旁白",
                "vote_counter": "计票员",
                "werewolf": "狼人",
                "seer": "预言家",
                "witch": "女巫",
                "hunter": "猎人",
                "villager": "平民",
            }
            return labels.get(role, role.replace("_", " ").title())
        labels = {
            "narrator": "Narrator",
            "vote_counter": "Vote Counter",
            "werewolf": "Werewolf",
            "seer": "Seer",
            "witch": "Witch",
            "hunter": "Hunter",
            "villager": "Villager",
        }
        return labels.get(role, role.replace("_", " ").title())

    def _is_werewolf_discussion(self, discussion_id: str | None) -> bool:
        return (discussion_id or "") == WEREWOLF_DISCUSSION_ID

    def _first_or_none(self, items: list[ResolvedParticipant]) -> ResolvedParticipant | None:
        return items[0] if items else None

    def _preferred_language(self, text: str) -> str:
        source = text or ""
        normalized = source.lower()
        if any(token in normalized for token in ["中文", "汉语", "汉文", "chinese", "mandarin"]):
            return LANGUAGE_ZH
        if any("\u4e00" <= char <= "\u9fff" for char in source):
            return LANGUAGE_ZH
        return LANGUAGE_EN

    def _t(self, text: str, zh: str, en: str) -> str:
        return zh if self._preferred_language(text) == LANGUAGE_ZH else en

    def _language_instruction(self, text: str, *, public: bool) -> str:
        if self._preferred_language(text) == LANGUAGE_ZH:
            if public:
                return "The original setup asks for Chinese. Write your public response in natural Chinese."
            return "The original setup asks for Chinese. Keep JSON keys in English, but write all string values in natural Chinese."
        if public:
            return "Match the language requested in the original setup question."
        return "Keep JSON keys in English and match the language requested in the original setup question for all string values."

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

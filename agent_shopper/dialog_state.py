"""Pillar II: per-session dialog state.

Held in-process in a plain dict keyed by session_id -- no database. Each
competition session is an isolated single-user interaction (per the
official spec), so there is nothing to persist beyond the process's own
lifetime, and no cross-session state is available or needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_shopper.models import DistilledProfile, SlotSet


@dataclass
class TurnRecord:
    turn: int
    user_message: str
    ask_attribute: str | None
    pool_size: int
    track: str


@dataclass
class OverrideEvent:
    turn: int
    slot: str
    old_value: object
    new_value: object


@dataclass
class ShownRecord:
    turn: int
    parent_asin: str


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    distilled_profile: DistilledProfile

    turn: int = 0
    track: str = "browsing"
    slots: SlotSet = field(default_factory=SlotSet)

    last_pool_size: int = 0
    turns_since_progress: int = 0
    clarified_attributes: set[str] = field(default_factory=set)
    override_events: list[OverrideEvent] = field(default_factory=list)
    history: list[TurnRecord] = field(default_factory=list)

    # Calibrated P(this turn is an intent reversal) -- see
    # agent_shopper.override_model. Recomputed every turn in
    # dialog_policy.process_turn; 0.0 until the first turn runs.
    override_probability: float = 0.0

    # Every (turn, asin) actually returned in `recommendations` -- raw
    # material for the reranker's soft repeat-penalty (see reranker.py).
    shown_history: list[ShownRecord] = field(default_factory=list)

    llm_failure_count: int = 0
    llm_disabled: bool = False

    def has_overridden(self) -> bool:
        return bool(self.override_events)

    def record_turn(self, user_message: str, ask_attribute: str | None, pool_size: int) -> None:
        progressed = pool_size < self.last_pool_size or self.last_pool_size == 0
        self.turns_since_progress = 0 if progressed else self.turns_since_progress + 1
        self.last_pool_size = pool_size
        self.history.append(TurnRecord(
            turn=self.turn, user_message=user_message, ask_attribute=ask_attribute,
            pool_size=pool_size, track=self.track,
        ))
        if ask_attribute:
            self.clarified_attributes.add(ask_attribute)


class SessionStore:
    """In-memory session registry. One instance lives on the Agent."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

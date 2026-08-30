"""The required competition Agent interface: reset() + respond().

Thin glue: builds the catalog + BM25 + TF-IDF indices once at construction
(the only place we pay the O(catalog) startup cost), keeps an in-memory
SessionStore, and delegates every turn to dialog_policy.process_turn. All
four pillars live in the modules that function calls into -- this class
intentionally does no retrieval/ranking/state logic of its own.
"""

from __future__ import annotations

from pathlib import Path

from agent_shopper.bm25_index import BM25Index
from agent_shopper.catalog import Catalog
from agent_shopper.context import distill_profile
from agent_shopper.dense_index import DenseIndex
from agent_shopper.dialog_policy import process_turn
from agent_shopper.dialog_state import SessionState, SessionStore
from agent_shopper.tfidf_index import TfidfIndex


class Agent:
    """Agent Shopper: dual-track intent routing, multi-route retrieval,
    pluggable LLM/heuristic semantic ranking, a slot-accumulation +
    intent-override dialog state machine, and adaptive orchestration driven
    by distilled long-term/short-term context. See README.md for the full
    architecture writeup mapping each pillar to its module."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)
        self.bm25_index = BM25Index(self.catalog)
        self.tfidf_index = TfidfIndex(self.catalog)
        self.dense_index = DenseIndex(self.catalog)
        self.sessions = SessionStore()

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState(
            session_id=session_id,
            user_profile=user_profile,
            distilled_profile=distill_profile(user_profile),
        )
        self.sessions.create(state)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        return process_turn(
            self.catalog, self.bm25_index, self.tfidf_index, state, user_message, turn, top_k, self.dense_index,
        )

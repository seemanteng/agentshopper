"""Pillar III: Adaptive Orchestration.

A small set of explicit, table-driven runtime decisions -- which routes run
and whether the buying gate applies, whether to spend an LLM call on
reranking at all, and whether to spend a turn clarifying vs. just
recommending. No learning/training involved (out of scope for the
challenge); "adaptive" here means the *policy itself* reacts to session
signals (turn budget, pool size, slot coverage, stuck-ness), not that any
parameter is fit from data.
"""

from __future__ import annotations

from agent_shopper.config import (
    BUDGET_RELAX_FACTOR,
    BUYING_GATE_MIN_SLOTS,
    CLARIFY_MAX_FILLED_SLOTS,
    ENDGAME_TURNS_REMAINING,
    MIN_CLARIFY_SPLIT_SCORE,
    OVER_GENERALITY_POOL_SIZE,
    ROUTE_WEIGHTS,
    SLOT_PRIORITY,
    TIGHT_POOL_SIZE,
)
from agent_shopper.models import RoutePlan, SlotSet


def decide_routes(track: str, slots: SlotSet) -> tuple[RoutePlan, SlotSet]:
    """Picks route weights and whether the buying hard-gate applies. Does
    not itself run retrieval -- see retrieval.retrieve()."""
    weights = dict(ROUTE_WEIGHTS[track])
    gate = track == "buying" and slots.hard_filled_count() >= BUYING_GATE_MIN_SLOTS
    # When not gated (browsing track, or buying under threshold), hard-marked
    # slots still get enforced as an exact filter over the fused pool rather
    # than only ranking -- see retrieval.py. Once gated, filter_products()
    # already enforces every filled slot, so there's nothing extra to do here.
    hard_filter_slots = () if gate else tuple(slots.hard_marked)
    return RoutePlan(weights=weights, gate_to_category=gate, hard_filter_slots=hard_filter_slots), slots


def droppable_slots_by_tier(slots: SlotSet) -> tuple[list[str], list[str]]:
    """Which currently-filled slots (other than category) are eligible to
    be relaxed, split into (soft, hard_marked) tiers, each preserving
    SLOT_PRIORITY order. Pure slot-state classification -- no retrieval
    cost. Used by relax_gate's static pick below."""
    droppable = [name for name in SLOT_PRIORITY if name != "category" and getattr(slots, name)]
    soft = [name for name in droppable if name not in slots.hard_marked]
    hard = [name for name in droppable if name in slots.hard_marked]
    return soft, hard


def relax_gate(slots: SlotSet) -> tuple[SlotSet, str, bool] | None:
    """Drops the single lowest-priority filled slot to unstick an
    over-constrained buying gate that returned zero products. Prefers
    dropping a slot that was never marked as a hard constraint; only falls
    back to a hard-marked slot when nothing soft remains. Returns
    (relaxed_slots, dropped_slot_name, was_hard_marked), or None if nothing
    safe to drop (category is never dropped -- it's the one signal that
    keeps a hard-gated buying session from wandering cross-category)."""
    soft, hard = droppable_slots_by_tier(slots)
    if not soft and not hard:
        return None
    name, was_hard = (soft[0], False) if soft else (hard[0], True)
    relaxed = slots.copy()
    setattr(relaxed, name, [] if name == "feature" else None)
    return relaxed, name, was_hard


def widen_budget(slots: SlotSet) -> SlotSet | None:
    if not slots.budget:
        return None
    lo, hi = slots.budget
    lo = None if lo is None else round(lo * (1 - BUDGET_RELAX_FACTOR), 2)
    hi = None if hi is None else round(hi * (1 + BUDGET_RELAX_FACTOR), 2)
    relaxed = slots.copy()
    relaxed.budget = (lo, hi)
    return relaxed


def should_clarify(pool_size: int, filled_hard_slots: int, turns_remaining: int, best_split_score: float) -> bool:
    if turns_remaining <= ENDGAME_TURNS_REMAINING:
        return False
    if pool_size <= TIGHT_POOL_SIZE:
        return False
    if pool_size <= OVER_GENERALITY_POOL_SIZE:
        return False
    if filled_hard_slots >= CLARIFY_MAX_FILLED_SLOTS:
        return False
    return best_split_score >= MIN_CLARIFY_SPLIT_SCORE


def decide_rerank_engine(
    pool_size: int, turn: int, max_turns: int, llm_available: bool, llm_disabled: bool, do_clarify: bool = False,
) -> tuple[str, str]:
    """Returns (engine, reason) -- engine is 'heuristic' or 'llm'; reason is
    which branch decided it ('tight_pool'/'clarify_skip'/'last_turn'/
    'circuit_breaker'/'no_provider'/'eligible'), read by dialog_policy to
    populate SessionState.engine_trace for the LLM-invocation-rate
    breakdown (see scripts/run_llm_benchmark.py)."""
    if pool_size <= TIGHT_POOL_SIZE:
        # Still worth the free, zero-latency heuristic composite (attribute
        # match / rating / price fit) instead of leaving a small pool sorted
        # by raw fused-RRF order alone -- see config.TIGHT_POOL_SIZE.
        return "heuristic", "tight_pool"
    if do_clarify:
        # The turn's primary output is a clarifying question, not the
        # ranked list -- don't spend an LLM call on a ranking the shopper's
        # attention isn't going to this turn.
        return "heuristic", "clarify_skip"
    if turn >= max_turns:
        # No latency risk on the final turn -- always the fast, free path.
        return "heuristic", "last_turn"
    if llm_disabled:
        return "heuristic", "circuit_breaker"
    if not llm_available:
        return "heuristic", "no_provider"
    return "llm", "eligible"

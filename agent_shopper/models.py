"""Shared dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass(frozen=True)
class Product:
    """A normalized catalog row. Only participant-visible fields."""

    parent_asin: str
    title: str
    features: list[str]
    description: list[str]
    price: Optional[float]
    categories: list[str]
    details: dict[str, str]
    average_rating: Optional[float]
    rating_number: Optional[int]
    store: str


# Slot names, in the exact vocabulary the competition's ask_attribute enum
# uses (plus "category" which is a slot but rarely worth asking about, see
# config.SIMULATOR_DISCLOSABLE_ATTRIBUTES).
SLOT_NAMES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case",
)


@dataclass
class SlotSet:
    """Accumulated hard/soft constraints for a session.

    `budget` is a (min, max) tuple in dollars, either side may be None.
    `feature` is a list (multiple free-text features can coexist without
    conflict). Every other slot is a single value, or a list when an
    "or"/"also" conjunction has accumulated more than one acceptable value.
    """

    category: Optional[str] = None
    material: Optional[str] | list[str] = None
    color: Optional[str] | list[str] = None
    size: Optional[str] | list[str] = None
    style: Optional[str] | list[str] = None
    brand: Optional[str] | list[str] = None
    budget: Optional[tuple[Optional[float], Optional[float]]] = None
    feature: list[str] = field(default_factory=list)
    use_case: Optional[str] = None

    # Names of slots that were stated with hard-constraint language ("max",
    # "must be", "only", ...) on the turn they were filled -- see
    # dialog_policy._HARD_CONSTRAINT_RE. A frozenset (rather than a plain
    # set) so `copy()`'s `dataclasses.replace(self)` can safely share it by
    # reference across instances without an aliasing bug: it's immutable, so
    # a "copy" is just handing out the same value, exactly like every other
    # scalar field on this dataclass already does.
    hard_marked: frozenset[str] = field(default_factory=frozenset)

    def filled_slots(self) -> list[str]:
        out = []
        for name in SLOT_NAMES:
            value = getattr(self, name)
            if name == "feature":
                if value:
                    out.append(name)
            elif value is not None:
                out.append(name)
        return out

    def hard_filled_count(self) -> int:
        """Count of filled slots excluding 'category' and 'feature', which
        drive retrieval differently than point constraints."""
        return len([s for s in self.filled_slots() if s not in ("category", "feature")])

    def copy(self) -> "SlotSet":
        return replace(self)


@dataclass
class Candidate:
    product: Product
    fused_score: float = 0.0
    route_ranks: dict[str, int] = field(default_factory=dict)
    route_scores: dict[str, float] = field(default_factory=dict)
    final_score: Optional[float] = None
    matched_slots: list[str] = field(default_factory=list)


@dataclass
class DistilledProfile:
    """Long-term signal: distilled once at reset() from the harness-provided
    user_profile. Soft priors only -- never a hard filter."""

    preference_tags: list[str]
    decisiveness_prior: float
    rating_floor_hint: float
    summary_short: str


@dataclass
class DistilledSession:
    """Short-term signal: recomputed every turn from SessionState."""

    slot_summary: dict[str, object]
    query_text: str
    turns_remaining: int
    stuck: bool
    track: str
    # asin -> number of *prior* turns already shown (see context.distill_session).
    shown_asin_counts: dict[str, int] = field(default_factory=dict)
    # Calibrated P(this turn is an intent reversal) -- see agent_shopper.override_model.
    override_probability: float = 0.0


@dataclass
class DistilledContext:
    profile: DistilledProfile
    session: DistilledSession


@dataclass
class RoutePlan:
    weights: dict[str, float]
    gate_to_category: bool
    gate_relaxed_slots: tuple[str, ...] = ()
    budget_relaxed: bool = False
    # Hard-marked slots to enforce as an exact AND-filter even when
    # gate_to_category is False (e.g. a hard-marked budget stated while the
    # session is still on the browsing track) -- see retrieval.py's
    # post-fusion hard-filter step and orchestrator.decide_routes.
    hard_filter_slots: tuple[str, ...] = ()

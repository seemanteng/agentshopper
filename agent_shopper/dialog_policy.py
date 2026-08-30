"""Pillar II: the per-turn dialog algorithm.

`process_turn` is the single entry point agent.py calls: extract this
turn's slots -> merge them (Accumulation vs Override) -> distill context ->
ask the orchestrator which routes to run -> retrieve -> ask the orchestrator
whether to clarify -> rerank -> build the response. Everything else in this
module is a pure helper the turn algorithm composes.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import replace

from agent_shopper import context as context_mod
from agent_shopper import intent as intent_mod
from agent_shopper import orchestrator as orchestrator_mod
from agent_shopper import reranker as reranker_mod
from agent_shopper import retrieval as retrieval_mod
from agent_shopper import slots as slots_mod
from agent_shopper.bm25_index import BM25Index
from agent_shopper.catalog import Catalog
from agent_shopper.config import LLM_MAX_FAILURES_BEFORE_CIRCUIT_BREAK, MAX_TURNS, SIMULATOR_DISCLOSABLE_ATTRIBUTES
from agent_shopper.dense_index import DenseIndex
from agent_shopper.dialog_state import OverrideEvent, SessionState, ShownRecord
from agent_shopper.llm_client import active_provider
from agent_shopper.models import Candidate, Product, SlotSet
from agent_shopper.override_model import OverrideModel
from agent_shopper.tfidf_index import TfidfIndex

_CONTRADICTION_RE = re.compile(
    r"\b(actually|instead|changed my mind|no longer|not\b.*\banymore|forget that|never ?mind|"
    r"ignore my earlier|scratch that)\b",
    re.I,
)
_CONJUNCTION_RE = re.compile(r"\b(or|also|either|as well|too)\b", re.I)

# Language that marks whatever slot(s) get filled this turn as a stated hard
# requirement rather than an incidentally-accumulated preference -- see
# SlotSet.hard_marked, orchestrator.decide_routes/relax_gate. Deliberately
# generic real-user phrasing only. Two of the competition simulator's own
# templates ("a key requirement is" from initial_message, "what I need is"
# from the intent_override override message -- see evaluator/local_
# evaluator.py) look like equally reliable hard-constraint signals, but "a
# key requirement is" fires on turn 1 of nearly every buying session, and an
# eval run confirmed that including it here regresses buying-scenario
# HitRate@10 from 0.51 to 0.41: hard-filtering the fused pool on a single,
# possibly-noisy, turn-1 slot extraction excludes the true target more often
# than it protects a real constraint. Left out on purpose -- see README's
# limitations section.
_HARD_CONSTRAINT_RE = re.compile(
    r"\b(max|maximum|no more than|must be|only|exactly|has to be|"
    r"need(?:s)? to be|strictly|absolutely|non[- ]negotiable|required?)\b",
    re.I,
)

_ATTRIBUTE_QUESTIONS = {
    "material": "What material would you like it to be made of?",
    "color": "Do you have a color preference?",
    "size": "What size are you looking for?",
    "style": "What style are you going for?",
    "budget": "What's your budget range?",
    "use_case": "What will you mainly use this for?",
    "feature": "Is there a specific feature that matters most to you?",
    "category": "What type of product are you looking for?",
    "brand": "Do you have a preferred brand?",
    "other": "Could you tell me a bit more about what you're looking for?",
}

_GATE_ATTR_VOCAB = {
    "material": slots_mod.MATERIALS,
    "color": slots_mod.COLORS,
    "style": slots_mod.STYLES,
    "use_case": slots_mod.USE_CASES,
    "size": slots_mod.SIZE_WORDS,
}


# --- Slot merge: Information Accumulation vs Intent Override ----------------


def _has_contradiction_language(message: str) -> bool:
    return bool(_CONTRADICTION_RE.search(message))


def _has_conjunction(message: str) -> bool:
    return bool(_CONJUNCTION_RE.search(message))


def _apply_override(state: SessionState, slot_name: str, old_value: object, new_value: object, turn: int) -> None:
    state.override_events.append(OverrideEvent(turn=turn, slot=slot_name, old_value=old_value, new_value=new_value))
    setattr(state.slots, slot_name, new_value)
    state.turns_since_progress = 0
    state.clarified_attributes.discard(slot_name)


def _clear_incompatible_attributes(state: SessionState) -> None:
    """Category override moved to a different department (see
    slots.same_department) -- style/feature from the old category may not
    apply to the new one. Clears them and reopens them for clarification,
    since a previously-clarified answer for the old category no longer
    necessarily holds. Attributes preserved (same department) stay marked
    clarified, since we already know the answer still applies."""
    state.slots.style = None
    state.slots.feature = []
    state.clarified_attributes -= {"style", "feature"}


def _is_category_refinement(old_value: str, new_value: str) -> bool:
    old_l, new_l = old_value.lower(), str(new_value).lower()
    return old_l == new_l or old_l in new_l or new_l in old_l


def _same_size_domain(old_value: object, new_value: object) -> bool:
    """Whether a new size mention is the same kind of measurement (letter
    vs numeric) as the existing one(s) -- see slots.size_domain. An
    "unknown"-domain value on either side is treated permissively (we can't
    tell, so don't force an override)."""
    old_values = old_value if isinstance(old_value, list) else [old_value]
    new_domain = slots_mod.size_domain(new_value)
    if new_domain == "unknown":
        return True
    return any(slots_mod.size_domain(v) in (new_domain, "unknown") for v in old_values)


def _budget_intersect(old: tuple, new: tuple) -> tuple | None:
    old_lo, old_hi = old
    new_lo, new_hi = new
    los = [v for v in (old_lo, new_lo) if v is not None]
    his = [v for v in (old_hi, new_hi) if v is not None]
    lo = max(los) if los else None
    hi = min(his) if his else None
    if lo is not None and hi is not None and lo > hi:
        return None
    return (lo, hi)


def _merge_one(state: SessionState, slot_name: str, new_value: object, message: str, turn: int, forced_override: bool) -> None:
    old_value = getattr(state.slots, slot_name)

    if slot_name == "feature":
        values = new_value if isinstance(new_value, list) else [new_value]
        for v in values:
            if v not in state.slots.feature:
                state.slots.feature.append(v)
        return

    if old_value is None:
        setattr(state.slots, slot_name, new_value)
        return

    if forced_override:
        _apply_override(state, slot_name, old_value, new_value, turn)
        if slot_name == "category" and not slots_mod.same_department(str(old_value), str(new_value)):
            _clear_incompatible_attributes(state)
        return

    if slot_name == "category":
        if _is_category_refinement(str(old_value), str(new_value)):
            state.slots.category = new_value
        else:
            _apply_override(state, slot_name, old_value, new_value, turn)
            if not slots_mod.same_department(str(old_value), str(new_value)):
                _clear_incompatible_attributes(state)
        return

    if slot_name == "budget":
        merged = _budget_intersect(old_value, new_value)
        if merged is not None:
            state.slots.budget = merged
        else:
            _apply_override(state, slot_name, old_value, new_value, turn)
        return

    # Point constraints: material/color/size/style/brand/use_case.
    existing = old_value if isinstance(old_value, list) else [old_value]
    if str(new_value).lower() in [str(v).lower() for v in existing]:
        return  # already known, no-op
    if slot_name == "size" and not _same_size_domain(old_value, new_value):
        # Letter clothing size ("M") vs numeric shoe/waist size ("9") aren't
        # comparable point values -- treat the new mention as this turn's
        # authoritative size instead of OR-accumulating them into one list.
        _apply_override(state, slot_name, old_value, new_value, turn)
        return
    if _has_conjunction(message):
        merged_list = existing + [new_value]
        setattr(state.slots, slot_name, merged_list if len(merged_list) > 1 else merged_list[0])
    else:
        _apply_override(state, slot_name, old_value, new_value, turn)


def _attribute_contradiction_count(old_slots: SlotSet, extracted: dict[str, object]) -> int:
    """How many of this turn's extracted slots contradict an already-filled,
    *different* value -- mirrors the exact decision `_merge_one` itself
    makes for each slot kind (point-constraint no-op vs. override,
    `_budget_intersect`, `_is_category_refinement`), reused here rather than
    re-implemented, so this can never silently drift from what merging
    actually does. Called on PRE-merge `old_slots` (i.e. `state.slots`
    before `merge_slot_updates` runs this turn) -- see
    process_turn/_override_features."""
    count = 0
    for slot_name, new_value in extracted.items():
        if slot_name == "feature":
            continue  # feature always accumulates, never contradicts
        old_value = getattr(old_slots, slot_name, None)
        if old_value is None:
            continue
        if slot_name == "budget":
            if _budget_intersect(old_value, new_value) is None:
                count += 1
            continue
        if slot_name == "category":
            if not _is_category_refinement(str(old_value), str(new_value)):
                count += 1
            continue
        existing = old_value if isinstance(old_value, list) else [old_value]
        if str(new_value).lower() not in [str(v).lower() for v in existing]:
            count += 1
    return count


def _override_features(
    old_slots: SlotSet, extracted: dict[str, object], contradiction_this_turn: bool, turn: int,
) -> dict[str, float]:
    """Primitive features for agent_shopper.override_model.OverrideModel,
    computed here (not in override_model.py) since this module already owns
    same_department/_budget_intersect/_is_category_refinement -- reusing
    them keeps this in lockstep with the actual merge decisions instead of
    a second, driftable copy (see _attribute_contradiction_count). Called
    on PRE-merge state, i.e. before merge_slot_updates mutates state.slots
    this turn -- see process_turn."""
    department_changed = 0.0
    new_category = extracted.get("category")
    if new_category and old_slots.category and not slots_mod.same_department(str(old_slots.category), str(new_category)):
        department_changed = 1.0
    budget_conflict = 0.0
    new_budget = extracted.get("budget")
    if new_budget and old_slots.budget and _budget_intersect(old_slots.budget, new_budget) is None:
        budget_conflict = 1.0
    return {
        "contradiction_language": 1.0 if contradiction_this_turn else 0.0,
        "department_changed": department_changed,
        "budget_conflict": budget_conflict,
        "attribute_contradiction_count": float(_attribute_contradiction_count(old_slots, extracted)),
        "is_first_turn": 1.0 if turn <= 1 else 0.0,
    }


def merge_slot_updates(state: SessionState, extracted: dict[str, object], message: str, turn: int) -> None:
    forced = _has_contradiction_language(message)
    hard = bool(_HARD_CONSTRAINT_RE.search(message))
    # A single hard-constraint cue could belong to any one of several slots
    # extracted the same turn ("budget must be under $50, maybe cotton" --
    # only budget is hard). extract_slots carries no per-slot span, so when
    # more than one slot is extracted on a hard-cue turn, attribution is
    # genuinely ambiguous -- skip marking anything hard rather than
    # guessing wrong on every extracted slot.
    apply_hard = hard and len(extracted) == 1
    for slot_name, new_value in extracted.items():
        _merge_one(state, slot_name, new_value, message, turn, forced_override=forced)
        if apply_hard:
            state.slots.hard_marked = state.slots.hard_marked | {slot_name}


# --- Proactive clarification: information-gain attribute selection ---------


def _bucket_value(product: Product, attr: str) -> str:
    if attr == "budget":
        if product.price is None:
            return "unknown"
        if product.price < 25:
            return "<25"
        if product.price < 50:
            return "25-50"
        if product.price < 100:
            return "50-100"
        return "100+"
    vocab = _GATE_ATTR_VOCAB.get(attr)
    if vocab is None:
        return "unknown"
    text = " ".join([product.title, " ".join(product.features), " ".join(product.description)]).lower()
    found = slots_mod.find_word(text, vocab)
    return found or "unknown"


def _entropy_and_coverage(candidates: list[Candidate], attr: str) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    buckets: dict[str, int] = defaultdict(int)
    for c in candidates:
        buckets[_bucket_value(c.product, attr)] += 1
    total = len(candidates)
    known = total - buckets.get("unknown", 0)
    coverage = known / total
    if len(buckets) <= 1:
        return 0.0, coverage
    probs = [count / total for count in buckets.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(buckets))
    return (entropy / max_entropy if max_entropy > 0 else 0.0), coverage


_CLARIFY_TIEBREAK = ("budget", "use_case", "style", "material", "color", "size", "feature")

# Per-department clarify-attribute relevance weights. Reuses
# slots.CATEGORY_DEPARTMENT rather than a third hand-maintained table --
# e.g. favors asking about "size" over "color" for footwear. Validated via
# scripts/run_local_eval.py on the 200-session public dev set --
# TechnicalScore 0.4186->0.4207, MRR 0.3197->0.3286, HitRate@10 unchanged,
# MTTC roughly flat (6.87->6.89). No scenario_type bucket regressed.
_DEPARTMENT_CLARIFY_RELEVANCE = {
    "footwear": {"size": 1.3, "material": 1.15},
    "jewelry": {"material": 1.2, "color": 1.1},
    "apparel": {"size": 1.3, "style": 1.15},
    "accessories": {"color": 1.1, "material": 1.1},
    "bags": {"material": 1.15, "color": 1.05},
}


def _clarify_relevance(attr: str, category: object) -> float:
    if not category:
        return 1.0
    department = slots_mod.CATEGORY_DEPARTMENT.get(str(category).lower())
    if department is None:
        return 1.0
    return _DEPARTMENT_CLARIFY_RELEVANCE.get(department, {}).get(attr, 1.0)


def choose_clarify_attribute(candidates: list[Candidate], slots: SlotSet, clarified_attributes: set[str]) -> tuple[str | None, float]:
    """Picks the unfilled, not-yet-asked attribute that best splits the
    current candidate pool (normalized entropy * catalog coverage), damped
    by a per-department relevance weight (see _clarify_relevance). Returns
    (None, 0.0) if nothing clears a useful score -- this doubles as the
    session's clarify-exhaustion signal (see process_turn): once every
    disclosable attribute is filled, asked, or scores too low, there is
    nothing left worth asking about. Restricted to the attribute buckets
    the competition simulator's classify_constraint() actually discloses
    info for -- "category"/"brand" are excluded since they never yield new
    disclosure (see config.SIMULATOR_DISCLOSABLE_ATTRIBUTES)."""
    scores: dict[str, float] = {}
    for attr in SIMULATOR_DISCLOSABLE_ATTRIBUTES:
        if attr in clarified_attributes:
            continue
        if attr != "feature" and getattr(slots, attr):
            continue
        if attr == "feature":
            entropy, coverage = 0.3, 1.0  # catch-all bucket, no clean split signal
        else:
            entropy, coverage = _entropy_and_coverage(candidates, attr)
        scores[attr] = entropy * coverage * _clarify_relevance(attr, slots.category)
    if not scores:
        return None, 0.0
    best_attr = max(
        scores,
        key=lambda a: (scores[a], -_CLARIFY_TIEBREAK.index(a) if a in _CLARIFY_TIEBREAK else 0),
    )
    return best_attr, scores[best_attr]


def _build_message(
    ask_attribute: str | None, num_recs: int, relaxed_slots: list[str], forced_relax_attribute: str | None = None,
) -> str:
    if forced_relax_attribute:
        # Every soft-preference slot is already exhausted (see the relax
        # loop in process_turn); the only thing left to drop was stated as a
        # hard requirement, so we ask rather than silently drop it.
        if num_recs:
            return (
                f"I couldn't find anything matching every requirement, so here "
                f"{'is' if num_recs == 1 else 'are'} {num_recs} option{'s' if num_recs != 1 else ''} that come close "
                f"except on {forced_relax_attribute}. Would you be open to adjusting that requirement?"
            )
        return (
            f"I couldn't find anything matching every requirement you've given me, even outside your "
            f"{forced_relax_attribute} requirement. Would you be open to adjusting it?"
        )
    if num_recs:
        if relaxed_slots:
            lead = f"I loosened the {relaxed_slots[-1]} requirement slightly and found {num_recs} option{'s' if num_recs != 1 else ''} for you."
        else:
            lead = f"Here are {num_recs} option{'s' if num_recs != 1 else ''} that match what you've told me so far."
    else:
        lead = "I couldn't find a close match yet with what you've told me so far."
    if ask_attribute:
        return f"{lead} {_ATTRIBUTE_QUESTIONS.get(ask_attribute, _ATTRIBUTE_QUESTIONS['other'])}"
    return lead


# --- The full per-turn algorithm --------------------------------------------


def process_turn(
    catalog: Catalog,
    bm25_index: BM25Index,
    tfidf_index: TfidfIndex,
    state: SessionState,
    message: str,
    turn: int,
    top_k: int,
    dense_index: DenseIndex | None = None,
) -> dict:
    state.turn = turn

    extracted = slots_mod.extract_slots(message)
    contradiction_this_turn = _has_contradiction_language(message)

    # Computed on PRE-merge state.slots (the "old" values merge_slot_updates
    # is about to compare `extracted` against) -- must run before that call
    # mutates state.slots in place. A continuous score, not a hard gate: see
    # agent_shopper.override_model's docstring for why this is meant to
    # feed a downstream feature/policy as one input among many, rather than
    # driving a hand-coded intervention directly the way an earlier,
    # reverted attempt did (README's "What we tried"). Currently unconsumed
    # (a LambdaMART reranker attempt that would have used it was tried and
    # rejected by CV -- also in "What we tried") -- computed regardless
    # since it's cheap and the next consumer (a future confidence-weighted
    # slots pass) will want the same historical signal available.
    override_features = _override_features(state.slots, extracted, contradiction_this_turn, turn)
    state.override_probability = OverrideModel().predict_proba(override_features)

    merge_slot_updates(state, extracted, message, turn)

    track = intent_mod.classify(
        message, state.slots, turn, state.has_overridden(), state.distilled_profile.decisiveness_prior,
        current_track=state.track,
    )
    state.track = track

    ctx = context_mod.distill(
        state.distilled_profile, state.slots, message, turn, state.turns_since_progress, track, state.shown_history,
        state.has_overridden(), state.override_probability,
    )

    plan, effective_slots = orchestrator_mod.decide_routes(track, state.slots)
    if contradiction_this_turn and plan.gate_to_category:
        # An override signal means some of our accumulated slots may now be
        # stale. Trusting the hard AND-filter here would silently exclude
        # the true target if it doesn't match the *old* constraint -- drop
        # the gate for this turn and let the fresh query text (which always
        # includes this turn's own tokens, see context.build_query_text)
        # compete via fused ranking instead of a hard filter. Slots that
        # genuinely carried over unaffected still contribute as a soft
        # signal through the browsing-style category route.
        plan = replace(plan, gate_to_category=False)
    result = retrieval_mod.retrieve(catalog, bm25_index, tfidf_index, ctx.session.query_text, effective_slots, plan, dense_index=dense_index)

    relaxed_slots: list[str] = []
    forced_relax_attribute: str | None = None
    attempts = 0
    while plan.gate_to_category and result.pool_size == 0 and attempts < 8:
        relaxed = orchestrator_mod.relax_gate(effective_slots)
        if relaxed is None:
            # Nothing left to drop at all (only category filled, or
            # nothing) -- de-gate fully, but keep enforcing any hard-marked
            # slots as a post-fusion filter rather than dropping them too.
            plan = replace(plan, gate_to_category=False, hard_filter_slots=tuple(effective_slots.hard_marked))
            result = retrieval_mod.retrieve(catalog, bm25_index, tfidf_index, ctx.session.query_text, effective_slots, plan, dense_index=dense_index)
            break
        candidate_slots, dropped_name, was_hard = relaxed
        if was_hard:
            # Only hard-marked slots remain droppable and the pool is still
            # empty -- don't silently drop a stated hard requirement. Ask
            # the user instead, and de-gate (keeping it as a filter, not an
            # AND with the now-abandoned category gate) so we still have a
            # best-effort pool to show this turn.
            forced_relax_attribute = dropped_name
            plan = replace(plan, gate_to_category=False, hard_filter_slots=tuple(effective_slots.hard_marked))
            result = retrieval_mod.retrieve(catalog, bm25_index, tfidf_index, ctx.session.query_text, effective_slots, plan, dense_index=dense_index)
            break
        effective_slots = candidate_slots
        relaxed_slots.append(dropped_name)
        result = retrieval_mod.retrieve(catalog, bm25_index, tfidf_index, ctx.session.query_text, effective_slots, plan, dense_index=dense_index)
        attempts += 1

    if ctx.session.stuck and result.pool_size > 0:
        widened = orchestrator_mod.widen_budget(effective_slots)
        if widened is not None:
            widened_result = retrieval_mod.retrieve(catalog, bm25_index, tfidf_index, ctx.session.query_text, widened, plan, dense_index=dense_index)
            if widened_result.pool_size > result.pool_size:
                effective_slots, result = widened, widened_result
                relaxed_slots.append("budget")

    best_attr, best_score = choose_clarify_attribute(result.candidates, state.slots, state.clarified_attributes)
    if forced_relax_attribute is not None:
        # A stated hard requirement is the reason nothing matches -- ask
        # about it directly rather than going through the normal
        # entropy-based proactive clarification.
        do_clarify = True
        ask_attribute = forced_relax_attribute
    else:
        # Natural exhaustion: nothing left to ask about once best_attr is
        # None (every disclosable attribute is filled, already asked, or
        # scores below MIN_CLARIFY_SPLIT_SCORE) -- see
        # choose_clarify_attribute's docstring.
        do_clarify = best_attr is not None and orchestrator_mod.should_clarify(
            result.pool_size, state.slots.hard_filled_count(), ctx.session.turns_remaining, best_score
        )
        ask_attribute = best_attr if do_clarify else None

    engine = orchestrator_mod.decide_rerank_engine(
        result.pool_size, turn, MAX_TURNS, active_provider() is not None, state.llm_disabled, do_clarify=do_clarify,
    )
    if engine == "llm":
        rr = reranker_mod.LLMReranker()
        ranked = rr.rerank(ctx, result.candidates, top_k)
        if rr.last_call_used_llm:
            state.llm_failure_count = 0
        else:
            state.llm_failure_count += 1
            if state.llm_failure_count >= LLM_MAX_FAILURES_BEFORE_CIRCUIT_BREAK:
                state.llm_disabled = True
    else:
        ranked = reranker_mod.HeuristicReranker().rerank(ctx, result.candidates, top_k)

    recommendations = [
        {"parent_asin": c.product.parent_asin, "score": round(c.final_score if c.final_score is not None else c.fused_score, 6)}
        for c in ranked[:top_k]
    ]

    state.shown_history.extend(
        ShownRecord(turn=turn, parent_asin=r["parent_asin"]) for r in recommendations
    )
    state.record_turn(message, ask_attribute, result.pool_size)

    return {
        "message": _build_message(ask_attribute, len(recommendations), relaxed_slots, forced_relax_attribute),
        "ask_attribute": ask_attribute,
        "recommendations": recommendations,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

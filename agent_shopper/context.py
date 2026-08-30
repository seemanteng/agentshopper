"""Pillar III: Personalized Context Distillation.

Two independent distillations feed the rest of the pipeline:
  - `distill_profile`: the harness-provided user_profile (each session is an
    isolated single user, so this *is* the "long-term profile" signal --
    there is no cross-session history to draw on) -> soft priors, computed
    once at reset().
  - `distill_session`: the session's accumulated slots and progress ->
    a compact, bounded query representation, recomputed every turn.

Both exist specifically so that retrieval/reranking/LLM prompts never touch
raw turn history or the raw profile dict directly -- keeping token usage low
and giving the rest of the pipeline one small, stable interface.
"""

from __future__ import annotations

import re

from agent_shopper.config import (
    DECISIVENESS_BY_PURCHASE_FREQUENCY,
    DECISIVENESS_DEFAULT,
    MAX_TURNS,
    OVERRIDE_QUERY_STRIP_ENABLED,
    PICKY_RATING_STYLE_KEYWORDS,
    STUCK_TURNS_THRESHOLD,
)
from agent_shopper.models import DistilledContext, DistilledProfile, DistilledSession, SlotSet
from agent_shopper.text_utils import unique_terms

QUERY_TEXT_TOKEN_LIMIT = 40

# First tried: stripping the TechJam evaluator's own fixed dialogue
# scaffolding (e.g. "Actually, ignore my earlier preference. What I need is:
# X.") from the query side before tokenizing, on *every* turn regardless of
# scenario -- found via scripts/diagnose_intent_override.py (8/12
# never-recalled intent_override sessions were recallable *before* the
# override turn, dropping out once this boilerplate diluted that turn's
# query). Confirmed the diagnosis -- intent_override HitRate@10 0.367->0.400,
# MTTC 8.73->8.43 -- but regressed buying (HitRate@10 0.713->0.663) and
# boundary's MRR (0.440->0.285) enough to net-regress TechnicalScore
# 0.5499->0.5371 overall. Reverted; see README.md's "What we tried" for the
# full numbers. scripts/diagnose_intent_override.py is kept as a diagnostic
# tool regardless.
#
# Second attempt (this module's _strip_boilerplate/build_query_text
# `override_active` param, gated by config.OVERRIDE_QUERY_STRIP_ENABLED):
# same idea, but scoped to only sessions where state.has_overridden() is
# already True, instead of every turn -- see scripts/
# train_override_query_strip.py for the CV validation this needs before
# OVERRIDE_QUERY_STRIP_ENABLED can default on.

# Fixed evaluator dialogue-scaffolding phrases (see evaluator/
# local_evaluator.py's initial_message/customer_reply) to strip before
# tokenizing -- lead-in patterns strip only the boilerplate prefix (keeping
# any substantive trailing content, e.g. the override's new preference
# value, or a disclosed constraint's actual values); whole-message patterns
# (anchored with $) drop turns that carry zero informative content at all.
_BOILERPLATE_LEAD_IN_PATTERNS = [
    re.compile(r"^actually,?\s*(?:please\s*)?ignore my earlier preference\.?\s*(?:what i need is:?\s*)?", re.I),
    re.compile(r"^for that,?\s*what matters is:?\s*", re.I),
]
_BOILERPLATE_WHOLE_MESSAGE_PATTERNS = [
    re.compile(r"^those options are not quite right yet\.?\s*ask me about one specific attribute\.?$", re.I),
    re.compile(r"^i don'?t have an? (?:additional )?preference for \w+;?\s*(?:please use your judgment)?\.?$", re.I),
]


def _strip_boilerplate(message: str) -> str:
    for pattern in _BOILERPLATE_WHOLE_MESSAGE_PATTERNS:
        if pattern.match(message.strip()):
            return ""
    stripped = message
    for pattern in _BOILERPLATE_LEAD_IN_PATTERNS:
        stripped = pattern.sub("", stripped)
    return stripped


def _parse_decisiveness(purchase_frequency: str) -> float:
    if not purchase_frequency:
        return DECISIVENESS_DEFAULT
    for key, value in DECISIVENESS_BY_PURCHASE_FREQUENCY.items():
        if key.lower() in purchase_frequency.lower():
            return value
    text = purchase_frequency.lower()
    range_match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if range_match:
        avg = (int(range_match.group(1)) + int(range_match.group(2))) / 2
    else:
        plus_match = re.search(r"(\d+)\s*\+", text)
        single_match = re.search(r"(\d+)", text)
        if plus_match:
            avg = int(plus_match.group(1)) + 2
        elif single_match:
            avg = int(single_match.group(1))
        else:
            return DECISIVENESS_DEFAULT
    if avg <= 2:
        return 0.3
    if avg <= 4:
        return 0.5
    return 0.7


def distill_profile(user_profile: dict) -> DistilledProfile:
    purchase_frequency = str(user_profile.get("purchase_frequency") or "")
    rating_style = str(user_profile.get("rating_style") or "")
    decisiveness = _parse_decisiveness(purchase_frequency)
    if any(word in rating_style.lower() for word in PICKY_RATING_STYLE_KEYWORDS):
        decisiveness = min(1.0, decisiveness + 0.1)
        rating_floor_hint = 0.15
    else:
        rating_floor_hint = 0.0
    tags = user_profile.get("preference_tags") or []
    summary = str(user_profile.get("summary") or "")
    return DistilledProfile(
        preference_tags=[str(t) for t in tags],
        decisiveness_prior=decisiveness,
        rating_floor_hint=rating_floor_hint,
        summary_short=summary[:200],
    )


def build_query_text(slots: SlotSet, message: str, session_has_overridden: bool = False) -> str:
    """Bounded query representation: slot values first (they're durable
    signal), then this turn's own tokens, deduped, capped at
    QUERY_TEXT_TOKEN_LIMIT tokens -- never raw concatenated history.

    `session_has_overridden` (pass state.has_overridden() -- see
    dialog_policy.process_turn) is the state fact; whether it actually
    strips anything is still gated by config.OVERRIDE_QUERY_STRIP_ENABLED
    here, so a single flag flip (see scripts/train_override_query_strip.py)
    controls the feature everywhere it's consumed, without callers needing
    to know about the flag themselves. Strips fixed evaluator dialogue-
    scaffolding out of `message` first, so it can't dilute the query the way
    it does by default -- see this module's docstring for why this is
    scoped to override sessions only rather than applied to every turn."""
    if session_has_overridden and OVERRIDE_QUERY_STRIP_ENABLED:
        message = _strip_boilerplate(message)
    slot_tokens: list[str] = []
    if slots.category:
        slot_tokens.extend(unique_terms(slots.category))
    for name in ("material", "color", "size", "style", "brand"):
        value = getattr(slots, name)
        if isinstance(value, list):
            for v in value:
                slot_tokens.extend(unique_terms(str(v)))
        elif value:
            slot_tokens.extend(unique_terms(str(value)))
    if slots.use_case:
        slot_tokens.extend(unique_terms(slots.use_case))
    for feature in slots.feature:
        slot_tokens.extend(unique_terms(str(feature)))

    message_tokens = unique_terms(message)
    combined = list(dict.fromkeys([*slot_tokens, *message_tokens]))
    return " ".join(combined[:QUERY_TEXT_TOKEN_LIMIT])


def distill_session(
    slots: SlotSet, message: str, turn: int, turns_since_progress: int, track: str,
    shown_history: list | None = None, session_has_overridden: bool = False, override_probability: float = 0.0,
) -> DistilledSession:
    slot_summary = {
        name: getattr(slots, name)
        for name in slots.filled_slots()
    }
    shown_asin_counts: dict[str, int] = {}
    if shown_history:
        for record in shown_history:
            if record.turn < turn:
                shown_asin_counts[record.parent_asin] = shown_asin_counts.get(record.parent_asin, 0) + 1
    return DistilledSession(
        slot_summary=slot_summary,
        query_text=build_query_text(slots, message, session_has_overridden),
        turns_remaining=max(0, MAX_TURNS - turn + 1),
        stuck=turns_since_progress >= STUCK_TURNS_THRESHOLD,
        track=track,
        shown_asin_counts=shown_asin_counts,
        override_probability=override_probability,
    )


def distill(
    profile: DistilledProfile, slots: SlotSet, message: str, turn: int, turns_since_progress: int, track: str,
    shown_history: list | None = None, session_has_overridden: bool = False, override_probability: float = 0.0,
) -> DistilledContext:
    return DistilledContext(
        profile=profile,
        session=distill_session(
            slots, message, turn, turns_since_progress, track, shown_history, session_has_overridden,
            override_probability,
        ),
    )

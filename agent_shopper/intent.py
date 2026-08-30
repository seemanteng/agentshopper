"""Pillar I: per-turn Buying vs Browsing intent classification.

Rule-based by design (no training/fine-tuning is in scope for this
challenge). Recomputed every turn from the current message, accumulated
slots, and the long-term profile's decisiveness prior -- a session can flip
tracks as the conversation evolves.

`score_intent` deliberately keeps two signals separate rather than folding
them into one scalar: how *decisive the shopper's language is* ("language",
e.g. "I need to buy" vs. "just browsing for ideas") and how *specified the
request already is* ("specificity", from accumulated slots). A user can be
highly specific while still explicitly browsing ("black waterproof jackets
under $100, just exploring"), and conflating the two would push that turn
toward "buying" on constraint count alone. `classify` is the only place the
two signals get combined into a single track decision, and it adds
hysteresis so a marginal score change doesn't flip the track turn to turn.
"""

from __future__ import annotations

import re

from agent_shopper.config import INTENT_BUYING_THRESHOLD
from agent_shopper.models import SlotSet

_BUDGET_RE = re.compile(r"\$\s*\d|under|less than|around \$|budget|no more than", re.I)
_IMPERATIVE_RE = re.compile(
    r"\b(buy|need|want|get me|looking for a|looking for the|show me only|purchase|order)\b", re.I
)
_BROWSING_RE = re.compile(
    r"\b(browsing|just looking|ideas|options|explore|exploring|variety|what do you have|any suggestions|not sure)\b",
    re.I,
)
_OPEN_QUESTION_RE = re.compile(r"^(what|which|any)\b.*\?$", re.I)

# Turn-count tie-breaker: only nudges a genuinely borderline score, and only
# when this turn's language gave no real signal either way -- a long
# conversation alone should not manufacture buying intent out of neutral
# language, but it's still a reasonable tie-break once nothing else says.
_TURN_NUDGE_MIN_TURN = 3
_TURN_NUDGE_AMOUNT = 0.5
_TURN_NUDGE_EPSILON = 0.5
_TURN_NUDGE_LANGUAGE_CEILING = 1.0

# Hysteresis margin required to flip *away* from the current track -- avoids
# flapping when the combined score crosses INTENT_BUYING_THRESHOLD by a
# hair. Turn 1 has no real prior to be sticky about, so it's exempt.
_TRACK_SWITCH_MARGIN = 0.75


def score_intent(message: str, slots: SlotSet, decisiveness_prior: float) -> tuple[float, float]:
    """Returns (language_score, specificity_score) separately -- see module
    docstring for why they aren't summed here."""
    language = 0.0
    if _BUDGET_RE.search(message):
        language += 2.0
    if _IMPERATIVE_RE.search(message):
        language += 2.0
    if _BROWSING_RE.search(message):
        language -= 2.0
    if _OPEN_QUESTION_RE.match(message.strip()):
        language -= 1.0
    # decisiveness_prior in [0, 1]; center at 0.5 so it's a +/-1 nudge.
    language += (decisiveness_prior - 0.5) * 2.0

    specificity = min(4.0, slots.hard_filled_count())
    return language, specificity


def classify(
    message: str,
    slots: SlotSet,
    turn: int,
    has_overridden: bool,
    decisiveness_prior: float,
    current_track: str = "browsing",
) -> str:
    language, specificity = score_intent(message, slots, decisiveness_prior)
    combined = language + specificity

    if (
        turn >= _TURN_NUDGE_MIN_TURN
        and not has_overridden
        and abs(language) < _TURN_NUDGE_LANGUAGE_CEILING
        and abs(combined - INTENT_BUYING_THRESHOLD) <= _TURN_NUDGE_EPSILON
    ):
        combined += _TURN_NUDGE_AMOUNT

    provisional = "buying" if combined >= INTENT_BUYING_THRESHOLD else "browsing"
    if turn <= 1 or provisional == current_track:
        return provisional
    if provisional == "buying" and combined >= INTENT_BUYING_THRESHOLD + _TRACK_SWITCH_MARGIN:
        return "buying"
    if provisional == "browsing" and combined <= INTENT_BUYING_THRESHOLD - _TRACK_SWITCH_MARGIN:
        return "browsing"
    return current_track

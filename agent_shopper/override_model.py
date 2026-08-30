"""Calibrated P(this turn is an intent reversal).

A continuous score meant to feed a downstream feature/policy as one input
among many, NOT to drive a hand-coded intervention directly. Two attempts
at using this signal directly have already been tried and rejected, both
documented in full in README's "What we tried": (1) the cheaper *boolean*
`state.has_overridden()` signal was used to hard-gate a query-text
intervention (stripping evaluator dialogue scaffolding on override-affected
turns) -- mechanically correct, but a net wash when measured (4/30
intent_override sessions changed outcome, net 0 HitRate@10, MRR regressed;
5-fold CV: 2 wins/1 loss/2 ties, mean TechnicalScore delta -0.0011); (2) a
LambdaMART reranker used `override_probability` (and its interaction with
`rating`) as one of 14 features -- 5-fold CV: 2 wins/3 losses, mean
TechnicalScore delta -0.0188, rejected, and the reranker code was removed
rather than kept gated off. Neither result is evidence the override signal
itself is worthless -- both are evidence that the specific lever tried
wasn't the right one. This module (and `state.override_probability`) is
kept as infrastructure for whatever tries next, e.g. a future confidence-
weighted slot representation.

Pure model, no dialog_policy dependency (avoids a circular import, since
dialog_policy imports this module): scoring is just a calibrated logistic
function over a small feature dict. See
`agent_shopper.dialog_policy._override_features` for how those primitive
features are actually computed from a turn's message/slots -- kept there
since that module already owns same_department/_budget_intersect/
_is_category_refinement, and reusing them (rather than a second copy here)
keeps this in lockstep with what slot merging actually decides.
"""

from __future__ import annotations

import math

from agent_shopper.config import OVERRIDE_MODEL_INTERCEPT, OVERRIDE_MODEL_WEIGHTS

FEATURE_NAMES = (
    "contradiction_language",
    "department_changed",
    "budget_conflict",
    "attribute_contradiction_count",
    "is_first_turn",
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class OverrideModel:
    """score(features) = sigmoid(intercept + w . features). `weights`/
    `intercept` default to the fitted config values (see
    scripts/train_override_model.py); a caller can substitute different
    coefficients without touching this scoring logic, same pattern
    HeuristicReranker(weights=...) already uses."""

    def __init__(self, weights: dict[str, float] | None = None, intercept: float | None = None) -> None:
        self.weights = weights if weights is not None else OVERRIDE_MODEL_WEIGHTS
        self.intercept = intercept if intercept is not None else OVERRIDE_MODEL_INTERCEPT

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.intercept + sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES)
        return _sigmoid(z)

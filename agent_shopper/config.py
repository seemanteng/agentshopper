"""Central home for every tunable weight/threshold in the pipeline.

Almost nothing here is fit or learned -- these are hand-set priors, tuned
against `scripts/run_local_eval.py` results on the public dev set. The one
exception is `HEURISTIC_RERANK_WEIGHTS`, fit by
`scripts/train_reranker_weights.py` (logistic regression, validated by
k-fold cross-validation) rather than hand-picked -- see its own comment
below and README.md's "what we tried" for why. Every value can be
overridden via an environment variable of the same name for quick sweeps
(see `_env_float`/`_env_int` below) without editing code.

This file holds only the single best-known configuration -- every tunable
here is either a plain constant or a knob with genuine operational purpose
(LLM cost control, latency). Speculative variants that were tried and
measured (via scripts/run_local_eval.py) and did not beat this configuration
have been removed rather than kept around as disabled toggles; see
README.md's "what we tried" section for that history.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Pillar I: intent routing ---------------------------------------------

# Score threshold at/above which a session's current turn is classified "buying".
INTENT_BUYING_THRESHOLD = _env_float("AGENT_SHOPPER_INTENT_THRESHOLD", 3.0)

# --- Pillar I: retrieval fusion ---------------------------------------------

# Reciprocal Rank Fusion smoothing constant (standard RRF choice).
RRF_K = _env_int("AGENT_SHOPPER_RRF_K", 60)

# Route weights per track. Buying leans on the structured category/attribute
# filter; browsing leans on the vector/dense routes for cross-category
# diversity. `dense` added alongside `keyword`/`vector` (not replacing
# either) -- scripts/diagnose_retrieval.py's per-route Recall@10/50/100
# breakdown showed keyword and vector are highly overlapping and both cap
# out around 15-21% R@100, motivating a genuinely different signal rather
# than a third lexical route. Hand-set starting point pending a
# scripts/run_local_eval.py sweep -- not yet tuned the way the other weights
# in this file are.
ROUTE_WEIGHTS = {
    "buying": {"category": 0.45, "keyword": 0.30, "vector": 0.10, "dense": 0.15},
    "browsing": {"category": 0.15, "keyword": 0.25, "vector": 0.30, "dense": 0.30},
}

# Buying track hard-gates retrieval to the category/attribute filter's exact
# match set once this many hard slots are filled.
BUYING_GATE_MIN_SLOTS = _env_int("AGENT_SHOPPER_GATE_MIN_SLOTS", 2)

# Slot priority order, low to high, used both when the gate must be relaxed
# (drop the lowest-priority slot first) and as a clarify-attribute tiebreak.
SLOT_PRIORITY = ("feature", "style", "brand", "material", "size", "color", "budget", "category")

BM25_K1 = _env_float("AGENT_SHOPPER_BM25_K1", 1.5)
BM25_B = _env_float("AGENT_SHOPPER_BM25_B", 0.75)

# Field weighting for BM25, applied by literal token repetition when the doc
# is built (a stock BM25 formula has no native per-field weight knob).
BM25_FIELD_WEIGHTS = {"title": 3, "features": 2, "description": 1, "store": 1}

TFIDF_MAX_FEATURES = _env_int("AGENT_SHOPPER_TFIDF_MAX_FEATURES", 50_000)
TFIDF_NGRAM_RANGE = (1, 2)

# --- Pillar I: dense semantic retrieval (the "dense" route) -----------------

# Frozen sentence-embedding model, no fine-tuning (competition spec puts
# full base-model training out of scope; a frozen encoder is explicitly
# allowed). Small and CPU-friendly by design -- see agent_shopper.dense_index.
DENSE_MODEL_NAME = os.environ.get("AGENT_SHOPPER_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Per-field weights combining title/attributes/category/description cosine
# similarities into one dense-route score. Title-heaviest, mirroring
# BM25_FIELD_WEIGHTS' existing precedent (title carries the most reliable
# signal); description lightest (catalog description snippets are often
# generic boilerplate). Hand-set starting point pending a
# scripts/run_local_eval.py sweep.
DENSE_FIELD_WEIGHTS = {"title": 0.45, "attributes": 0.30, "category": 0.15, "description": 0.10}

# Directory dense_index.py caches its (slow to build) embedding matrices in,
# keyed by catalog content + model name -- see its module docstring.
DENSE_CACHE_DIR = os.environ.get("AGENT_SHOPPER_DENSE_CACHE_DIR", "data")

# --- Pillar I: reranking ---------------------------------------------------

RERANK_CANDIDATE_LIMIT = _env_int("AGENT_SHOPPER_RERANK_LIMIT", 20)

# Learned, not hand-tuned: fit by scripts/train_reranker_weights.py
# (LogisticRegression(class_weight="balanced") on the 5 features below,
# labeled by whether a candidate was the session's true target) on the full
# 200-session public dev set, after a 5-fold stratified cross-validation run
# validated the approach generalizes (5/5 folds improved TechnicalScore on
# held-out sessions never seen during that fold's fit, mean delta +0.0613,
# net +16 hit-turn flips, zero folds net-negative -- see README.md's "What
# we tried" for the full fold-by-fold numbers). No intercept: this score is
# a raw ranking score, never passed through a sigmoid, so an intercept would
# only shift every candidate in a turn by the same constant and never change
# rank order. `rating` dominating by ~5x is a genuine fitted result, not a
# typo -- re-run the training script (same command, same seed) to reproduce.
#
# Twice tried adding a learned `preference_tag` feature and refitting --
# reverted both times, see README.md's "What we tried" and
# reranker._feature_vector's docstring for the full story.
HEURISTIC_RERANK_WEIGHTS = {
    "bm25": 1.693,
    "vector": 1.9413,
    "attr_match": 1.9814,
    "rating": 9.0258,
    "price_fit": 0.2614,
}

# Forces the free/local path even when an API key is present -- used for fast
# iteration during local tuning, and as the honest zero-cost path when no key
# is configured (a paid LLM is not required to run this agent).
FORCE_HEURISTIC = os.environ.get("AGENT_SHOPPER_FORCE_HEURISTIC", "") not in ("", "0", "false", "False")

LLM_MODEL_ENV = "AGENT_SHOPPER_RERANK_MODEL"
LLM_MODEL_FALLBACK = "gpt-4o-mini"
LLM_MAX_FAILURES_BEFORE_CIRCUIT_BREAK = 2

# --- Pillar II: dialog state machine ----------------------------------------

# A candidate pool bigger than this (and otherwise ambiguous) is "over-general".
OVER_GENERALITY_POOL_SIZE = _env_int("AGENT_SHOPPER_OVER_GENERALITY_POOL", 60)

# A pool this small or smaller is already precise enough: never clarify, and
# never spend an LLM call on it -- just run the free heuristic composite
# (attribute match / rating / price fit) and answer. Validated via
# scripts/run_local_eval.py: running the heuristic composite here instead of
# leaving the pool in raw fused-RRF order is measurement-neutral on the
# public dev set (no regression, no measurable gain) but strictly more
# informed, so it's the default.
TIGHT_POOL_SIZE = _env_int("AGENT_SHOPPER_TIGHT_POOL", 10)

# Minimum normalized-entropy*coverage score an attribute must clear to be
# worth asking about.
MIN_CLARIFY_SPLIT_SCORE = _env_float("AGENT_SHOPPER_MIN_SPLIT_SCORE", 0.15)

# Below this many hard slots filled, over-generality clarification is allowed
# to fire (past this, we trust the pool is already reasonably targeted).
CLARIFY_MAX_FILLED_SLOTS = _env_int("AGENT_SHOPPER_CLARIFY_MAX_FILLED", 5)

MAX_TURNS = 10

# Attribute buckets the simulator's classify_constraint() actually produces
# (see evaluator/local_evaluator.py). "category" and "brand" are never
# produced by it, so asking about them yields no new disclosure from the
# simulated customer -- deprioritized accordingly in dialog_policy.
SIMULATOR_DISCLOSABLE_ATTRIBUTES = (
    "budget", "material", "color", "size", "style", "use_case", "feature",
)

# --- Pillar III: adaptive orchestration -------------------------------------

# Turns remaining at/below which we stop clarifying and force a best-effort
# recommendation, since MTTC is weighted heavily and late clarification is
# net-negative.
ENDGAME_TURNS_REMAINING = _env_int("AGENT_SHOPPER_ENDGAME_TURNS", 2)

# Consecutive no-progress turns before we consider a session "stuck" and
# start relaxing constraints.
STUCK_TURNS_THRESHOLD = _env_int("AGENT_SHOPPER_STUCK_TURNS", 2)

BUDGET_RELAX_FACTOR = _env_float("AGENT_SHOPPER_BUDGET_RELAX", 0.30)

# Soft multiplicative penalty on a candidate already shown on
# REJECTED_ITEM_MIN_SHOWN_TURNS+ prior turns without the session converting.
# Never a hard exclusion -- MRR/MTTC only score the *first* hit turn, so
# blanket-excluding a previously-shown true target could only ever hurt
# (hide it from a later turn where it'd rank better), never help. Validated
# via scripts/run_local_eval.py (in isolation, and stacked on top of the
# per-attribute clarify-exhaustion fix below): both runs showed the same
# clean, net-positive per-session diff -- miss->hit flips outnumbering
# hit->miss flips several-to-one, rank improvements with zero regressions,
# against one known, accepted hit->miss loss (`public_0046`,
# `intent_override`). Stacked on top of the rest of the pipeline:
# TechnicalScore 0.4855->0.4927, HitRate@10 0.565->0.575, MRR 0.369->0.376,
# MTTC 6.38->6.375.
REJECTED_ITEM_MIN_SHOWN_TURNS = _env_int("AGENT_SHOPPER_REJECTED_ITEM_MIN_TURNS", 2)
REJECTED_ITEM_DEMOTION_FACTOR = _env_float("AGENT_SHOPPER_REJECTED_ITEM_DEMOTION_FACTOR", 0.85)

# --- Intent override: calibrated P(override) model --------------------------

# Fit by scripts/train_override_model.py (LogisticRegression on
# agent_shopper.dialog_policy._override_features' 5 features, labeled by
# whether a turn is the ground-truth scripted override turn -- from
# sample["behavior"]["override"]["turn"] on intent_override sessions, 0
# everywhere else across all 200 public sessions; 1073 total turns, 30
# positive). 5-fold CV: AUC=1.0/precision=1.0/recall=1.0 on every fold --
# expected, not overfitting: `contradiction_language` alone is a
# near-perfect predictor of this scripted dataset's labels by construction
# (the override message literally contains "ignore my earlier preference").
# The real test of this model isn't its own classification accuracy but
# whether `override_probability` moves intent_override's numbers once
# something downstream actually consumes it (a LambdaMART reranker attempt
# was tried and rejected by CV -- see README's "What we tried" -- so
# nothing currently does) -- `budget_conflict` never fired in training data
# (no budget-triggered overrides among the 30 examples) and
# `department_changed` is ~0
# (redundant with contradiction_language at the override turn itself), left
# in rather than dropped since they may still contribute on override
# patterns the public set's 30 examples don't cover. See
# agent_shopper/override_model.py for how these are used
# (sigmoid(intercept + w.features)).
OVERRIDE_MODEL_WEIGHTS = {
    "contradiction_language": 8.1169,
    "department_changed": -0.0409,
    "budget_conflict": 0.0,
    "attribute_contradiction_count": 0.7463,
    "is_first_turn": -0.8277,
}
OVERRIDE_MODEL_INTERCEPT = -4.0721

# --- Intent override: query-side boilerplate suppression -------------------

# Whether to strip the evaluator's own fixed dialogue-scaffolding phrases
# (e.g. "Actually, ignore my earlier preference. What I need is:", "Those
# options are not quite right yet. Ask me about one specific attribute.")
# out of a turn's message before it's tokenized into the BM25/TF-IDF query,
# but ONLY on sessions where state.has_overridden() is already True --
# narrower than an earlier attempt that stripped this scaffolding on every
# turn regardless of scenario (see context.py's module docstring and
# README's "What we tried": that blanket version fixed intent_override
# HitRate@10 0.367->0.400 but regressed buying 0.713->0.663 and boundary's
# MRR enough to net-regress TechnicalScore overall, so it was reverted).
# Off by default until scripts/train_override_query_strip.py's CV run
# confirms this narrower, override-only version doesn't repeat that
# regression. See scripts/diagnose_intent_override.py's per-route recall
# breakdown for the diagnosis this targets.
OVERRIDE_QUERY_STRIP_ENABLED = os.environ.get("AGENT_SHOPPER_OVERRIDE_QUERY_STRIP", "") not in ("", "0", "false", "False")

# --- Personalization ---------------------------------------------------

PREFERENCE_TAG_BOOST = _env_float("AGENT_SHOPPER_PREF_TAG_BOOST", 0.05)
PREFERENCE_TAG_BOOST_CAP = _env_float("AGENT_SHOPPER_PREF_TAG_BOOST_CAP", 0.15)

DECISIVENESS_BY_PURCHASE_FREQUENCY = {
    "1-2 prior purchases": 0.3,
    "3-4 prior purchases": 0.5,
    "5+ prior purchases": 0.7,
}
DECISIVENESS_DEFAULT = 0.4
PICKY_RATING_STYLE_KEYWORDS = ("critical", "picky", "particular", "demanding")

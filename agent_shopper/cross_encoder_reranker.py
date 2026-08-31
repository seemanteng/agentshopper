"""Shipped default: a frozen, local, pointwise cross-encoder semantic scorer,
fused with the existing HeuristicReranker by weighted Reciprocal Rank Fusion
over a K=100-hybrid candidate union. Entirely separate from
reranker.LLMReranker (the paid-API, listwise judge) -- see this module's own
docstring section below for why the two are architecturally different, not
just alternatives.

Gated by config.FROZEN_CROSS_ENCODER_ENABLED (on by default as of the
"Deployment readiness" promotion -- see README.md). With the flag off
(AGENT_SHOPPER_FROZEN_CROSS_ENCODER=0), this module is imported (cheap -- no
torch/sentence_transformers import at module scope, see
FrozenCrossEncoderScorer._ensure_loaded) but nothing in it is ever called
from dialog_policy.process_turn, and the plain heuristic ranking is used
instead (reproducing the 0.5674 baseline).

## Frozen, not fine-tuned

Hard requirement (see README.md's pilot entry once measured): no gradient
updates, no optimizer, no LoRA/adapter/projection-head training, no ESCI
fine-tuning. FrozenCrossEncoderScorer._ensure_loaded calls `.eval()` and sets
every parameter's `requires_grad_(False)`; `.score()` runs inference under
`torch.inference_mode()`. This is inference-only, exactly like the existing
frozen `sentence-transformers/all-MiniLM-L6-v2` dense-retrieval encoder
(agent_shopper/dense_index.py, config.DENSE_MODEL_NAME) -- same "frozen
encoder is explicitly allowed, full-model training is out of scope"
competition-spec boundary (docs/competition_specification.md).

## Why this is architecturally different from the reverted LLM reranker

README.md's "What we tried" documents the LLM listwise reranker (a paid API
judging a whole candidate list jointly in one call) as a net regression on
`intent_override` specifically, root-caused to severe candidate-order
sensitivity (92.5% top-pick flip rate under reordering, holding content
fixed) -- NOT prompt injection. This pilot scores query-candidate pairs
POINTWISE and INDEPENDENTLY (no candidate ever sees another candidate's
text, and the model has no notion of "list position"), so that specific
failure mode cannot occur here by construction -- see
tests/agent_shopper/test_cross_encoder_reranker.py's order-invariance test,
and scripts/replay_cross_encoder_offline.py's empirical check against the
real model.

## Packaging for network-free judging

See docs/competition_specification.md / submission_rules.md: official
scoring may run with network disabled. config.FROZEN_CROSS_ENCODER_MODEL
defaults to a module-relative path (models/cross_encoder/ms-marco-TinyBERT-L-6/
under the repo root, resolved via Path(__file__).resolve() in config.py --
never CWD-relative) produced by scripts/prepare_cross_encoder_artifact.py,
and config.FROZEN_CROSS_ENCODER_LOCAL_FILES_ONLY defaults to True -- so a
judge's run needs zero AGENT_SHOPPER_* environment variables to load the
exact validated checkpoint fully offline. See README.md's "Deployment
readiness" subsection for the packaging method, artefact size, checksum,
and the still-unresolved organizer questions (archive-size limit, whether
checkpoint bundling/Git-LFS is accepted) that this promotion does not
resolve on its own.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from agent_shopper.config import (
    FROZEN_CROSS_ENCODER_BATCH_SIZE,
    FROZEN_CROSS_ENCODER_DEPTH,
    FROZEN_CROSS_ENCODER_LOCAL_FILES_ONLY,
    FROZEN_CROSS_ENCODER_MAX_LENGTH,
    FROZEN_CROSS_ENCODER_MODEL,
    RRF_K,
)
from agent_shopper.models import Candidate, DistilledContext, Product
from agent_shopper.reranker import HeuristicReranker

if TYPE_CHECKING:  # pragma: no cover -- typing only, never imported at runtime unless the feature is used
    from sentence_transformers import CrossEncoder as _CrossEncoderType

# Module-level weight cache: model_name_or_path -> loaded, frozen model.
# Loading a cross-encoder checkpoint costs real wall-clock time (disk/network
# I/O + weight materialization); this cache means only the FIRST turn that
# actually uses the feature pays that cost, not every FrozenCrossEncoderScorer
# instance (constructed fresh each turn, mirroring HeuristicReranker's own
# per-turn construction -- see dialog_policy.process_turn).
_MODEL_CACHE: dict[tuple[str, int, bool], object] = {}


class CrossEncoderUnavailable(RuntimeError):
    """Raised by FrozenCrossEncoderScorer on any load or inference failure --
    callers (FrozenCrossEncoderReranker) must always catch this and fall back
    to the heuristic ranking, never let it crash Agent.respond(). Mirrors
    llm_client.LLMUnavailable's contract/shape for the same reason."""

    def __init__(self, message: str, cause_type: str | None = None) -> None:
        super().__init__(message)
        self.cause_type = cause_type


def _format_product_text(product: Product) -> str:
    """Deterministic, participant-visible-fields-only product formatter for
    semantic scoring. Explicitly excludes parent_asin (no identifier leakage
    into the model input) and price/average_rating/rating_number (kept in
    the structured heuristic instead -- a cross-encoder trained on text
    relevance, not numeric constraint satisfaction, has no reliable way to
    reason about "under $50" or "4+ stars"). Dict fields (`details`) are
    sorted by key so output never depends on insertion order; list fields
    are joined in their existing (catalog) order, which is itself
    deterministic per product. Truncation is left to the model's own
    tokenizer/max_length (config.FROZEN_CROSS_ENCODER_MAX_LENGTH), not done
    here, per this pilot's design."""
    parts: list[str] = []
    title = (product.title or "").strip()
    if title:
        parts.append(f"Title: {title}")
    categories = [c.strip() for c in product.categories if c and c.strip()]
    if categories:
        parts.append(f"Category: {' > '.join(categories)}")
    features = [f.strip() for f in product.features if f and f.strip()]
    if features:
        parts.append(f"Features: {'; '.join(features)}")
    description = [d.strip() for d in product.description if d and d.strip()]
    if description:
        parts.append(f"Description: {' '.join(description)}")
    if product.details:
        detail_items = [f"{k}: {v}" for k, v in sorted(product.details.items()) if v not in (None, "")]
        if detail_items:
            parts.append(f"Details: {', '.join(detail_items)}")
    store = (product.store or "").strip()
    if store:
        parts.append(f"Brand or store: {store}")
    return "\n".join(parts)


class FrozenCrossEncoderScorer:
    """Lazy-loading, frozen, pointwise cross-encoder scorer.

    `score()` never relies on input order (each pair is scored
    independently) and never sees parent_asin, price, rating, evaluator
    metadata, or hidden identifiers -- only query_text (the same bounded,
    distilled ctx.session.query_text every other pipeline stage uses) and
    each candidate Product's participant-visible text fields, formatted by
    _format_product_text. Product text is treated as plain relevance data,
    never as instructions (mirrors reranker._LLM_SYSTEM_PROMPT's untrusted-
    data framing, even though a cross-encoder has no notion of "following
    instructions" at all -- it's a pure relevance-score regression head)."""

    def __init__(
        self,
        model_name_or_path: str = FROZEN_CROSS_ENCODER_MODEL,
        batch_size: int = FROZEN_CROSS_ENCODER_BATCH_SIZE,
        max_length: int = FROZEN_CROSS_ENCODER_MAX_LENGTH,
        local_files_only: bool = FROZEN_CROSS_ENCODER_LOCAL_FILES_ONLY,
        score_cache: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        # Optional (query_text, parent_asin) -> score memo, shared across
        # instances by a caller that passes the same dict in (see
        # scripts/replay_cross_encoder_offline.py, scripts/
        # cv_cross_encoder.py) -- cuts redundant inference when the same
        # session/turn's (query, candidate) pairs recur across alpha runs
        # that haven't diverged yet. None (the production default) disables
        # this -- production never shares a cache across turns/sessions.
        self._score_cache = score_cache
        self._model: object | None = None
        self._torch = None
        self._load_failed = False
        self._load_failure_reason: str | None = None
        self._text_cache: dict[str, str] = {}
        self.last_load_seconds: float | None = None
        self.last_score_seconds: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_failed:
            return
        key = (self.model_name_or_path, self.max_length, self.local_files_only)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            self._model = cached
            import torch  # safe: only reached once a load has already succeeded once
            self._torch = torch
            self.last_load_seconds = 0.0
            return
        if self.local_files_only:
            # Belt-and-suspenders offline enforcement: local_files_only=True
            # (passed to CrossEncoder(...) below) already blocks the
            # top-level load call from downloading, but sentence-transformers/
            # transformers can still make indirect network calls (e.g.
            # tokenizer/config revalidation) unless these are set too. This
            # is what lets the shipped default work with zero judge-provided
            # environment variables -- setdefault() never overrides a value
            # the caller/judge did set.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            self._load_failed = True
            self._load_failure_reason = f"package_missing: {exc}"
            return
        try:
            start = time.monotonic()
            model = CrossEncoder(
                self.model_name_or_path,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._model = model
            self._torch = torch
            self.last_load_seconds = time.monotonic() - start
            _MODEL_CACHE[key] = model
        except Exception as exc:  # noqa: BLE001 -- any load failure degrades to heuristic, never crashes
            self._load_failed = True
            self._load_failure_reason = f"{type(exc).__name__}: {exc}"

    def _formatted(self, product: Product) -> str:
        cached = self._text_cache.get(product.parent_asin)
        if cached is not None:
            return cached
        text = _format_product_text(product)
        self._text_cache[product.parent_asin] = text
        return text

    def score(self, query_text: str, candidates: list[Product]) -> dict[str, float]:
        """Returns {parent_asin: score}. Raises CrossEncoderUnavailable on
        any load/inference failure -- callers must catch and fall back."""
        if not candidates:
            return {}
        self._ensure_loaded()
        if self._model is None:
            raise CrossEncoderUnavailable(self._load_failure_reason or "model not loaded", cause_type="load_failed")

        to_score: list[Product] = []
        result: dict[str, float] = {}
        if self._score_cache is not None:
            for product in candidates:
                cached = self._score_cache.get((query_text, product.parent_asin))
                if cached is not None:
                    result[product.parent_asin] = cached
                else:
                    to_score.append(product)
        else:
            to_score = list(candidates)

        if to_score:
            pairs = [(query_text, self._formatted(p)) for p in to_score]
            try:
                start = time.monotonic()
                with self._torch.inference_mode():
                    raw_scores = self._model.predict(
                        pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True,
                    )
                self.last_score_seconds = time.monotonic() - start
            except Exception as exc:  # noqa: BLE001 -- inference failure degrades to heuristic, never crashes
                raise CrossEncoderUnavailable(f"inference failed: {type(exc).__name__}: {exc}", cause_type=type(exc).__name__) from exc
            for product, raw in zip(to_score, raw_scores):
                value = float(raw)
                result[product.parent_asin] = value
                if self._score_cache is not None:
                    self._score_cache[(query_text, product.parent_asin)] = value
        return result


def _rank_lookup(ranked: list[Candidate]) -> dict[str, int]:
    """1-indexed parent_asin -> rank within `ranked`'s own order."""
    return {c.product.parent_asin: i + 1 for i, c in enumerate(ranked)}


def build_candidate_union(fused_candidates: list[Candidate], heuristic_ranked_full: list[Candidate], depth: int) -> list[Candidate]:
    """U = deduplicated_union(F[:depth], H[:10]) -- preserves every
    candidate from the pre-rerank fused order's top-`depth` first, then
    appends any of the existing heuristic's own top-10 not already present.
    Guarantees every current heuristic hit's target stays a candidate in the
    set the semantic scorer sees (candidate-AVAILABILITY only -- see the
    Hybrid-Union Oracle's scope-limit note in scripts/diagnose_retrieval.py;
    this does NOT guarantee the fused ranking below keeps it in the final
    top-10). Deterministic: no randomness, no dependency on set iteration
    order (membership tracked by an ordinary `set` used only for `in`
    checks, output order comes solely from the two input lists' own order).
    Size is at most `depth` + 10 unique candidates."""
    seen: set[str] = set()
    union: list[Candidate] = []
    for c in fused_candidates[:depth]:
        asin = c.product.parent_asin
        if asin not in seen:
            seen.add(asin)
            union.append(c)
    for c in heuristic_ranked_full[:10]:
        asin = c.product.parent_asin
        if asin not in seen:
            seen.add(asin)
            union.append(c)
    return union


def fuse_hybrid_scores(
    union: list[Candidate],
    heuristic_rank: dict[str, int],
    semantic_rank: dict[str, int],
    fused_rank: dict[str, int],
    alpha: float,
    rrf_k: int = RRF_K,
) -> list[Candidate]:
    """Weighted RRF over the union candidate set:

        combined_score = (1-alpha)/(rrf_k+heuristic_rank) + alpha/(rrf_k+semantic_rank)

    Deterministic tie-break: combined score desc, heuristic_rank asc,
    fused_rank asc, parent_asin asc. Sets each returned Candidate's
    `final_score` to its combined score (matches HeuristicReranker/
    LLMReranker's own convention -- dialog_policy's response builder reads
    `c.final_score if c.final_score is not None else c.fused_score`).
    A candidate missing from `heuristic_rank` or `semantic_rank` (shouldn't
    happen -- both are computed over every union member -- but handled
    defensively) falls back to a rank of len(union)+1 so it sorts last
    within that component rather than raising a KeyError."""
    fallback_rank = len(union) + 1
    rows: list[tuple[float, int, int, str, Candidate]] = []
    for c in union:
        asin = c.product.parent_asin
        h_rank = heuristic_rank.get(asin, fallback_rank)
        s_rank = semantic_rank.get(asin, fallback_rank)
        f_rank = fused_rank.get(asin, fallback_rank)
        combined = (1.0 - alpha) / (rrf_k + h_rank) + alpha / (rrf_k + s_rank)
        rows.append((combined, h_rank, f_rank, asin, c))
    rows.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    ranked = [row[4] for row in rows]
    for combined, *_rest, c in rows:
        c.final_score = combined
    return ranked


class FrozenCrossEncoderReranker:
    """Reranker protocol implementation (see reranker.Reranker): runs the
    existing HeuristicReranker over the full candidate pool (unchanged --
    this pilot augments it, never replaces it, per the Hybrid-Union Oracle
    finding that a hard top-K-only replacement would exclude real current
    hits), builds the K=100-hybrid union, scores it with a frozen
    cross-encoder, and fuses by weighted RRF. Falls back to the plain
    heuristic ranking on any scorer failure -- never raises out of
    rerank()."""

    def __init__(
        self,
        scorer: FrozenCrossEncoderScorer | None = None,
        heuristic: HeuristicReranker | None = None,
        depth: int = FROZEN_CROSS_ENCODER_DEPTH,
        alpha: float | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        from agent_shopper.config import FROZEN_CROSS_ENCODER_ALPHA
        self.scorer = scorer or FrozenCrossEncoderScorer()
        self.heuristic = heuristic or HeuristicReranker()
        self.depth = depth
        self.alpha = FROZEN_CROSS_ENCODER_ALPHA if alpha is None else alpha
        self.rrf_k = rrf_k
        # Diagnostic-only fields, mirroring LLMReranker's last_* attributes --
        # production code never reads these; benchmark/CV scripts do.
        self.last_used_cross_encoder = False
        self.last_failure_reason: str | None = None
        self.last_load_seconds: float | None = None
        self.last_score_seconds: float | None = None
        self.last_union_size: int | None = None

    def rerank(self, ctx: DistilledContext, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        self.last_used_cross_encoder = False
        self.last_failure_reason = None
        self.last_union_size = None
        if not candidates:
            return []
        # Full heuristic ranking over the WHOLE pool (top_k=len(candidates)),
        # not just top_k -- as a side effect this also sets `final_score` on
        # every input candidate, same as a normal top_k call would (see
        # HeuristicReranker.rerank). Needed so heuristic_rank below covers
        # every candidate that could end up in the union, not only top_k.
        heuristic_full = self.heuristic.rerank(ctx, candidates, top_k=len(candidates))
        if self.alpha == 0.0:
            # alpha=0.0 must reproduce the heuristic's own top-10 exactly --
            # short-circuit rather than trust the fusion formula's algebra
            # under floating point, AND so the CV/offline-replay baseline
            # column never pays the model-load/inference cost at all.
            return heuristic_full[:top_k]

        fused_rank = _rank_lookup(candidates)  # F: pre-rerank fused order (already what retrieve() returned)
        heuristic_rank = _rank_lookup(heuristic_full)
        union = build_candidate_union(candidates, heuristic_full, self.depth)
        self.last_union_size = len(union)

        try:
            scores = self.scorer.score(ctx.session.query_text, [c.product for c in union])
            self.last_used_cross_encoder = True
            self.last_load_seconds = self.scorer.last_load_seconds
            self.last_score_seconds = self.scorer.last_score_seconds
        except CrossEncoderUnavailable as exc:
            self.last_failure_reason = str(exc)
            return heuristic_full[:top_k]
        except Exception as exc:  # noqa: BLE001 -- never let an unexpected scoring bug crash a turn
            self.last_failure_reason = f"unexpected: {type(exc).__name__}: {exc}"
            return heuristic_full[:top_k]

        # Deterministic semantic rank: score desc, parent_asin asc tie-break
        # -- never depends on `union`'s own input order (see
        # build_candidate_union's docstring and this module's order-
        # invariance test).
        semantic_ranked_asins = sorted(
            (c.product.parent_asin for c in union),
            key=lambda asin: (-scores.get(asin, float("-inf")), asin),
        )
        semantic_rank = {asin: i + 1 for i, asin in enumerate(semantic_ranked_asins)}

        ranked_union = fuse_hybrid_scores(union, heuristic_rank, semantic_rank, fused_rank, self.alpha, self.rrf_k)
        return ranked_union[:top_k]


def get_frozen_cross_encoder_reranker() -> FrozenCrossEncoderReranker:
    """Factory mirroring reranker.get_reranker()'s shape -- constructed
    fresh per turn (cheap: the underlying model is cached at module scope,
    see _MODEL_CACHE, so this never re-triggers a load after the first
    successful one)."""
    return FrozenCrossEncoderReranker()

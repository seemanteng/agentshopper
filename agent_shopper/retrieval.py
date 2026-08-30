"""Multi-route retrieval: keyword (BM25) + category (structured filter) +
vector (TF-IDF/cosine), fused by weighted Reciprocal Rank Fusion.

This module runs the routes and fuses them for whatever `effective_slots`
and `RoutePlan` it's handed -- it does not decide track weights, whether to
gate, or how to relax an over-constrained buying filter. Those are runtime
policy decisions made by orchestrator.py (Pillar III); this module is the
mechanical "Pillar I" pipeline base.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_shopper.bm25_index import BM25Index
from agent_shopper.category_index import attr_match_fraction, filter_products, rank_by_rating
from agent_shopper.catalog import Catalog
from agent_shopper.config import RRF_K
from agent_shopper.dense_index import DenseIndex
from agent_shopper.models import Candidate, RoutePlan, SlotSet
from agent_shopper.tfidf_index import TfidfIndex

ROUTE_SEARCH_LIMIT = 200


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    pool_size: int
    gated: bool


def _rrf_add(scores: dict[int, float], ranked: list[tuple[int, float]], weight: float, route: str,
             route_ranks: dict[int, dict[str, int]], route_scores: dict[int, dict[str, float]]) -> None:
    for rank, (doc_index, raw_score) in enumerate(ranked, start=1):
        scores[doc_index] = scores.get(doc_index, 0.0) + weight / (RRF_K + rank)
        route_ranks.setdefault(doc_index, {})[route] = rank
        route_scores.setdefault(doc_index, {})[route] = raw_score


def retrieve(
    catalog: Catalog,
    bm25_index: BM25Index,
    tfidf_index: TfidfIndex,
    query_text: str,
    effective_slots: SlotSet,
    plan: RoutePlan,
    limit: int = 200,
    dense_index: DenseIndex | None = None,
) -> RetrievalResult:
    scores: dict[int, float] = {}
    route_ranks: dict[int, dict[str, int]] = {}
    route_scores: dict[int, dict[str, float]] = {}

    if plan.gate_to_category:
        category_ids = filter_products(catalog, effective_slots)
        if not category_ids:
            return RetrievalResult(candidates=[], pool_size=0, gated=True)
        allowed = set(category_ids)
        category_ranked = rank_by_rating(catalog, category_ids, len(category_ids))
        _rrf_add(scores, category_ranked, plan.weights.get("category", 0.0), "category", route_ranks, route_scores)

        bm25_ranked = [(i, s) for i, s in bm25_index.search(query_text, ROUTE_SEARCH_LIMIT) if i in allowed]
        vector_ranked = [(i, s) for i, s in tfidf_index.search(query_text, ROUTE_SEARCH_LIMIT) if i in allowed]
        _rrf_add(scores, bm25_ranked, plan.weights.get("keyword", 0.0), "keyword", route_ranks, route_scores)
        _rrf_add(scores, vector_ranked, plan.weights.get("vector", 0.0), "vector", route_ranks, route_scores)
        if dense_index is not None and plan.weights.get("dense", 0.0):
            dense_ranked = [(i, s) for i, s in dense_index.search(query_text, ROUTE_SEARCH_LIMIT) if i in allowed]
            _rrf_add(scores, dense_ranked, plan.weights.get("dense", 0.0), "dense", route_ranks, route_scores)

        # Anything in the gated pool that neither ranked route touched still
        # belongs in the candidate list (it satisfies the hard filter).
        for doc_index in category_ids:
            scores.setdefault(doc_index, 0.0)
        pool_size = len(category_ids)
    else:
        bm25_ranked = bm25_index.search(query_text, ROUTE_SEARCH_LIMIT)
        vector_ranked = tfidf_index.search(query_text, ROUTE_SEARCH_LIMIT)
        _rrf_add(scores, bm25_ranked, plan.weights.get("keyword", 0.0), "keyword", route_ranks, route_scores)
        _rrf_add(scores, vector_ranked, plan.weights.get("vector", 0.0), "vector", route_ranks, route_scores)
        if dense_index is not None and plan.weights.get("dense", 0.0):
            dense_ranked = dense_index.search(query_text, ROUTE_SEARCH_LIMIT)
            _rrf_add(scores, dense_ranked, plan.weights.get("dense", 0.0), "dense", route_ranks, route_scores)

        if effective_slots.filled_slots() and scores:
            category_weight = plan.weights.get("category", 0.0)
            if category_weight:
                # Scored only over what keyword/vector already surfaced, not
                # a full catalog scan -- browsing's category signal is a
                # soft re-ranking boost on top of those routes, not an
                # independent full-catalog route (that's the buying gate's
                # job). Keeps this O(surfaced candidates), not O(catalog).
                soft_ranked = _soft_category_ranking(catalog, effective_slots, scores.keys())
                _rrf_add(scores, soft_ranked, category_weight, "category", route_ranks, route_scores)

        allowed = _hard_filter_allowed_ids(catalog, effective_slots, plan.hard_filter_slots)
        if allowed is not None:
            scores = {doc_index: s for doc_index, s in scores.items() if doc_index in allowed}
        pool_size = len(scores)

    ranked_ids = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    candidates = []
    for doc_index, fused_score in ranked_ids:
        product = catalog.products[doc_index]
        candidates.append(Candidate(
            product=product,
            fused_score=fused_score,
            route_ranks=route_ranks.get(doc_index, {}),
            route_scores=route_scores.get(doc_index, {}),
        ))
    return RetrievalResult(candidates=candidates, pool_size=pool_size, gated=plan.gate_to_category)


def _hard_filter_allowed_ids(catalog: Catalog, slots: SlotSet, hard_filter_slots: tuple[str, ...]) -> set[int] | None:
    """Intersects the fused pool with an exact AND-filter over only the
    hard-marked slots -- enforcing a stated hard constraint (e.g. a
    hard-marked budget) even on a turn that isn't gated to the category
    route at all (browsing track, or too few slots to trip the gate).
    Returns None when there's nothing to enforce (no-op for the caller),
    reusing category_index.filter_products rather than duplicating its
    matching logic."""
    if not hard_filter_slots:
        return None
    only_hard = SlotSet()
    for name in hard_filter_slots:
        value = getattr(slots, name)
        if value:
            setattr(only_hard, name, value)
    if not only_hard.filled_slots():
        return None
    return set(filter_products(catalog, only_hard))


def _soft_category_ranking(catalog: Catalog, slots: SlotSet, doc_indices) -> list[tuple[int, float]]:
    """Browsing-track category signal: rank the already-surfaced candidates
    by soft attribute-match fraction instead of hard-filtering, preserving
    cross-category diversity while staying O(surfaced candidates)."""
    scored = [(i, attr_match_fraction(catalog.products[i], slots)) for i in doc_indices]
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored

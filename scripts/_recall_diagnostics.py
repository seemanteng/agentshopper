"""Shared per-route Recall@10/50/100 computation for scripts/diagnose_retrieval.py
and scripts/diagnose_intent_override.py.

Existing diagnostics only know whether the target ASIN is anywhere in the
*fused, already-truncated* candidate pool (`retrieve()`'s `limit`-sized
output, default 200) that gets handed to the reranker. That conflates two
different questions this module separates: (1) which individual route
(keyword/BM25, vector/TF-IDF, category) actually recalls the target, and (2)
at what rank, so Hit Rate@10-style cutoffs can be checked per route rather
than only against the fused pool.

Keyword/vector ranks are computed by querying `bm25_index`/`tfidf_index`
directly (not reused from `retrieve()`'s fused-and-truncated output) so a
target that fell out of the final top-`limit` fused list still gets an
honest rank if one individual route ranked it well on its own -- exactly the
"recalled by one weak route" case the CV pipeline needs to see. This mirrors
`retrieval.retrieve()`'s own route calls (same `ROUTE_SEARCH_LIMIT`) but
never mutates or wraps agent_shopper/ -- read-only queries against the same
indexes the agent already built for this session, same technique as this
directory's other diagnose_*.py scripts.
"""

from __future__ import annotations

from agent_shopper.bm25_index import BM25Index
from agent_shopper.category_index import filter_products, rank_by_rating
from agent_shopper.catalog import Catalog
from agent_shopper.models import RoutePlan, SlotSet
from agent_shopper.tfidf_index import TfidfIndex

RECALL_CUTOFFS = (10, 50, 100)
ROUTE_NAMES = ("keyword", "vector", "dense", "category", "fused")


def _rank_of(target_doc_index: int, ranked: list[tuple[int, float]]) -> int | None:
    for rank, (doc_index, _score) in enumerate(ranked, start=1):
        if doc_index == target_doc_index:
            return rank
    return None


def build_asin_index(catalog: Catalog) -> dict[str, int]:
    """parent_asin -> doc index, built once per script run and passed into
    every route_ranks_for_target() call -- avoids an O(catalog) linear scan
    (catalog.products.index(...)) on every one of up to ~2,000 turns across
    the public set."""
    return {product.parent_asin: i for i, product in enumerate(catalog.products)}


def route_ranks_for_target(
    catalog: Catalog,
    bm25_index: BM25Index,
    tfidf_index: TfidfIndex,
    query_text: str,
    effective_slots: SlotSet,
    plan: RoutePlan,
    target_asin: str,
    fused_candidates: list,
    asin_to_index: dict[str, int],
    search_limit: int = 200,
    dense_index=None,
) -> dict[str, int | None]:
    """Returns {route_name: rank_or_None} (1-indexed; None = not recalled by
    that route within `search_limit`). `fused_candidates` is the actual
    `RetrievalResult.candidates` list already produced for this turn -- its
    own order gives the fused rank for free, no re-querying needed.
    `dense_index` is optional (None before agent_shopper.dense_index existed,
    or when the caller doesn't have one on hand) -- ranks["dense"] stays None
    in that case rather than erroring."""
    ranks: dict[str, int | None] = {"keyword": None, "vector": None, "dense": None, "category": None, "fused": None}
    target_doc_index = asin_to_index.get(target_asin)
    if target_doc_index is None:
        return ranks

    ranks["keyword"] = _rank_of(target_doc_index, bm25_index.search(query_text, search_limit))
    ranks["vector"] = _rank_of(target_doc_index, tfidf_index.search(query_text, search_limit))
    if dense_index is not None:
        ranks["dense"] = _rank_of(target_doc_index, dense_index.search(query_text, search_limit))

    if plan.gate_to_category:
        category_ids = filter_products(catalog, effective_slots)
        if target_doc_index in category_ids:
            ranks["category"] = _rank_of(target_doc_index, rank_by_rating(catalog, category_ids, len(category_ids)))
        # else: target fails the hard category/attribute filter this turn --
        # correctly None, distinct from "not queried".

    for rank, candidate in enumerate(fused_candidates, start=1):
        if candidate.product.parent_asin == target_asin:
            ranks["fused"] = rank
            break

    return ranks


def recall_flags(rank: int | None) -> dict[int, bool]:
    """{cutoff: bool} -- whether `rank` is within each of RECALL_CUTOFFS."""
    return {cutoff: rank is not None and rank <= cutoff for cutoff in RECALL_CUTOFFS}


def new_recall_accumulator() -> dict[str, dict[int, int]]:
    """{route: {cutoff: hit_count}} counters, one accumulator per report."""
    return {route: {cutoff: 0 for cutoff in RECALL_CUTOFFS} for route in ROUTE_NAMES}


def accumulate(acc: dict[str, dict[int, int]], ranks: dict[str, int | None]) -> None:
    for route, rank in ranks.items():
        flags = recall_flags(rank)
        for cutoff, hit in flags.items():
            acc[route][cutoff] += 1 if hit else 0


def report_recall_table(acc: dict[str, dict[int, int]], n: int, title: str = "") -> None:
    if title:
        print(title)
    header = f"{'route':10s}" + "".join(f"{'R@' + str(c):>8s}" for c in RECALL_CUTOFFS)
    print(header)
    print("-" * len(header))
    for route in ROUTE_NAMES:
        row = f"{route:10s}" + "".join(f"{acc[route][c] / n:8.1%}" if n else f"{'n/a':>8s}" for c in RECALL_CUTOFFS)
        print(row)
    print()

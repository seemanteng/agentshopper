"""Closes the loop attempt #3 (scripts/diagnose_conservative_injection.py)
left open: is `HeuristicReranker`'s rating-dominant weighting a landslide
or a close margin for the 26 never-retrieved sessions? README's
"Limitations" currently states "rating dominates ~9x" as an *inference*
from the weights (agent_shopper.config.HEURISTIC_RERANK_WEIGHTS) -- this
script measures it directly instead of assuming it.

For each of the 26 sessions, builds the exact same augmented candidate
pool attempt #3 used (real retrieve() candidates + up to 10 capped,
not-already-surfaced, filter-matching, rating-ranked injections -- same
`filter_products`+`rank_by_rating` mechanism, same trigger condition: only
ungated turns with filled slots), runs it through the real, unmodified
`HeuristicReranker().rerank()`, and reads off:

- the target's real final_score (HeuristicReranker sets this on every
  input candidate as a side effect, whether or not it makes top_k --
  no need to reimplement the scoring formula to get this number)
- the rank-10 cutoff candidate's final_score
- gap = rank10_score - target_score
- each of the 5 features' (bm25, vector, attr_match, rating, price_fit)
  weighted contribution to that gap, via the same `_feature_vector`
  agent_shopper.reranker.HeuristicReranker.rerank itself calls (imported
  directly, not reimplemented -- its own docstring says it exists exactly
  so a diagnostic/training script can log the same numbers the live
  reranker scores with)

Read-only, heuristic-only, one interactive pass -- no production code
touched. Outputs are written only to a gitignored .log file.

Usage:
    python3 scripts/diagnose_reranker_weight_gap.py [--catalog ...] [--dataset ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), matches this repo's other scripts

import agent_shopper.reranker as reranker_mod  # noqa: E402
import agent_shopper.retrieval as retrieval_mod  # noqa: E402
from agent_shopper.category_index import filter_products, rank_by_rating  # noqa: E402
from agent_shopper.config import HEURISTIC_RERANK_WEIGHTS  # noqa: E402
from agent_shopper.models import Candidate  # noqa: E402
from agent_shopper.reranker import HeuristicReranker, _feature_vector, _minmax_normalize, _slots_from_summary  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

CAP = 10  # same as attempt #3's chosen cap -- not re-swept here, this measures that same injection's real gap.
FEATURE_NAMES = ("bm25", "vector", "attr_match", "rating", "price_fit")


def _collect(catalog_path: str, dataset_path: str):
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    pools: dict[tuple[str, int], set[str]] = {}
    raw_calls: dict[tuple[str, int], dict] = {}
    ctx_by_turn: dict[tuple[str, int], object] = {}
    current: list[object] = [None, None]

    real_heuristic_rerank = reranker_mod.HeuristicReranker.rerank

    def recording_heuristic_rerank(self, ctx, candidates, top_k):
        session_id, turn = current
        if session_id is not None:
            pools[(session_id, turn)] = {c.product.parent_asin for c in candidates}
            ctx_by_turn[(session_id, turn)] = ctx
        return real_heuristic_rerank(self, ctx, candidates, top_k)

    real_retrieve = retrieval_mod.retrieve

    def recording_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit=200, dense_index=None):
        result = real_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit, dense_index)
        session_id, turn = current
        if session_id is not None:
            raw_calls[(session_id, turn)] = {
                "catalog": catalog, "bm25_index": bm25_index, "tfidf_index": tfidf_index,
                "query_text": query_text, "effective_slots": effective_slots, "plan": plan,
                "dense_index": dense_index, "result": result,
            }
        return result

    sessions_out: list[dict] = []

    with patch.object(reranker_mod.HeuristicReranker, "rerank", recording_heuristic_rerank), \
            patch.object(retrieval_mod, "retrieve", recording_retrieve):
        for sample in samples:
            session_id = f"diaggap_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

            hit_turn: int | None = None
            recall_turn: int | None = None
            turn_keys: list[tuple[str, int]] = []

            for turn in range(1, MAX_TURNS + 1):
                current[0], current[1] = session_id, turn
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}

                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                pool_this_turn = pools.get((session_id, turn), set())
                if override_applied:
                    turn_keys.append((session_id, turn))
                    if recall_turn is None and target in pool_this_turn:
                        recall_turn = turn
                if override_applied and target in ranked:
                    hit_turn = turn
                    break
                if turn == MAX_TURNS:
                    break
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    user_message, boundary_used = customer_reply(
                        effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                    )

            sessions_out.append({
                "sample_id": sample["sample_id"], "scenario_type": sample["scenario_type"],
                "target_asin": target, "hit_turn": hit_turn, "recall_turn": recall_turn, "turn_keys": turn_keys,
            })

    return sessions_out, raw_calls, ctx_by_turn


def _augmented_candidates(call: dict, target_asin: str) -> tuple[list[Candidate], bool] | None:
    """Same injection as scripts/diagnose_conservative_injection.py's
    _corrected_counterfactual: up to CAP not-already-surfaced,
    filter-matching, rating-ranked candidates. Returns (augmented_list,
    target_was_injected) or None if the trigger condition doesn't apply
    (gated turn, or no filled slots)."""
    catalog = call["catalog"]
    real_candidates: list[Candidate] = call["result"].candidates
    real_slots = call["effective_slots"]
    plan = call["plan"]

    if plan.gate_to_category or not real_slots.filled_slots():
        return None

    already_present = {c.product.parent_asin for c in real_candidates}
    category_ids = filter_products(catalog, real_slots)
    not_surfaced = [i for i in category_ids if catalog.products[i].parent_asin not in already_present]
    ranked = rank_by_rating(catalog, not_surfaced, min(len(not_surfaced), CAP))

    target_in_injection = any(catalog.products[i].parent_asin == target_asin for i, _ in ranked)

    augmented = list(real_candidates)
    for doc_index, _rating_score in ranked:
        product = catalog.products[doc_index]
        augmented.append(Candidate(product=product, fused_score=0.0, route_scores={}, route_ranks={}))

    return augmented, target_in_injection


def _feature_gap(ctx, augmented: list[Candidate], target_asin: str) -> dict | None:
    """Runs the real HeuristicReranker (unmodified) over `augmented`, then
    reads off the target's and the rank-10 cutoff's real final_score
    (already set as a side effect of rerank(), no reimplementation needed)
    plus a per-feature weighted-contribution breakdown for the gap between
    them, using the reranker's own _feature_vector/_minmax_normalize."""
    rr = HeuristicReranker()
    ranked = rr.rerank(ctx, list(augmented), top_k=10)
    if len(ranked) < 10:
        return None  # pool too small this turn to even have a rank-10 cutoff

    target_candidate = next((c for c in augmented if c.product.parent_asin == target_asin), None)
    if target_candidate is None or target_candidate.final_score is None:
        return None  # target wasn't in the augmented pool at all this turn

    rank10_candidate = ranked[9]
    target_score = target_candidate.final_score
    rank10_score = rank10_candidate.final_score
    gap = rank10_score - target_score

    # Recompute each candidate's raw feature vector (not just the weighted
    # sum HeuristicReranker already gave us) to attribute the gap.
    bm25_raw = {i: c.route_scores.get("keyword", 0.0) for i, c in enumerate(augmented)}
    vector_raw = {i: c.route_scores.get("vector", 0.0) for i, c in enumerate(augmented)}
    bm25_norm = _minmax_normalize(bm25_raw)
    vector_norm = _minmax_normalize(vector_raw)
    slots = _slots_from_summary(ctx.session.slot_summary)

    def features_of(candidate: Candidate) -> dict[str, float]:
        i = augmented.index(candidate)
        return _feature_vector(
            candidate.product, slots, bm25_norm.get(i, 0.0), vector_norm.get(i, 0.0), ctx.profile.rating_floor_hint,
        )

    target_features = features_of(target_candidate)
    rank10_features = features_of(rank10_candidate)
    weights = HEURISTIC_RERANK_WEIGHTS
    contributions = {
        name: weights[name] * (rank10_features[name] - target_features[name]) for name in FEATURE_NAMES
    }

    return {
        "target_score": target_score, "rank10_score": rank10_score, "gap": gap,
        "target_features": target_features, "rank10_features": rank10_features,
        "contributions": contributions,
        "dominant_feature": max(contributions, key=lambda k: contributions[k]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="diagnose_reranker_weight_gap.log")
    args = parser.parse_args()

    print("Collecting: one real interactive pass (heuristic-only)...")
    sessions, raw_calls, ctx_by_turn = _collect(str(ROOT / args.catalog), str(ROOT / args.dataset))
    never_retrieved = [s for s in sessions if s["recall_turn"] is None]
    print(f"Population check: {len(never_retrieved)} of {len(sessions)} never recalled (expect 26).")

    comparable: list[dict] = []
    not_comparable: dict[str, str] = {}
    for session in never_retrieved:
        target = session["target_asin"]
        best_for_session: dict | None = None
        reason = "no applicable turn (gated or no filled slots on every turn)"
        for (sid, turn) in session["turn_keys"]:
            call = raw_calls.get((sid, turn))
            ctx = ctx_by_turn.get((sid, turn))
            if call is None or ctx is None:
                continue
            aug = _augmented_candidates(call, target)
            if aug is None:
                continue
            augmented, target_injected = aug
            if not target_injected:
                reason = "target never entered the capped, rating-ranked injected set on any applicable turn"
                continue
            result = _feature_gap(ctx, augmented, target)
            if result is None:
                reason = "target entered the injected set but pool too small for a rank-10 cutoff, or side-effect score missing"
                continue
            result["sample_id"] = session["sample_id"]
            result["scenario_type"] = session["scenario_type"]
            result["turn"] = turn
            if best_for_session is None or result["gap"] < best_for_session["gap"]:
                best_for_session = result  # smallest (most favorable) gap this session ever achieved
        if best_for_session is not None:
            comparable.append(best_for_session)
        else:
            not_comparable[session["sample_id"]] = reason

    print(f"\nComparable (target injected + real rank-10 cutoff available): {len(comparable)} of {len(never_retrieved)}")
    print(f"Not comparable: {len(not_comparable)} -- {not_comparable}")

    if comparable:
        gaps = [r["gap"] for r in comparable]
        dominant_counts = Counter(r["dominant_feature"] for r in comparable)
        print(f"\nGap (rank10_score - target_score): mean={statistics.fmean(gaps):.4f} median={statistics.median(gaps):.4f} "
              f"min={min(gaps):.4f} max={max(gaps):.4f}")
        print(f"Dominant contributor to the gap, by session: {dict(dominant_counts)}")
        rating_dominant = dominant_counts.get("rating", 0)
        print(f"\nVERDICT: 'rating' is the single largest contributor to the gap in {rating_dominant} of {len(comparable)} "
              f"comparable sessions -- {'LANDSLIDE' if rating_dominant >= max(1, 0.7 * len(comparable)) else 'MIXED/CLOSE'}.")
        for r in comparable:
            print(f"  {r['sample_id']:14s} gap={r['gap']:+.4f} dominant={r['dominant_feature']:12s} "
                  f"contributions={ {k: round(v, 4) for k, v in r['contributions'].items()} }")

    out_path = ROOT / args.output
    out_path.write_text(json.dumps({
        "population": len(never_retrieved), "comparable_count": len(comparable),
        "not_comparable": not_comparable, "comparable": comparable,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote full report to {out_path}")


if __name__ == "__main__":
    main()

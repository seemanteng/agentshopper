"""Attempt #3 at the 26-never-retrieved-sessions gap: corrected counterfactual
FIRST, before any production code.

Lesson from attempts #1/#2 (see README.md's "What we tried"): their
validating counterfactuals measured candidate AVAILABILITY only (does the
target enter the pool, at an unrealistic depth) -- never whether the real,
rating-heavy HeuristicReranker would actually place it in the real top-10,
under the real ROUTE_SEARCH_LIMIT=200. Both looked promising on that basis
and both regressed TechnicalScore in the real pipeline.

This script fixes that: for each of the 26 never-retrieved public sessions,
it takes the REAL retrieve() candidates (ROUTE_SEARCH_LIMIT=200, unmodified)
and the REAL DistilledContext HeuristicReranker actually saw that turn
(captured via the same monkeypatch technique scripts/diagnose_retrieval.py
already uses -- read-only, never alters production behavior), injects a
small, capped (K=5 or K=10) set of NOT-already-surfaced structured-filter
matches ranked by rating, and runs the REAL, unmodified
`HeuristicReranker().rerank(ctx, augmented_candidates, top_k=10)` to check
whether the target actually lands in the top 10 -- not just whether it's
"in the pool."

Read-only: no production code is touched. Heuristic-only
(AGENT_SHOPPER_FORCE_HEURISTIC=1, AGENT_SHOPPER_FROZEN_CROSS_ENCODER=0),
one interactive pass, matching this project's own established diagnostic
convention. Outputs are written only to a gitignored .log file.

Usage:
    python3 scripts/diagnose_conservative_injection.py [--catalog ...] [--dataset ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), matches this repo's other scripts

import agent_shopper.reranker as reranker_mod  # noqa: E402
import agent_shopper.retrieval as retrieval_mod  # noqa: E402
from agent_shopper.category_index import filter_products, rank_by_rating  # noqa: E402
from agent_shopper.models import Candidate  # noqa: E402
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

# Candidate injection caps to check -- the design constraint's own stated
# range endpoints (5-10), not a swept/tuned parameter. No other values are
# tried; see the module docstring and the "DO NOT sweep" instruction this
# script follows.
CAPS_TO_CHECK = (5, 10)


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
            session_id = f"diag3_{uuid.uuid4().hex}"
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


def _corrected_counterfactual(call: dict, ctx, target_asin: str, cap: int) -> dict:
    """The actual fix for #1/#2's mistake: builds the augmented candidate
    list exactly as the proposed production change would, then runs it
    through the REAL, unmodified HeuristicReranker -- top-10 membership,
    not pool membership, is what's reported."""
    catalog = call["catalog"]
    real_candidates: list[Candidate] = call["result"].candidates
    real_slots = call["effective_slots"]
    plan = call["plan"]

    if plan.gate_to_category:
        return {"applicable": False, "reason": "gated turn -- attempt #3's trigger condition (same as #1) excludes this"}
    if not real_slots.filled_slots():
        return {"applicable": False, "reason": "no filled slots -- trigger condition excludes this"}

    already_present = {c.product.parent_asin for c in real_candidates}
    category_ids = filter_products(catalog, real_slots)
    not_surfaced = [i for i in category_ids if catalog.products[i].parent_asin not in already_present]
    ranked = rank_by_rating(catalog, not_surfaced, min(len(not_surfaced), cap))

    target_in_injection = any(catalog.products[i].parent_asin == target_asin for i, _ in ranked)

    augmented = list(real_candidates)
    for doc_index, _rating_score in ranked:
        product = catalog.products[doc_index]
        augmented.append(Candidate(product=product, fused_score=0.0, route_scores={}, route_ranks={}))

    reranked = reranker_mod.HeuristicReranker().rerank(ctx, augmented, top_k=10)
    top10_asins = [c.product.parent_asin for c in reranked]
    target_reaches_top10 = target_asin in top10_asins

    return {
        "applicable": True,
        "target_in_injected_set": target_in_injection,
        "target_reaches_top10_after_real_rerank": target_reaches_top10,
        "injected_count": len(ranked),
        "already_present_count": len(real_candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="diagnose_conservative_injection.log")
    args = parser.parse_args()

    print("Collecting: one real interactive pass (heuristic-only), capturing real ctx + real candidates...")
    sessions, raw_calls, ctx_by_turn = _collect(str(ROOT / args.catalog), str(ROOT / args.dataset))

    never_retrieved = [s for s in sessions if s["recall_turn"] is None]
    print(f"Population check: {len(never_retrieved)} of {len(sessions)} never recalled (expect 26, same definition as before).")

    results_by_cap: dict[int, dict] = {}
    for cap in CAPS_TO_CHECK:
        recovered_sessions: set[str] = set()
        session_detail: dict[str, list[dict]] = {}
        for session in never_retrieved:
            target = session["target_asin"]
            per_turn = []
            for (sid, turn) in session["turn_keys"]:
                call = raw_calls.get((sid, turn))
                ctx = ctx_by_turn.get((sid, turn))
                if call is None or ctx is None:
                    continue
                result = _corrected_counterfactual(call, ctx, target, cap)
                result["turn"] = turn
                per_turn.append(result)
                if result.get("target_reaches_top10_after_real_rerank"):
                    recovered_sessions.add(session["sample_id"])
            session_detail[session["sample_id"]] = per_turn

        results_by_cap[cap] = {
            "cap": cap,
            "corrected_recovery_count": len(recovered_sessions),
            "corrected_recovered_sample_ids": sorted(recovered_sessions),
            "session_detail": session_detail,
        }
        print(f"\nCap={cap}: corrected (real-reranker) recovery = {len(recovered_sessions)} of {len(never_retrieved)}")
        print(f"  recovered: {sorted(recovered_sessions)}")

    out_path = ROOT / args.output
    out_path.write_text(json.dumps({
        "population": {"actual": len(never_retrieved), "sample_ids": [s["sample_id"] for s in never_retrieved]},
        "results_by_cap": results_by_cap,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote full corrected-counterfactual report to {out_path}")


if __name__ == "__main__":
    main()

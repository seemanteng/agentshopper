"""Diagnostic: for each public-dev-set session, determine whether a miss is
a *retrieval* failure (the target never enters the candidate pool that
actually gets reranked) or a *ranking* failure (it's in that pool but never
makes the final top-10) -- broken down by scenario_type.

Read-only with respect to agent_shopper/: HeuristicReranker.rerank and
LLMReranker.rerank are monkeypatched only for the duration of this script's
own run, purely to record their `candidates` argument (the exact pool that
determines that turn's response) before delegating to the real
implementation unchanged -- agent behavior is byte-identical to a normal
evaluator run. Recording at the reranker entry point (rather than wrapping
retrieval.retrieve) is deliberate: process_turn can call retrieve() more
than once in a turn (the zero-pool relax loop, the stuck-budget-widen
retry), and only the *last, actually-adopted* result feeds the reranker --
wrapping retrieve() itself would risk recording an explored-but-discarded
attempt (e.g. a widened-budget retry that turned out not to help) instead
of what was really used.

Usage:
    python3 scripts/diagnose_retrieval.py [--catalog data/catalog.jsonl] [--dataset data/public_set.jsonl]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.reranker as reranker_mod  # noqa: E402
import agent_shopper.retrieval as retrieval_mod  # noqa: E402
from scripts._recall_diagnostics import (  # noqa: E402
    accumulate,
    build_asin_index,
    new_recall_accumulator,
    report_recall_table,
    route_ranks_for_target,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(ROOT / args.dataset)
    catalog_ids, categories, products = catalog_index(ROOT / args.catalog)
    agent = Agent(str(ROOT / args.catalog))

    # (session_id, turn) -> set(parent_asin) actually handed to the reranker
    # that turn -- the pool that determines that turn's `recommendations`.
    pools: dict[tuple[str, int], set[str]] = {}
    current: list[object] = [None, None]  # [session_id, turn], set right before agent.respond()

    def _record(candidates) -> None:
        session_id, turn = current
        if session_id is not None:
            pools[(session_id, turn)] = {c.product.parent_asin for c in candidates}

    real_heuristic_rerank = reranker_mod.HeuristicReranker.rerank
    real_llm_rerank = reranker_mod.LLMReranker.rerank

    def recording_heuristic_rerank(self, ctx, candidates, top_k):
        _record(candidates)
        return real_heuristic_rerank(self, ctx, candidates, top_k)

    def recording_llm_rerank(self, ctx, candidates, top_k):
        _record(candidates)
        return real_llm_rerank(self, ctx, candidates, top_k)

    # (session_id, turn) -> the LAST retrieve() call's args+result for that
    # turn -- process_turn can call retrieve() more than once (zero-pool
    # relax loop, budget-widen retry); only the final call's output is what
    # actually reached the reranker, same reasoning as this module's
    # docstring gives for recording at the reranker entry point instead of
    # wrapping retrieve() naively.
    last_retrieve_call: dict[tuple[str, int], dict] = {}
    real_retrieve = retrieval_mod.retrieve
    asin_index_cache: dict[int, dict[str, int]] = {}

    def recording_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit=200, dense_index=None):
        result = real_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit, dense_index)
        session_id, turn = current
        if session_id is not None:
            last_retrieve_call[(session_id, turn)] = {
                "catalog": catalog, "bm25_index": bm25_index, "tfidf_index": tfidf_index,
                "query_text": query_text, "effective_slots": effective_slots, "plan": plan, "result": result,
                "dense_index": dense_index,
            }
        return result

    recall_acc_by_scenario: dict[str, dict] = defaultdict(new_recall_accumulator)
    recall_n_by_scenario: dict[str, int] = defaultdict(int)

    sessions_out: list[dict] = []
    with patch.object(reranker_mod.HeuristicReranker, "rerank", recording_heuristic_rerank), \
            patch.object(reranker_mod.LLMReranker, "rerank", recording_llm_rerank), \
            patch.object(retrieval_mod, "retrieve", recording_retrieve):
        for sample in samples:
            session_id = f"diag_{uuid.uuid4().hex}"
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
                # Gated identically to the official evaluator's hit check --
                # pre-override, "recall" of the eventual target is incidental
                # (the shopper hasn't stated the override-driven preference
                # yet), so it can't count as a real recall any more than it
                # can count as a real hit.
                if override_applied and recall_turn is None and target in pool_this_turn:
                    recall_turn = turn
                if override_applied:
                    call = last_retrieve_call.get((session_id, turn))
                    if call is not None:
                        cache_key = id(call["catalog"])
                        if cache_key not in asin_index_cache:
                            asin_index_cache[cache_key] = build_asin_index(call["catalog"])
                        ranks = route_ranks_for_target(
                            call["catalog"], call["bm25_index"], call["tfidf_index"], call["query_text"],
                            call["effective_slots"], call["plan"], target, call["result"].candidates,
                            asin_index_cache[cache_key], dense_index=call["dense_index"],
                        )
                        accumulate(recall_acc_by_scenario[sample["scenario_type"]], ranks)
                        recall_n_by_scenario[sample["scenario_type"]] += 1
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
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit_turn": hit_turn,
                "recall_turn": recall_turn,
            })

    _report(sessions_out)
    print()
    for scenario in sorted(recall_acc_by_scenario):
        report_recall_table(
            recall_acc_by_scenario[scenario], recall_n_by_scenario[scenario],
            title=f"Per-route Recall@10/50/100 -- {scenario} ({recall_n_by_scenario[scenario]} turns):",
        )


def _report(sessions: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        groups[s["scenario_type"]].append(s)

    header = f"{'scenario_type':16s} {'n':>4s} {'miss':>5s} {'never_recalled':>15s} {'recalled_not_ranked':>20s} {'median_lag':>11s}"
    print(header)
    print("-" * len(header))
    for scenario in sorted(groups):
        rows = groups[scenario]
        misses = [r for r in rows if r["hit_turn"] is None]
        never_recalled = [r for r in misses if r["recall_turn"] is None]
        recalled_not_ranked = [r for r in misses if r["recall_turn"] is not None]
        lags = [r["hit_turn"] - r["recall_turn"] for r in rows if r["hit_turn"] is not None and r["recall_turn"] is not None]
        median_lag = f"{statistics.median(lags):.1f}" if lags else "n/a"
        print(
            f"{scenario:16s} {len(rows):4d} {len(misses):5d} "
            f"{len(never_recalled):15d} {len(recalled_not_ranked):20d} {median_lag:>11s}"
        )

    overall_misses = [r for r in sessions if r["hit_turn"] is None]
    never = sum(1 for r in overall_misses if r["recall_turn"] is None)
    recalled = len(overall_misses) - never
    print()
    print(
        f"Total: {len(sessions)} sessions, {len(overall_misses)} misses "
        f"({len(overall_misses) / len(sessions):.1%}) -- "
        f"{never} never recalled (retrieval failure), "
        f"{recalled} recalled but not ranked into top-10 (ranking failure)."
    )
    print(
        "\n'never_recalled' = a retrieval-side failure (query construction, "
        "slot extraction, or the buying-gate filter never surfaced the "
        "target) -- a learned/tuned reranker cannot fix these.\n"
        "'recalled_not_ranked' = the target was reachable at least once but "
        "never survived reranking/top-10 truncation -- the only bucket a "
        "reranker change (learned or hand-tuned) can plausibly help with.\n"
        "'median_lag' = among sessions that DID hit, how many turns after "
        "first becoming recallable it took to actually rank into the top-10 "
        "(0 = ranked in immediately)."
    )


if __name__ == "__main__":
    main()

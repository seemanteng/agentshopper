"""Diagnostic-only (Phases 1-5): for every public-set session whose target
was NEVER recalled by retrieval -- absent from the fused candidate pool at
EVERY turn, up to retrieval.retrieve()'s own hard depth cap of 200, which is
strictly stronger than "missed" (ranked but not top-10) -- produce a
per-turn trace, offline read-only retrieval counterfactuals, and a
root-cause classification.

Entirely read-only with respect to agent_shopper/ and evaluator/: the one
real interactive pass reuses scripts/diagnose_retrieval.py's own recording
technique (HeuristicReranker.rerank / retrieval.retrieve are monkeypatched
only for the duration of THIS script's own run, purely to observe -- never
alter -- their real inputs/outputs; agent behavior is byte-identical to a
normal evaluator run). Every Phase-3 counterfactual is a direct, OFFLINE
call into retrieval.retrieve()/bm25_index/tfidf_index/dense_index/
category_index using the REAL recorded query_text/slots/plan from that one
pass, with exactly one input deliberately changed -- never a second full
interactive re-simulation.

Run heuristic-only (AGENT_SHOPPER_FORCE_HEURISTIC=1,
AGENT_SHOPPER_FROZEN_CROSS_ENCODER=0) deliberately: retrieval (which routes
recall the target, at what rank) is entirely upstream of and unaffected by
which reranker is used downstream, and this matches scripts/
diagnose_retrieval.py's own existing baseline-reproduction assumption --
this keeps the diagnosed population decoupled from cross-encoder-specific
session-trajectory effects, and cheap enough to run several offline
counterfactual passes against.

Outputs are written ONLY to a gitignored .log file (see .gitignore's `*.log`
rule) -- never to results.json/eval_runs.jsonl. This script makes no
production code, config, weight, or ranking-logic changes; it uses target
ASINs only for offline diagnosis, never introduced into production logic.

Usage:
    python3 scripts/diagnose_never_retrieved.py [--catalog ...] [--dataset ...] [--output FILE.log]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), matches this repo's other scripts

import agent_shopper.orchestrator as orchestrator_mod  # noqa: E402
import agent_shopper.reranker as reranker_mod  # noqa: E402
import agent_shopper.retrieval as retrieval_mod  # noqa: E402
from agent_shopper.category_index import filter_products  # noqa: E402
from agent_shopper.context import _strip_boilerplate, build_query_text  # noqa: E402
from agent_shopper.models import SlotSet  # noqa: E402
from scripts._recall_diagnostics import build_asin_index, route_ranks_for_target  # noqa: E402
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

EXPECTED_NEVER_RETRIEVED = 26
ROUTE_DEPTH = 500
UNION_DEPTHS = (10, 50, 100, 200, 500)


# --- Phase 1/2: one real interactive pass, recording everything needed ------

def _collect(catalog_path: str, dataset_path: str):
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    # (session_id, turn) -> pool of ASINs actually handed to the reranker.
    pools: dict[tuple[str, int], set[str]] = {}
    # (session_id, turn) -> raw retrieve() call inputs+result (kept for every
    # turn of every session -- cheap, since catalog/index objects are shared
    # references, not copies; needed so Phase 3 can re-query offline for
    # whichever sessions turn out to be never-retrieved).
    raw_calls: dict[tuple[str, int], dict] = {}
    current: list[object] = [None, None]

    real_heuristic_rerank = reranker_mod.HeuristicReranker.rerank

    def recording_heuristic_rerank(self, ctx, candidates, top_k):
        session_id, turn = current
        if session_id is not None:
            pools[(session_id, turn)] = {c.product.parent_asin for c in candidates}
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
    asin_index_cache: dict[int, dict[str, int]] = {}

    with patch.object(reranker_mod.HeuristicReranker, "rerank", recording_heuristic_rerank), \
            patch.object(retrieval_mod, "retrieve", recording_retrieve):
        for sample in samples:
            session_id = f"diag26_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            track_guess = "buying" if sample["scenario_type"] in ("buying", "intent_override") else "browsing"
            effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

            hit_turn: int | None = None
            hit_rank: int | None = None
            recall_turn: int | None = None
            turn_keys: list[tuple[str, int]] = []
            turn_messages: dict[int, str] = {}
            turn_tracks: dict[int, str] = {}

            for turn in range(1, MAX_TURNS + 1):
                current[0], current[1] = session_id, turn
                turn_messages[turn] = user_message
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}

                state = agent.sessions.get(session_id) if hasattr(agent, "sessions") else None
                turn_tracks[turn] = getattr(state, "track", track_guess) if state is not None else track_guess

                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                pool_this_turn = pools.get((session_id, turn), set())
                if override_applied:
                    turn_keys.append((session_id, turn))
                    if recall_turn is None and target in pool_this_turn:
                        recall_turn = turn
                if override_applied and target in ranked:
                    hit_turn = turn
                    hit_rank = ranked.index(target) + 1
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
                "target_asin": target,
                "target_category_path": categories.get(target, []),
                "target_title": (products.get(target) or {}).get("title"),
                "target_details": _product_summary(products.get(target)),
                "hit_turn": hit_turn,
                "hit_rank": hit_rank,
                "recall_turn": recall_turn,
                "turn_keys": turn_keys,
                "turn_messages": turn_messages,
                "turn_tracks": turn_tracks,
                "override_events_before_recall": None,  # filled by caller for never-retrieved sessions only
            })

    return sessions_out, raw_calls, asin_index_cache, agent


def _product_summary(product: dict | None) -> dict | None:
    if product is None:
        return None
    return {
        "title": product.get("title"), "categories": product.get("categories"),
        "features": (product.get("features") or [])[:5],
        "details": product.get("details"), "store": product.get("store"),
    }


def _slotset_to_dict(slots: SlotSet) -> dict:
    out = {name: getattr(slots, name) for name in slots.filled_slots()}
    out["hard_marked"] = sorted(slots.hard_marked)
    return out


# --- Phase 3: offline, read-only retrieval counterfactuals -------------------

def _reachable(retrieve_result, target_asin: str) -> bool:
    return any(c.product.parent_asin == target_asin for c in retrieve_result.candidates)


def _turn_counterfactuals(call: dict, raw_message: str, target_asin: str, track: str, asin_to_index: dict[str, int]) -> dict:
    catalog, bm25, tfidf, dense = call["catalog"], call["bm25_index"], call["tfidf_index"], call["dense_index"]
    real_slots: SlotSet = call["effective_slots"]
    real_plan = call["plan"]
    out: dict = {}

    def run(query_text: str, slots: SlotSet, plan, limit: int = 200) -> bool:
        result = retrieval_mod.retrieve(catalog, bm25, tfidf, query_text, slots, plan, limit=limit, dense_index=dense)
        return _reachable(result, target_asin)

    out["latest_utterance_only"] = run(build_query_text(SlotSet(), raw_message), real_slots, real_plan)
    out["accumulated_query_text"] = run(call["query_text"], real_slots, real_plan)  # reference: must be False by construction
    out["accumulated_scaffolding_removed"] = run(build_query_text(real_slots, _strip_boilerplate(raw_message)), real_slots, real_plan)
    out["slots_only_query"] = run(build_query_text(real_slots, ""), real_slots, real_plan)

    filter_slot_names: list[str] = []
    if real_plan.gate_to_category:
        filter_slot_names = [n for n in real_slots.filled_slots() if n not in ("category", "feature")]
    else:
        filter_slot_names = list(real_plan.hard_filter_slots)
    for name in filter_slot_names:
        relaxed = real_slots.copy()
        setattr(relaxed, name, [] if name == "feature" else None)
        relaxed_plan, _ = orchestrator_mod.decide_routes(track, relaxed)
        out[f"hard_filter_removed:{name}"] = run(call["query_text"], relaxed, relaxed_plan)

    soft_removed = SlotSet(category=real_slots.category, hard_marked=real_slots.hard_marked)
    for name in real_slots.hard_marked:
        setattr(soft_removed, name, getattr(real_slots, name))
    soft_removed_plan, _ = orchestrator_mod.decide_routes(track, soft_removed)
    out["all_soft_constraints_removed"] = run(call["query_text"], soft_removed, soft_removed_plan)

    if real_plan.gate_to_category:
        out["category_filter_relaxed"] = run(call["query_text"], real_slots, replace(real_plan, gate_to_category=False))
    else:
        out["category_filter_relaxed"] = None

    target_doc_index = asin_to_index.get(target_asin)
    if target_doc_index is not None:
        out["route_keyword_at_500"] = any(i == target_doc_index for i, _ in bm25.search(call["query_text"], ROUTE_DEPTH))
        out["route_vector_at_500"] = any(i == target_doc_index for i, _ in tfidf.search(call["query_text"], ROUTE_DEPTH))
        out["route_dense_at_500"] = (
            any(i == target_doc_index for i, _ in dense.search(call["query_text"], ROUTE_DEPTH)) if dense is not None else None
        )
        category_ids = set(filter_products(catalog, real_slots)) if real_slots.filled_slots() else set()
        out["route_category_at_500"] = target_doc_index in category_ids
    else:
        out["route_keyword_at_500"] = out["route_vector_at_500"] = out["route_dense_at_500"] = out["route_category_at_500"] = False

    for depth in UNION_DEPTHS:
        out[f"union_all_routes_at_{depth}"] = run(call["query_text"], real_slots, real_plan, limit=depth)

    return out


# --- Phase 4: root-cause classification --------------------------------------

def _classify(session: dict, all_turn_cfs: list[dict], all_turn_infos: list[dict]) -> tuple[str, list[str], str]:
    """Classifies using evidence from EVERY recorded turn's counterfactuals
    (union), not just the last turn -- a session can only be recovered by a
    counterfactual applied on the turn where the relevant state (slots,
    query) actually existed, which for e.g. a stale hard-marked slot may be
    an earlier turn than the session's last one. Priority order below
    reflects what a real fix would target, most-generalizable first."""
    merged: dict[str, bool | None] = {}
    for cf in all_turn_cfs:
        for k, v in (cf or {}).items():
            if v is True:
                merged[k] = True
            elif k not in merged:
                merged[k] = v

    ever_gated = any(info.get("gate_to_category") for info in all_turn_infos)
    filter_recoveries = sorted({k.split(":", 1)[1] for k, v in merged.items() if k.startswith("hard_filter_removed:") and v})
    category_relaxed_recovers = merged.get("category_filter_relaxed") is True
    category_route_recovers = merged.get("route_category_at_500") is True
    text_route_recoveries = [k for k in ("route_keyword_at_500", "route_vector_at_500", "route_dense_at_500") if merged.get(k)]
    union_500 = merged.get("union_all_routes_at_500") is True
    scaffolding = merged.get("accumulated_scaffolding_removed") is True
    slots_only = merged.get("slots_only_query") is True
    latest_only = merged.get("latest_utterance_only") is True

    secondary: list[str] = []

    # Stale post-override state: a HARD-MARKED slot (an exact AND-filter,
    # never just a re-ranking nudge) that, once removed, recovers the
    # target -- on an intent_override session this is the clear signature
    # of a pre-override constraint that was never corrected/cleared.
    stale_hard_marked = [n for n in filter_recoveries if any(n in i.get("hard_marked", []) for i in all_turn_infos)]
    if session["scenario_type"] == "intent_override" and stale_hard_marked:
        primary = "stale_state_after_override"
        secondary = ["false_hard_filter_exclusion"]
        return primary, secondary, f"intent_override session; hard-marked slot(s) {stale_hard_marked} still filtering the pool -- removing them recovers the target"

    # Dominant pattern in this population: on every one of these 26
    # sessions' relevant turns, gate_to_category was False (buying track
    # never reached BUYING_GATE_MIN_SLOTS, or browsing/boundary track) --
    # meaning the category/attribute-match signal was either absent
    # entirely or only a soft RE-RANKING boost limited to whatever keyword/
    # vector already surfaced (retrieval.py's _soft_category_ranking), not
    # an independent recall route. route_category_at_500 (filter_products
    # with NO depth cap) recovering the target proves it structurally
    # matches the accumulated slots and was simply never looked for.
    if category_route_recovers:
        primary = "incorrect_category_routing"
        if text_route_recoveries:
            secondary.append("candidate_depth_truncation")
            evidence = (
                f"target satisfies the accumulated slot filter (route_category_at_500=True) but gate_to_category was "
                f"False on every recorded turn, so category was never an independent recall route; also recoverable via "
                f"{text_route_recoveries} beyond the live 200-depth cutoff"
            )
        else:
            secondary.append("vocabulary_mismatch")
            evidence = (
                "target satisfies the accumulated slot filter (route_category_at_500=True) but gate_to_category was "
                "False on every recorded turn; NO text route (keyword/vector/dense) recalls it even at depth 500, "
                "so structured attribute matching is the only viable recall path for this session"
            )
        return primary, secondary, evidence

    if ever_gated and (filter_recoveries or category_relaxed_recovers):
        primary = "false_hard_filter_exclusion"
        evidence = f"a gated turn occurred; removing filter(s) {filter_recoveries or ['the gate itself']} recovers the target"
        if text_route_recoveries:
            secondary.append("candidate_depth_truncation")
        return primary, secondary, evidence

    if not text_route_recoveries and not union_500:
        primary = "vocabulary_mismatch"
        evidence = f"no route (keyword/vector/dense/category) recalls the target even at depth {ROUTE_DEPTH}"
        return primary, secondary, evidence

    if merged.get("route_dense_at_500") and not merged.get("route_keyword_at_500") and not merged.get("route_vector_at_500"):
        primary = "vocabulary_mismatch"
        secondary.append("dense_semantic_recovery_only")
        evidence = "only the dense route reaches the target at depth 500 -- keyword/vector both fail lexically; dense alone cannot rescue it within the live depth cap"
        return primary, secondary, evidence

    if scaffolding and not slots_only and not latest_only:
        primary = "accumulated_query_dilution"
        evidence = "stripping evaluator scaffolding recovers the target, but neither a slots-only nor a latest-utterance-only query does"
        return primary, secondary, evidence

    primary = "candidate_depth_truncation" if union_500 else "other"
    evidence = (
        "target enters the real fused union only beyond the live depth cap, within depth 500"
        if union_500 else
        "no counterfactual in this battery recovers the target and no single dominant pattern matched -- see raw trace"
    )
    return primary, secondary, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="diagnose_never_retrieved.log")
    args = parser.parse_args()

    print("Phase 1/2: one real interactive pass (heuristic-only), recording per-turn retrieval detail...")
    sessions, raw_calls, asin_index_cache, agent = _collect(str(ROOT / args.catalog), str(ROOT / args.dataset))

    never_retrieved = [s for s in sessions if s["recall_turn"] is None]
    print(f"\nPhase 1 result: {len(never_retrieved)} of {len(sessions)} sessions never recalled the target "
          f"(absent from the fused pool, up to depth 200, at every turn).")
    if len(never_retrieved) != EXPECTED_NEVER_RETRIEVED:
        print(f"COUNT MISMATCH vs expected {EXPECTED_NEVER_RETRIEVED} -- see summary below for a full accounting.")

    session_rows: list[dict] = []
    cause_counter: Counter = Counter()
    secondary_counter: Counter = Counter()
    cf_recovery_counter: Counter = Counter()
    cf_recovery_ids: dict[str, list[str]] = defaultdict(list)

    for session in never_retrieved:
        target = session["target_asin"]
        turn_traces = []
        session_recovered_by: set[str] = set()
        all_turn_cfs: list[dict] = []
        all_turn_infos: list[dict] = []

        for (sid, turn) in session["turn_keys"]:
            call = raw_calls.get((sid, turn))
            if call is None:
                continue
            cache_key = id(call["catalog"])
            if cache_key not in asin_index_cache:
                asin_index_cache[cache_key] = build_asin_index(call["catalog"])
            asin_to_index = asin_index_cache[cache_key]

            ranks = route_ranks_for_target(
                call["catalog"], call["bm25_index"], call["tfidf_index"], call["query_text"],
                call["effective_slots"], call["plan"], target, call["result"].candidates,
                asin_to_index, search_limit=ROUTE_DEPTH, dense_index=call["dense_index"],
            )
            track = session["turn_tracks"].get(turn, "buying")
            cf = _turn_counterfactuals(call, session["turn_messages"].get(turn, ""), target, track, asin_to_index)

            first_unreachable_stage = _first_unreachable_stage(call, ranks)

            turn_traces.append({
                "turn": turn,
                "user_utterance": session["turn_messages"].get(turn, ""),
                "query_text": call["query_text"],
                "slots": _slotset_to_dict(call["effective_slots"]),
                "gate_to_category": call["plan"].gate_to_category,
                "hard_filter_slots": list(call["plan"].hard_filter_slots),
                "route_ranks": ranks,
                "fused_ranks_at_depths": {d: ranks.get("fused") if ranks.get("fused") and ranks["fused"] <= d else None for d in (10, 50, 100, 200)},
                "pool_size_before_filter": call["result"].pool_size,
                "candidate_pool_after": len(call["result"].candidates),
                "first_unreachable_stage": first_unreachable_stage,
                "counterfactuals": cf,
            })
            all_turn_cfs.append(cf)
            all_turn_infos.append({
                "gate_to_category": call["plan"].gate_to_category,
                "hard_marked": sorted(call["effective_slots"].hard_marked),
            })
            for name, recovered in cf.items():
                if recovered is True:
                    session_recovered_by.add(name)

        for name in session_recovered_by:
            cf_recovery_counter[name] += 1
            cf_recovery_ids[name].append(session["sample_id"])

        primary, secondary, evidence = _classify(session, all_turn_cfs, all_turn_infos)
        cause_counter[primary] += 1
        for s in secondary:
            secondary_counter[s] += 1

        session_rows.append({
            "sample_id": session["sample_id"], "scenario_type": session["scenario_type"],
            "target_asin": target, "target_title": session["target_title"],
            "target_category_path": session["target_category_path"],
            "primary_cause": primary, "secondary_causes": secondary, "evidence": evidence,
            "recovered_by_counterfactuals": sorted(session_recovered_by),
            "turns": turn_traces,
        })

    report = {
        "population": {
            "expected": EXPECTED_NEVER_RETRIEVED, "actual": len(never_retrieved),
            "sample_ids": [s["sample_id"] for s in never_retrieved],
        },
        "root_cause_counts": dict(cause_counter),
        "secondary_cause_counts": dict(secondary_counter),
        "counterfactual_recovery_counts": dict(cf_recovery_counter),
        "counterfactual_recovery_sample_ids": dict(cf_recovery_ids),
        "sessions": session_rows,
    }

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote full diagnostic report to {out_path} ({len(never_retrieved)} sessions).")
    print("\nRoot-cause counts:")
    for cause, n in cause_counter.most_common():
        print(f"  {cause:32s} {n}")
    print("\nCounterfactual recovery counts (sessions where >=1 turn's counterfactual reached the target):")
    for name, n in sorted(cf_recovery_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {n:3d}  {cf_recovery_ids[name]}")


def _first_unreachable_stage(call: dict, ranks: dict) -> str:
    """The earliest pipeline stage at which the target became unreachable
    this turn, in pipeline order: filter -> route -> fused-depth-cap."""
    plan = call["plan"]
    if plan.gate_to_category and ranks.get("category") is None:
        # Was it excluded by the hard filter, or just never scored by it
        # (filter never ran because no slots were filled)?
        return "removed_by_category_filter"
    if plan.hard_filter_slots and ranks.get("fused") is None and any(
        ranks.get(r) is not None for r in ("keyword", "vector", "dense")
    ):
        return "removed_by_hard_filter_post_route"
    if ranks.get("keyword") is None and ranks.get("vector") is None and ranks.get("dense") is None and ranks.get("category") is None:
        return "absent_from_every_individual_route"
    if ranks.get("fused") is None:
        return "present_in_a_route_but_not_in_fused_pool"
    return "present_in_fused_pool_but_never_passed_to_reranker"  # shouldn't normally occur for a never-recalled session's recorded turn


if __name__ == "__main__":
    main()

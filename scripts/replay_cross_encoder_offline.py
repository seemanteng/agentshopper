"""Phase 7 offline replay gate for the frozen cross-encoder pilot
(agent_shopper.cross_encoder_reranker) -- the cheap filter to run BEFORE any
expensive interactive 5-fold CV (scripts/cv_cross_encoder.py).

Runs the agent ONCE over the public dev set with the shipped, deterministic
heuristic-only behavior (FROZEN_CROSS_ENCODER_ENABLED off, exactly
reproducing 'post-revert-confirm'), recording per scored turn: the pre-
rerank fused order (F), the full heuristic ranking (H, recovered from the
`final_score` HeuristicReranker.rerank already sets on every input candidate
as a side effect -- no change to the real per-turn top_k=10 call needed),
the query_text, target, and scenario_type. For every turn where the target
is in the K=100-hybrid union U = build_candidate_union(F, H, depth=100), the
union is scored ONCE with the real frozen cross-encoder (regardless of how
many alphas are evaluated -- scores don't depend on alpha, only the RRF
fusion weight does), then every fixed ALPHAS value is evaluated offline from
that one score set.

This is explicitly a PRELIMINARY filter, not a substitute for Phase 8: it
replays a single fixed (real, shipped) trajectory, so it can't see how a
different alpha's different top-10 would change shown-item history or which
turn a session ends on -- see cross_encoder_reranker.py's module docstring
and this project's own README for why the actual interactive loop is what
Phase 8 runs.

Usage:
    python3 scripts/replay_cross_encoder_offline.py [--catalog data/catalog.jsonl] [--dataset data/public_set.jsonl] [--model MODEL_NAME_OR_PATH]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
import agent_shopper.reranker as reranker_mod  # noqa: E402
from agent_shopper.cross_encoder_reranker import (  # noqa: E402
    FrozenCrossEncoderScorer,
    build_candidate_union,
    fuse_hybrid_scores,
)
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

ALPHAS = (0.0, 0.15, 0.30, 0.50)
DEPTH = 100
RRF_K = 60
ORDER_INVARIANCE_SAMPLE = 20
ORDER_INVARIANCE_TOL = 1e-4


def _rank_lookup(ranked: list[Candidate]) -> dict[str, int]:
    return {c.product.parent_asin: i + 1 for i, c in enumerate(ranked)}


def _collect_turns(catalog_path: str, dataset_path: str) -> list[dict]:
    """One deterministic pass over the public set, heuristic-only (matches
    'post-revert-confirm' exactly -- see run_local_eval.py's own baseline
    convention), recording every scored turn's (F, H_full, query_text,
    target, scenario_type)."""
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    turns_out: list[dict] = []
    current: list[object] = [None, None, None, None]  # [sample_id, scenario_type, target, turn]

    real_rerank = reranker_mod.HeuristicReranker.rerank

    def recording_rerank(self, ctx, candidates, top_k):
        result = real_rerank(self, ctx, candidates, top_k)
        sample_id, scenario_type, target, turn = current
        if sample_id is not None and candidates:
            # `candidates` is the pre-rerank fused order F, untouched -- real_rerank
            # sets .final_score on every element as a side effect regardless of top_k.
            fused = list(candidates)
            heuristic_full = sorted(fused, key=lambda c: c.final_score, reverse=True)
            turns_out.append({
                "sample_id": sample_id, "scenario_type": scenario_type, "target": target, "turn": turn,
                "query_text": ctx.session.query_text, "fused": fused, "heuristic_full": heuristic_full,
            })
        return result

    with patch.object(dialog_policy_mod, "active_provider", return_value=None), \
            patch.object(reranker_mod.HeuristicReranker, "rerank", recording_rerank):
        for sample in samples:
            session_id = f"replay_{sample['sample_id']}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

            for turn in range(1, MAX_TURNS + 1):
                # Only record turns the official protocol would ever score as a
                # potential hit for this sample -- pre-override intent_override
                # turns can't convert (docs/competition_specification.md), same
                # gating scripts/diagnose_retrieval.py and scripts/
                # diagnose_intent_override.py already use.
                current[0] = sample["sample_id"] if override_applied else None
                current[1], current[2], current[3] = sample["scenario_type"], target, turn
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                if override_applied and target in ranked:
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
    return turns_out


def _eligible(turns: list[dict]) -> list[dict]:
    """Turns where the target enters the K=100-hybrid union -- the only
    turns any alpha could possibly affect (matches the Hybrid-Union Oracle's
    own reachability condition, scripts/diagnose_retrieval.py)."""
    out = []
    for t in turns:
        union = build_candidate_union(t["fused"], t["heuristic_full"], DEPTH)
        if any(c.product.parent_asin == t["target"] for c in union):
            t2 = dict(t)
            t2["union"] = union
            out.append(t2)
    return out


def _score_and_evaluate(eligible: list[dict], scorer: FrozenCrossEncoderScorer) -> tuple[list[dict], list[float]]:
    """Scores each eligible turn's union ONCE with the real model, then
    evaluates every ALPHAS value from that one score set. Returns
    (per-turn-per-alpha result rows, per-turn scoring latencies)."""
    rows: list[dict] = []
    latencies: list[float] = []
    for t in eligible:
        union = t["union"]
        fused_rank = _rank_lookup(t["fused"])
        heuristic_rank = _rank_lookup(t["heuristic_full"])
        products = [c.product for c in union]
        start = time.monotonic()
        scores = scorer.score(t["query_text"], products)
        latencies.append(time.monotonic() - start)
        semantic_ranked_asins = sorted(
            (c.product.parent_asin for c in union),
            key=lambda asin: (-scores.get(asin, float("-inf")), asin),
        )
        semantic_rank = {asin: i + 1 for i, asin in enumerate(semantic_ranked_asins)}

        for alpha in ALPHAS:
            ranked = fuse_hybrid_scores(union, heuristic_rank, semantic_rank, fused_rank, alpha, RRF_K)
            target_rank = next(i + 1 for i, c in enumerate(ranked) if c.product.parent_asin == t["target"])
            rows.append({
                "sample_id": t["sample_id"], "scenario_type": t["scenario_type"], "turn": t["turn"], "alpha": alpha,
                "baseline_heuristic_rank": heuristic_rank[t["target"]],
                "fused_rank": fused_rank[t["target"]],
                "semantic_rank": semantic_rank[t["target"]],
                "target_rank": target_rank,
                "top10": target_rank <= 10,
                "reciprocal_rank": 1.0 / target_rank,
            })
    return rows, latencies


def _check_order_invariance(eligible: list[dict], scorer: FrozenCrossEncoderScorer, sample_size: int) -> tuple[bool, int]:
    """Real-model empirical check (unit tests already prove this structurally
    with a fake scorer -- this proves it against the actual checkpoint on a
    bounded sample, matching this project's llm_rerank_diagnostics.py
    --position-bias precedent of spot-checking real model behavior, not just
    code logic)."""
    import random
    sample = eligible[:sample_size]
    mismatches = 0
    for t in sample:
        union = t["union"]
        products = [c.product for c in union]
        scores_a = scorer.score(t["query_text"], products)
        shuffled = list(products)
        random.Random(0).shuffle(shuffled)
        scores_b = scorer.score(t["query_text"], shuffled)
        for asin in scores_a:
            if abs(scores_a[asin] - scores_b.get(asin, float("nan"))) > ORDER_INVARIANCE_TOL:
                mismatches += 1
                break
    return mismatches == 0, len(sample)


def _report(rows: list[dict], latencies: list[float], eligible_n: int, total_turns: int, invariance_ok: bool, invariance_n: int) -> dict:
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_alpha[r["alpha"]].append(r)

    baseline_top10 = {(r["sample_id"], r["turn"]): r["top10"] for r in by_alpha[0.0]}
    baseline_rr = {(r["sample_id"], r["turn"]): r["reciprocal_rank"] for r in by_alpha[0.0]}

    print(f"Eligible target-containing turns: {eligible_n} of {total_turns} scored turns.")
    if latencies:
        sorted_lat = sorted(latencies)

        def pct(p):
            idx = min(len(sorted_lat) - 1, int(p * len(sorted_lat)))
            return sorted_lat[idx]

        print(
            f"Semantic scoring latency per turn (union size varies, up to {DEPTH + 10}): "
            f"mean={statistics.fmean(latencies):.3f}s p50={pct(0.5):.3f}s p95={pct(0.95):.3f}s "
            f"min={sorted_lat[0]:.3f}s max={sorted_lat[-1]:.3f}s"
        )
    print(f"Candidate-order invariance spot check ({invariance_n} turns, real model): {'PASS' if invariance_ok else 'FAIL'}")
    print()

    header = (
        f"{'alpha':>6s} {'n':>5s} {'top10_gain':>10s} {'top10_loss':>10s} {'net':>5s} "
        f"{'mean_RR_delta':>14s} {'median_rank_delta':>18s}"
    )
    print(header)
    print("-" * len(header))
    alpha_summary: dict[float, dict] = {}
    for alpha in ALPHAS:
        alpha_rows = by_alpha[alpha]
        gained = sum(1 for r in alpha_rows if r["top10"] and not baseline_top10[(r["sample_id"], r["turn"])])
        lost = sum(1 for r in alpha_rows if not r["top10"] and baseline_top10[(r["sample_id"], r["turn"])])
        net = gained - lost
        rr_deltas = [r["reciprocal_rank"] - baseline_rr[(r["sample_id"], r["turn"])] for r in alpha_rows]
        rank_deltas = [r["baseline_heuristic_rank"] - r["target_rank"] for r in alpha_rows]  # positive = improved (lower rank)
        mean_rr_delta = statistics.fmean(rr_deltas) if rr_deltas else 0.0
        median_rank_delta = statistics.median(rank_deltas) if rank_deltas else 0.0
        alpha_summary[alpha] = {
            "n": len(alpha_rows), "gained": gained, "lost": lost, "net": net,
            "mean_rr_delta": mean_rr_delta, "median_rank_delta": median_rank_delta,
        }
        print(
            f"{alpha:>6.2f} {len(alpha_rows):>5d} {gained:>10d} {lost:>10d} {net:>+5d} "
            f"{mean_rr_delta:>+14.4f} {median_rank_delta:>+18.1f}"
        )
    print()

    print("By scenario_type:")
    scenarios = sorted({r["scenario_type"] for r in rows})
    for alpha in ALPHAS:
        if alpha == 0.0:
            continue
        print(f"  alpha={alpha}:")
        for scenario in scenarios:
            alpha_rows = [r for r in by_alpha[alpha] if r["scenario_type"] == scenario]
            if not alpha_rows:
                print(f"    {scenario:16s} n=0")
                continue
            gained = sum(1 for r in alpha_rows if r["top10"] and not baseline_top10[(r["sample_id"], r["turn"])])
            lost = sum(1 for r in alpha_rows if not r["top10"] and baseline_top10[(r["sample_id"], r["turn"])])
            print(f"    {scenario:16s} n={len(alpha_rows):3d}  top10 gain={gained:+d} loss={lost:+d} net={gained - lost:+d}")
    print()

    return alpha_summary


def decide_gate(alpha_summary: dict[float, dict], latencies: list[float], invariance_ok: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    nonzero = {a: s for a, s in alpha_summary.items() if a != 0.0}
    any_positive_net = any(s["net"] > 0 for s in nonzero.values())
    if not any_positive_net:
        reasons.append("No nonzero alpha has positive net top-10 change.")
    all_reduce_rr = all(s["mean_rr_delta"] < 0 for s in nonzero.values())
    if all_reduce_rr:
        reasons.append("Every nonzero alpha reduces mean reciprocal rank.")
    if not invariance_ok:
        reasons.append("Candidate-order invariance check failed against the real model.")
    if latencies:
        p95 = sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        if p95 > 5.0:
            reasons.append(f"p95 scoring latency ({p95:.2f}s/turn) looks operationally infeasible (>5s/turn threshold).")
    passed = len(reasons) == 0
    return passed, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model", default=None, help="Override AGENT_SHOPPER_CROSS_ENCODER_MODEL for this run only.")
    args = parser.parse_args()

    catalog_path = str(ROOT / args.catalog)
    dataset_path = str(ROOT / args.dataset)

    print("Collecting turns (single deterministic heuristic-only pass)...")
    start = time.time()
    turns = _collect_turns(catalog_path, dataset_path)
    print(f"  {len(turns)} scored turns collected in {time.time() - start:.1f}s")

    eligible = _eligible(turns)
    print(f"  {len(eligible)} of {len(turns)} turns have the target within the K={DEPTH} hybrid union")

    if not eligible:
        print("\nNo eligible turns -- offline gate FAILS (nothing for any alpha to possibly improve).")
        return

    scorer_kwargs = {}
    if args.model:
        scorer_kwargs["model_name_or_path"] = args.model
    scorer = FrozenCrossEncoderScorer(**scorer_kwargs)

    print("\nScoring eligible turns' unions with the frozen cross-encoder...")
    start = time.time()
    rows, latencies = _score_and_evaluate(eligible, scorer)
    print(f"  scored {len(eligible)} turns in {time.time() - start:.1f}s (model load included in the first call)")

    invariance_ok, invariance_n = _check_order_invariance(eligible, scorer, ORDER_INVARIANCE_SAMPLE)

    print()
    alpha_summary = _report(rows, latencies, len(eligible), len(turns), invariance_ok, invariance_n)

    passed, reasons = decide_gate(alpha_summary, latencies, invariance_ok)
    print("=" * 100)
    if passed:
        print("OFFLINE GATE: PASS -- proceed to scripts/cv_cross_encoder.py (Phase 8).")
    else:
        print("OFFLINE GATE: FAIL -- do not proceed to interactive integration. Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
        print("Leave the production feature disabled; document this as a negative result.")
    print("=" * 100)


if __name__ == "__main__":
    main()

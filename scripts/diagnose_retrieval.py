"""Diagnostic: for each public-dev-set session, determine whether a miss is
a *retrieval* failure (the target never enters the candidate pool that
actually gets reranked) or a *ranking* failure (it's in that pool but never
makes the final top-10) -- broken down by scenario_type.

Also reports two "oracle ceiling" analyses, at each of ORACLE_DEPTHS
(K=20/50/100/200), that separate how much upside is available from a better
*reranker* (bounded by how deep the target is ever recallable in the fused
pool) versus a better *retriever* (bounded by whether it's recallable at
all) -- and, within the reranker question, distinguishes two different
hypothetical changes:

- REPLACEMENT ORACLE: a perfect reranker that *replaces* the current
  full-depth heuristic and only ever sees the fused top-K. This can both
  gain misses (target enters top-K) AND lose existing hits (target was
  ranked into today's top-10 by the heuristic from beyond position K in the
  raw fused order -- the heuristic isn't depth-limited today, so this is a
  real, not hypothetical, risk of truncating its input).
- HYBRID-UNION ORACLE: a perfect scorer that sees fused top-K *unioned with*
  today's actual top-10, i.e. a candidate set every current hit's target is
  guaranteed to appear in, and misses whose target enters fused top-K are
  additionally recovered. This is the ceiling for "add a semantic score
  alongside the existing heuristic" rather than "replace it".

  SCOPE LIMIT, easy to over-read: "guaranteed" here means candidate
  AVAILABILITY only. Every current hit's target is guaranteed to be IN the
  set a hybrid pipeline would score, under a perfect-oracle assumption a
  real scorer doesn't satisfy. A real semantic scorer can still rank an
  available candidate below position 10 and turn a current hit into a real
  miss -- this report only checks presence, not scoring behavior, so it
  cannot detect or bound that. Any real pilot MUST measure hit-to-miss
  regressions empirically (offline replay against recorded candidate lists,
  then a session-level before/after diff), exactly as this project's own
  convention already requires for every other change (README.md's "What we
  tried").

See _oracle_reports' docstring for the exact metric formulas and
accounting identities, and README.md's "What we tried" (the reverted LLM
listwise reranker) for why this project checks the ceiling before building
a reranker upgrade, not after.

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

# Candidate-pool depths the oracle analyses assume a perfect reranker (or
# scorer, for the hybrid case) can see. 20 matches RERANK_CANDIDATE_LIMIT
# (config.py) -- today's actual LLM-rerank depth; 50/100 ask how much
# *more* upside a deeper-searching reranker could unlock. 200 is
# retrieval.retrieve()'s own hard `limit` cap (its fused pool is always
# `sorted(...)[:limit]`) -- the absolute ranking-side ceiling, since no
# candidate the reranker could ever see lies beyond it. Included for
# completeness only, NOT as a practical pilot target: scoring 200
# candidates/turn is very likely too expensive for real-time inference (see
# PRACTICAL_PILOT_DEPTH below, which the recommendation section uses
# instead). A provable invariant falls out of this: at depth 200, a session
# can never be "excluded" by REPLACEMENT ORACLE (see _oracle_reports'
# assertions) -- every current hit's target is, by construction, already
# somewhere in that same capped pool, so REPLACEMENT and HYBRID-UNION must
# coincide exactly there.
ORACLE_DEPTHS = (20, 50, 100, 200)

# The deepest depth actually worth recommending as a real pilot -- used only
# by _print_recommendation's framing, so the depth-200 ceiling (which no one
# would run a cross-encoder at) doesn't get presented as "the next
# experiment" just because it has the largest recoverable-miss count.
PRACTICAL_PILOT_DEPTH = 100

# Expected reproduction of the shipped baseline (README.md / eval_runs.jsonl's
# "post-revert-confirm" run) -- only enforced as a hard mismatch-visible
# check when run against the default public_set.jsonl; see
# _current_baseline_metrics' caller in main().
_EXPECTED_BASELINE_N = 200
_EXPECTED_BASELINE_HITS = 135
_EXPECTED_BASELINE_HR = 0.675
_BASELINE_TOL = 1e-9

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
            # NOTE on scope, re: "record every scored turn, not only inside
            # `if override_applied`": for buying/browsing/boundary this is
            # already true -- override_applied is True from turn 1 below, so
            # every turn is recorded for those three scenario_types. For
            # intent_override it stays False until the override turn fires,
            # deliberately -- this isn't a gap in instrumentation, it mirrors
            # the official rule (docs/competition_specification.md's Session
            # Protocol: "An Intent Override session cannot convert before the
            # new intent is sent"). Recording pre-override "recall" of a
            # target the shopper hasn't described yet would inflate this
            # oracle's intent_override ceiling with turns that could never
            # count as a real hit in the official evaluator either -- kept
            # gated identically to hit_turn/recall_turn below for that reason.
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
            hit_turn: int | None = None
            hit_rank: int | None = None
            recall_turn: int | None = None
            best_fused_rank: int | None = None
            # First turn the target's fused-pool rank crosses each oracle
            # depth -- None until it does. A real hit implies the target was
            # top-10 (hence top-K for every K >= 10) in the reranker's own
            # input pool at hit_turn or earlier, so these end up <= hit_turn
            # whenever both are set -- but are NOT guaranteed to be set at
            # all for a real hit (see REPLACEMENT ORACLE's docstring: the
            # heuristic reranker isn't depth-limited, so it can promote a
            # candidate from beyond position K in the raw fused order).
            earliest_fused_turn_at_depth: dict[int, int | None] = {d: None for d in ORACLE_DEPTHS}

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
                        fused_rank = ranks["fused"]  # 1-indexed; None if target absent from the fused pool this turn.
                        if fused_rank is not None:
                            if best_fused_rank is None or fused_rank < best_fused_rank:
                                best_fused_rank = fused_rank
                            for d in ORACLE_DEPTHS:
                                if fused_rank <= d and earliest_fused_turn_at_depth[d] is None:
                                    earliest_fused_turn_at_depth[d] = turn
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
                "hit_turn": hit_turn,
                "hit_rank": hit_rank,
                "recall_turn": recall_turn,
                "best_fused_rank": best_fused_rank,
                "earliest_fused_turn_at_depth": dict(earliest_fused_turn_at_depth),
            })

    _report(sessions_out)
    print()
    for scenario in sorted(recall_acc_by_scenario):
        report_recall_table(
            recall_acc_by_scenario[scenario], recall_n_by_scenario[scenario],
            title=f"Per-route Recall@10/50/100 -- {scenario} ({recall_n_by_scenario[scenario]} turns):",
        )
    print()
    is_default_dataset = Path(args.dataset).name == "public_set.jsonl"
    _oracle_reports(sessions_out, check_baseline_reproduction=is_default_dataset)


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


# --- Oracle ceiling analyses -------------------------------------------------


def _min_non_null(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _metrics_from_turns(rows: list[dict], turn_of, rank_of=None) -> dict[str, float]:
    """Recomputes the exact official metric formulas (docs/
    competition_specification.md's Metrics section) from an arbitrary
    per-row "reachable turn" function. `turn_of(row)` returns the turn a row
    counts as a hit on, or None for a miss. `rank_of(row)` returns the rank
    to use for that hit's MRR contribution (defaults to always 1 -- the
    oracle assumption of a perfect reranker placing the target first;
    pass the row's real `hit_rank` for the actual, non-oracle baseline)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "hits": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": 11.0, "efficiency": 0.0, "technical_score": 0.0}
    hits = 0
    mrr_sum = 0.0
    mttc_sum = 0.0
    for r in rows:
        turn = turn_of(r)
        if turn is not None:
            hits += 1
            rank = rank_of(r) if rank_of is not None else 1
            mrr_sum += 1.0 / rank
            mttc_sum += turn
        else:
            mttc_sum += 11
    hit_rate = hits / n
    mrr = mrr_sum / n
    mttc = mttc_sum / n
    efficiency = min(max((11 - mttc) / 10, 0.0), 1.0)
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {"n": n, "hits": hits, "hit_rate_at_10": hit_rate, "mrr": mrr, "mttc": mttc, "efficiency": efficiency, "technical_score": technical_score}


def _current_baseline_metrics(rows: list[dict]) -> dict[str, float]:
    """The REAL (non-oracle) baseline, computed directly from this run's own
    sessions -- not read from eval_runs.jsonl, which may reflect a different
    experiment. Uses each hit's actual rank (hit_rank), not an assumed 1."""
    return _metrics_from_turns(rows, turn_of=lambda r: r["hit_turn"], rank_of=lambda r: r["hit_rank"])


def _depth_accounting(rows: list[dict], depth: int) -> dict[str, int]:
    """Per-depth partition of `rows` into current hit/miss x fused-top-depth
    reachable/not -- the shared bookkeeping both oracle tables and Step 7's
    invariants are built from."""
    current_hits = [r for r in rows if r["hit_turn"] is not None]
    current_misses = [r for r in rows if r["hit_turn"] is None]

    def reachable(r: dict) -> bool:
        return r["earliest_fused_turn_at_depth"][depth] is not None

    recoverable_misses = [r for r in current_misses if reachable(r)]
    unrecoverable_misses = [r for r in current_misses if not reachable(r)]
    hits_retained = [r for r in current_hits if reachable(r)]
    hits_excluded = [r for r in current_hits if not reachable(r)]
    return {
        "current_hits": len(current_hits),
        "current_misses": len(current_misses),
        "recoverable_misses": len(recoverable_misses),
        "unrecoverable_misses": len(unrecoverable_misses),
        "hits_retained": len(hits_retained),
        "hits_excluded": len(hits_excluded),
        "net_hit_change": len(recoverable_misses) - len(hits_excluded),
        "replacement_hits": len(hits_retained) + len(recoverable_misses),
        "hybrid_hits": len(current_hits) + len(recoverable_misses),
    }


def _replacement_metrics(rows: list[dict], depth: int) -> dict[str, float]:
    """REPLACEMENT ORACLE at `depth`: a session counts as a hit iff its
    target ever entered the fused top-`depth` pool, REGARDLESS of whether
    today's real (undepth-limited) heuristic already hit it -- modeling a
    reranker that fully replaces the heuristic and only sees the top-`depth`
    fused candidates."""
    return _metrics_from_turns(rows, turn_of=lambda r: r["earliest_fused_turn_at_depth"][depth])


def _hybrid_metrics(rows: list[dict], depth: int) -> dict[str, float]:
    """HYBRID-UNION ORACLE at `depth`: a session counts as a hit iff EITHER
    it's already a real hit today OR its target entered fused top-`depth` --
    modeling a scorer added alongside (not replacing) the current heuristic.
    Every current hit's target is guaranteed to be a CANDIDATE in that
    scorer's input by construction -- this is NOT a guarantee that a real
    scorer would keep ranking it top-10; see the module docstring's SCOPE
    LIMIT note. Only a perfect oracle (rank=1 whenever present, which is
    what this function assumes) can never lose an existing hit."""
    return _metrics_from_turns(
        rows, turn_of=lambda r: _min_non_null(r["hit_turn"], r["earliest_fused_turn_at_depth"][depth])
    )


def _assert_monotonic_across_depths(values_by_depth: dict[int, float], label: str, non_decreasing: bool) -> None:
    depths = sorted(values_by_depth)
    for a, b in zip(depths, depths[1:]):
        if non_decreasing:
            assert values_by_depth[b] >= values_by_depth[a] - 1e-9, (
                f"{label}: value decreased from depth {a} ({values_by_depth[a]:.4f}) to {b} ({values_by_depth[b]:.4f})."
            )
        else:
            assert values_by_depth[b] <= values_by_depth[a] + 1e-9, (
                f"{label}: value increased from depth {a} ({values_by_depth[a]:.4f}) to {b} ({values_by_depth[b]:.4f})."
            )


def _print_replacement_table(rows: list[dict], label: str) -> dict[int, dict]:
    print(f"REPLACEMENT ORACLE -- perfect reranker sees fused top-K only -- {label} (n={len(rows)}):")
    header = (
        f"  {'K':>4s} {'new_miss':>9s} {'hit_ret':>8s} {'hit_excl':>9s} {'net_chg':>8s} "
        f"{'HR@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'Eff':>6s} {'TechScore':>10s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_depth: dict[int, dict] = {}
    for depth in ORACLE_DEPTHS:
        acc = _depth_accounting(rows, depth)
        m = _replacement_metrics(rows, depth)
        # Accounting identity (Step 4): replacement hits = retained current
        # hits + newly recovered misses.
        assert m["hits"] == acc["replacement_hits"] == acc["hits_retained"] + acc["recoverable_misses"], (
            f"{label} depth {depth}: replacement hit count mismatch "
            f"({m['hits']} vs {acc['replacement_hits']}) -- bug in bucketing."
        )
        print(
            f"  {depth:>4d} {acc['recoverable_misses']:>9d} {acc['hits_retained']:>8d} {acc['hits_excluded']:>9d} "
            f"{acc['net_hit_change']:>+8d} {m['hit_rate_at_10']:>7.1%} {m['mrr']:>7.3f} {m['mttc']:>6.2f} "
            f"{m['efficiency']:>6.3f} {m['technical_score']:>10.4f}"
        )
        by_depth[depth] = {"acc": acc, "m": m}
    print()
    _assert_monotonic_across_depths({d: v["m"]["hit_rate_at_10"] for d, v in by_depth.items()}, f"{label} replacement HR@10", non_decreasing=True)
    _assert_monotonic_across_depths({d: v["m"]["technical_score"] for d, v in by_depth.items()}, f"{label} replacement TechScore", non_decreasing=True)
    _assert_monotonic_across_depths({d: v["m"]["mttc"] for d, v in by_depth.items()}, f"{label} replacement MTTC", non_decreasing=False)
    return by_depth


def _print_hybrid_table(rows: list[dict], label: str, baseline: dict[str, float]) -> dict[int, dict]:
    print(f"HYBRID-UNION ORACLE -- fused top-K UNION current heuristic top-10 -- {label} (n={len(rows)}):")
    print("  (candidate-AVAILABILITY ceiling only -- a real scorer can still demote a preserved hit; measure hit->miss regressions empirically before trusting this as \"no regression\")")
    header = (
        f"  {'K':>4s} {'new_miss':>9s} {'hit_preserved':>13s} {'tot_hits':>8s} "
        f"{'HR@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'Eff':>6s} {'TechScore':>10s} {'tech_gain':>9s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_depth: dict[int, dict] = {}
    for depth in ORACLE_DEPTHS:
        acc = _depth_accounting(rows, depth)
        m = _hybrid_metrics(rows, depth)
        # Accounting identity (Step 5): hybrid hits = current hits + newly
        # recovered misses -- every current hit is preserved by construction.
        assert m["hits"] == acc["hybrid_hits"] == acc["current_hits"] + acc["recoverable_misses"], (
            f"{label} depth {depth}: hybrid hit count mismatch "
            f"({m['hits']} vs {acc['hybrid_hits']}) -- bug in bucketing."
        )
        assert m["hit_rate_at_10"] >= baseline["hit_rate_at_10"] - 1e-9, (
            f"{label} depth {depth}: hybrid HitRate@10 ({m['hit_rate_at_10']:.4f}) is below the real baseline "
            f"({baseline['hit_rate_at_10']:.4f}) -- hybrid must never lose a current hit."
        )
        tech_gain = m["technical_score"] - baseline["technical_score"]
        print(
            f"  {depth:>4d} {acc['recoverable_misses']:>9d} {acc['current_hits']:>13d} {acc['hybrid_hits']:>8d} "
            f"{m['hit_rate_at_10']:>7.1%} {m['mrr']:>7.3f} {m['mttc']:>6.2f} {m['efficiency']:>6.3f} "
            f"{m['technical_score']:>10.4f} {tech_gain:>+9.4f}"
        )
        by_depth[depth] = {"acc": acc, "m": m}
    print()
    _assert_monotonic_across_depths({d: v["m"]["hit_rate_at_10"] for d, v in by_depth.items()}, f"{label} hybrid HR@10", non_decreasing=True)
    _assert_monotonic_across_depths({d: v["m"]["technical_score"] for d, v in by_depth.items()}, f"{label} hybrid TechScore", non_decreasing=True)
    _assert_monotonic_across_depths({d: v["m"]["mttc"] for d, v in by_depth.items()}, f"{label} hybrid MTTC", non_decreasing=False)
    return by_depth


def _print_recommendation(overall_repl: dict[int, dict], overall_hyb: dict[int, dict], scenario_hyb: dict[str, dict[int, dict]], never_recalled: int, total_misses: int) -> None:
    print("=" * 100)
    print("INTERPRETATION (diagnostic only -- not an implementation recommendation)")
    print("=" * 100)
    absolute_depth = max(ORACLE_DEPTHS)  # 200 -- retrieve()'s hard cap, the true ranking-side ceiling.
    pilot_depth = PRACTICAL_PILOT_DEPTH  # 100 -- what a real pilot would actually target.
    min_depth = min(ORACLE_DEPTHS)
    recov_pilot = overall_hyb[pilot_depth]["acc"]["recoverable_misses"]
    recov_min = overall_hyb[min_depth]["acc"]["recoverable_misses"]
    recov_absolute = overall_hyb[absolute_depth]["acc"]["recoverable_misses"]
    hits_excluded_pilot = overall_repl[pilot_depth]["acc"]["hits_excluded"]

    if hits_excluded_pilot > 0 and recov_pilot >= 10:
        print(
            f"- {recov_pilot} misses enter fused top-{pilot_depth}, but replacing the heuristic outright would "
            f"exclude {hits_excluded_pilot} sessions it currently hits from beyond position {pilot_depth} in the "
            f"raw fused order. A hybrid semantic score BLENDED WITH the existing full-depth heuristic (see "
            f"HYBRID-UNION ORACLE) is justified; replacing the heuristic outright is not, on this data. Note this "
            "is a candidate-AVAILABILITY guarantee only (see HYBRID-UNION's scope-limit note) -- an actual pilot "
            "must still measure hit-to-miss regressions empirically, not assume the union is safe by construction."
        )
    elif recov_pilot >= 10:
        print(
            f"- {recov_pilot} misses enter fused top-{pilot_depth} with zero current hits excluded at that depth -- "
            "a reranker change looks safe to pilot without a hybrid-preservation mechanism, though HYBRID-UNION "
            "is still the more conservative starting point, and empirical regression testing is still required."
        )
    else:
        print(f"- Only {recov_pilot} of {total_misses} current misses are recoverable even at depth {pilot_depth} -- reranking upside is limited; prioritize retrieval-side work.")

    if recov_pilot > 0 and recov_min < 0.5 * recov_pilot:
        print(
            f"- Most of the recoverable upside ({recov_pilot - recov_min} of {recov_pilot} sessions) only appears "
            f"between depth {min_depth} and depth {pilot_depth}, not within depth {min_depth} alone -- any semantic "
            "scorer would need to process a deep candidate set to capture it. Flag latency/cost as a feasibility "
            "consideration before committing to a specific depth."
        )

    # Absolute ranking-side ceiling (depth 200 = retrieve()'s hard cap) --
    # reported for completeness, explicitly NOT as a practical pilot target:
    # scoring 200 candidates/turn is very likely too expensive for real-time
    # cross-encoder inference. Gives the exact partition of today's misses.
    within_pilot = overall_hyb[pilot_depth]["acc"]["recoverable_misses"]
    between_pilot_and_absolute = recov_absolute - within_pilot
    hybrid_hr_absolute = overall_hyb[absolute_depth]["m"]["hit_rate_at_10"]
    print(
        f"- Absolute ranking-side ceiling (depth {absolute_depth}, retrieve()'s own hard cap -- not a practical "
        f"pilot depth): of {total_misses} current misses, {within_pilot} are within top-{pilot_depth}, "
        f"{between_pilot_and_absolute} more lie between {pilot_depth} and {absolute_depth}, and {never_recalled} "
        f"are never recalled at all. Hybrid-Union Oracle HitRate@10 @ {absolute_depth} = {hybrid_hr_absolute:.1%}. "
        f"No reranker of any depth can exceed this."
    )

    if never_recalled > 0:
        share = never_recalled / total_misses if total_misses else 0.0
        bigger = "a larger" if never_recalled > recov_absolute else "a smaller but still real"
        print(
            f"- {never_recalled} of {total_misses} misses ({share:.0%}) are never recalled at all (unreachable "
            f"even at depth {absolute_depth}) -- no reranker, however good, can fix these; this is {bigger} share "
            f"of misses than reranking could EVER reach (at any depth up to {absolute_depth}), and needs separate "
            "retrieval/query/gating diagnosis regardless of what happens with reranking."
        )

    print()
    print(f"Per-scenario (recoverable-by-depth-{pilot_depth}, the practical pilot depth):")
    for scenario in sorted(scenario_hyb):
        by_depth = scenario_hyb[scenario]
        n_misses = by_depth[pilot_depth]["acc"]["current_misses"]
        recov = by_depth[pilot_depth]["acc"]["recoverable_misses"]
        if scenario == "boundary":
            print(f"  - boundary: n={n_misses} misses -- sample too small (10 sessions total) for a strong conclusion either way.")
        elif scenario == "intent_override":
            print(
                f"  - intent_override: {recov} of {n_misses} misses recoverable by depth {pilot_depth}. This oracle "
                "report does not isolate *why* the rest are unrecoverable -- describe it as a broader query/state/"
                "retrieval problem rather than assuming a stale-slot or unextracted-slot cause; that requires the "
                "turn-by-turn slot tracing scripts/diagnose_intent_override.py does, not this report."
            )
        else:
            print(f"  - {scenario}: {recov} of {n_misses} misses recoverable by depth {pilot_depth}.")
    print()


def _oracle_reports(sessions: list[dict], check_baseline_reproduction: bool) -> None:
    print("=" * 100)
    print("ORACLE RECALL-CEILING REPORT")
    print("=" * 100)

    baseline = _current_baseline_metrics(sessions)
    n = len(sessions)
    print(
        f"Current baseline (computed from THIS run's sessions, not eval_runs.jsonl): "
        f"n={n} hits={baseline['hits']} HitRate@10={baseline['hit_rate_at_10']:.4f} "
        f"MRR={baseline['mrr']:.4f} MTTC={baseline['mttc']:.2f} Efficiency={baseline['efficiency']:.4f} "
        f"TechnicalScore={baseline['technical_score']:.4f}"
    )
    if check_baseline_reproduction:
        mismatch = (
            n != _EXPECTED_BASELINE_N
            or baseline["hits"] != _EXPECTED_BASELINE_HITS
            or abs(baseline["hit_rate_at_10"] - _EXPECTED_BASELINE_HR) > _BASELINE_TOL
        )
        if mismatch:
            banner = "!" * 100
            msg = (
                f"{banner}\n"
                f"BASELINE MISMATCH: this run reproduced n={n} hits={baseline['hits']} "
                f"HitRate@10={baseline['hit_rate_at_10']:.4f}, expected n={_EXPECTED_BASELINE_N} "
                f"hits={_EXPECTED_BASELINE_HITS} HitRate@10={_EXPECTED_BASELINE_HR}. The current code/configuration "
                "does NOT reproduce the 'post-revert-confirm' shipped baseline -- check AGENT_SHOPPER_FORCE_HEURISTIC, "
                "uncommitted changes to agent_shopper/, or LLM-path nondeterminism before trusting the oracle numbers "
                "below as comparable to prior eval_runs.jsonl entries.\n"
                f"{banner}"
            )
            print(msg)
            print(msg, file=sys.stderr)
    print()

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        groups[s["scenario_type"]].append(s)

    overall_repl = _print_replacement_table(sessions, "Overall")
    for scenario in sorted(groups):
        _print_replacement_table(groups[scenario], scenario)

    overall_hyb = _print_hybrid_table(sessions, "Overall", baseline)
    scenario_hyb: dict[str, dict[int, dict]] = {}
    for scenario in sorted(groups):
        scenario_baseline = _current_baseline_metrics(groups[scenario])
        scenario_hyb[scenario] = _print_hybrid_table(groups[scenario], scenario, scenario_baseline)

    # Step 7 invariants not already asserted inline in the print functions.
    assert baseline["hits"] + (n - baseline["hits"]) == n
    total_misses = n - baseline["hits"]
    never_recalled = sum(1 for s in sessions if s["hit_turn"] is None and s["recall_turn"] is None)
    for depth in ORACLE_DEPTHS:
        acc = overall_repl[depth]["acc"]
        assert acc["hits_retained"] + acc["hits_excluded"] == acc["current_hits"]
        assert acc["recoverable_misses"] + acc["unrecoverable_misses"] == acc["current_misses"]
        assert acc["replacement_hits"] <= overall_hyb[depth]["acc"]["hybrid_hits"], (
            f"depth {depth}: replacement_hits ({acc['replacement_hits']}) exceeds hybrid_hits "
            f"({overall_hyb[depth]['acc']['hybrid_hits']}) -- hybrid must be at least as good as replacement."
        )
        assert overall_hyb[depth]["m"]["hit_rate_at_10"] >= baseline["hit_rate_at_10"] - 1e-9

    # Hard invariant at ORACLE_DEPTHS' max (200, matching retrieve()'s own
    # hard `limit` cap -- retrieval.retrieve does `sorted(...)[:limit]`, see
    # recording_retrieve above): NO current hit can ever be excluded there --
    # its target must already be somewhere in that same capped pool, or the
    # heuristic couldn't have ranked it into today's top-10 from it.
    # REPLACEMENT and HYBRID-UNION must therefore coincide exactly at this
    # depth. This also cross-checks that this script's two independent
    # recording paths (the reranker-entry `pools` dict backing
    # recall_turn/never_recalled, and route_ranks_for_target's fused-
    # candidate-list rank) agree with each other -- if they didn't, this
    # would be the assertion that catches it, not a silent number mismatch.
    absolute_depth = max(ORACLE_DEPTHS)
    for label, rows in [("Overall", sessions)] + [(s, groups[s]) for s in sorted(groups)]:
        acc = _depth_accounting(rows, absolute_depth)
        assert acc["hits_excluded"] == 0, (
            f"{label}: {acc['hits_excluded']} current hit(s) excluded at depth {absolute_depth} (retrieve()'s hard "
            "cap) -- every current hit's target must be within retrieve()'s own capped pool; this indicates the "
            "two recording paths (reranker-entry `pools` vs. route_ranks_for_target's fused rank) have drifted "
            "apart, or retrieve()'s limit no longer matches ORACLE_DEPTHS' max."
        )
        assert acc["replacement_hits"] == acc["hybrid_hits"], (
            f"{label}: replacement_hits ({acc['replacement_hits']}) != hybrid_hits ({acc['hybrid_hits']}) at depth "
            f"{absolute_depth} -- these must coincide exactly once no current hit can be excluded."
        )

    _print_recommendation(overall_repl, overall_hyb, scenario_hyb, never_recalled, total_misses)
    print(f"Never-recalled misses (unreachable at any depth, incl. beyond 200): {never_recalled} of {total_misses}.")


if __name__ == "__main__":
    main()

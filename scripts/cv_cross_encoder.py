"""Phase 8: interactive 5-fold validation for the frozen cross-encoder pilot
(agent_shopper.cross_encoder_reranker), gated on scripts/
replay_cross_encoder_offline.py's Phase 7 offline gate already passing.

Unlike scripts/train_reranker_weights.py, nothing here is FIT from training
data -- alpha is one of a small fixed set (ALPHAS), never learned, never
selected per-session/per-scenario. The "5-fold" split (reusing
train_reranker_weights.build_folds, same seed/stratification convention)
exists purely to see fold-to-fold variance on session subsets, matching this
project's own established discipline that a single 200-session number can
be swung by a couple of flips in a small scenario bucket (see README's "What
we tried").

Runs the REAL interactive evaluator loop (evaluator.local_evaluator.evaluate,
via the Agent's normal respond() path) for every (alpha, fold) pair -- not
just offline-replayed candidate lists -- because a different alpha can
change which turn a session hits on, which changes shown-item history and
the rest of that session's trajectory (see cross_encoder_reranker.py's
module docstring). alpha=0.0 is included as the exact baseline every fold
(and is free -- FrozenCrossEncoderReranker short-circuits before ever
loading/calling the model at alpha=0.0).

Efficiency: the frozen cross-encoder's score for a given (query_text,
parent_asin) pair never depends on alpha or which other candidates are
present (pointwise, order-independent -- see FrozenCrossEncoderScorer). A
single FrozenCrossEncoderScorer instance, with its (query_text, parent_asin)
-> score cache, is shared across every fold and every alpha in this run
(patched in place of the production per-turn factory) -- turns whose
query_text/candidate-set recur across alpha runs (common for early-session
turns before trajectories diverge on a hit) are scored once, not once per
alpha.

Usage:
    AGENT_SHOPPER_CROSS_ENCODER_MODEL=<model> python3 scripts/cv_cross_encoder.py [--folds 5] [--seed 42] [--label my-run]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.config as config_mod  # noqa: E402
import agent_shopper.cross_encoder_reranker as cer_mod  # noqa: E402
import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
from agent_shopper.cross_encoder_reranker import FrozenCrossEncoderReranker, FrozenCrossEncoderScorer  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.train_reranker_weights import _force_heuristic_only, build_folds  # noqa: E402
from starter.agent import Agent  # noqa: E402

ALPHAS = (0.0, 0.15, 0.30, 0.50)


def _headline(result: dict) -> dict:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
    }


def run_cv(catalog_path: str, dataset_path: str, n_splits: int, seed: int) -> tuple[list[dict], dict]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    folds = build_folds(samples, n_splits=n_splits, seed=seed)
    agent = Agent(catalog_path)  # one Agent (and its indices) reused across the whole run

    shared_cache: dict[tuple[str, str], float] = {}
    shared_scorer = FrozenCrossEncoderScorer(score_cache=shared_cache)
    call_latencies: list[float] = []
    real_scorer_score = FrozenCrossEncoderScorer.score

    def timed_score(self, query_text, candidates):
        start = time.monotonic()
        result = real_scorer_score(self, query_text, candidates)
        call_latencies.append(time.monotonic() - start)
        return result

    def patched_factory() -> FrozenCrossEncoderReranker:
        return FrozenCrossEncoderReranker(scorer=shared_scorer)

    fold_reports: list[dict] = []
    load_start = time.monotonic()
    with _force_heuristic_only(), \
            patch.object(dialog_policy_mod, "FROZEN_CROSS_ENCODER_ENABLED", True), \
            patch.object(cer_mod, "get_frozen_cross_encoder_reranker", patched_factory), \
            patch.object(FrozenCrossEncoderScorer, "score", timed_score):
        for i, (_train, holdout) in enumerate(folds, start=1):
            fold_start = time.time()
            alpha_results: dict[float, dict] = {}
            for alpha in ALPHAS:
                with patch.object(config_mod, "FROZEN_CROSS_ENCODER_ALPHA", alpha):
                    result = evaluate(agent, holdout, catalog_ids, categories, products)
                alpha_results[alpha] = result
                print(
                    f"  fold {i} alpha={alpha:.2f}: HR@10={result['hit_rate_at_10']:.4f} "
                    f"MRR={result['mrr']:.4f} MTTC={result['mttc']:.2f} "
                    f"TechScore={result['recommended_technical_score']:.4f}  "
                    f"[{time.time() - fold_start:.0f}s elapsed this fold]",
                    flush=True,
                )

            baseline = alpha_results[0.0]
            baseline_hits = {s["sample_id"]: s for s in baseline["sessions"]}

            per_alpha_summary = {}
            for alpha in ALPHAS:
                res = alpha_results[alpha]
                sessions_by_id = {s["sample_id"]: s for s in res["sessions"]}
                flips_up = flips_down = 0
                rank_improved = rank_regressed = 0
                for sid, base_s in baseline_hits.items():
                    s = sessions_by_id[sid]
                    if not base_s["hit"] and s["hit"]:
                        flips_up += 1
                    elif base_s["hit"] and not s["hit"]:
                        flips_down += 1
                    elif base_s["hit"] and s["hit"] and base_s["best_rank"] != s["best_rank"]:
                        if s["best_rank"] < base_s["best_rank"]:
                            rank_improved += 1
                        else:
                            rank_regressed += 1
                per_alpha_summary[alpha] = {
                    "headline": _headline(res),
                    # metric_summary()'s own per-scenario dict (sample_count/
                    # hit_rate_at_10/mrr/mttc) -- no TechnicalScore per
                    # scenario (that's only meaningful as an overall blend),
                    # sufficient for decide()'s hit_rate_at_10 regression check.
                    "scenario_metrics": res["scenario_metrics"],
                    "flips_up": flips_up, "flips_down": flips_down, "net_flips": flips_up - flips_down,
                    "rank_improved": rank_improved, "rank_regressed": rank_regressed,
                }

            fold_reports.append({
                "fold": i, "n_holdout": len(holdout),
                "elapsed_seconds": round(time.time() - fold_start, 1),
                "per_alpha": per_alpha_summary,
            })
            print(f"  fold {i} done in {time.time() - fold_start:.0f}s\n", flush=True)

    total_load_elapsed = time.monotonic() - load_start
    latency_stats = {}
    if call_latencies:
        sorted_lat = sorted(call_latencies)
        pct = lambda p: sorted_lat[min(len(sorted_lat) - 1, int(p * len(sorted_lat)))]  # noqa: E731
        latency_stats = {
            "n_calls": len(call_latencies), "mean": statistics.fmean(call_latencies),
            "p50": pct(0.5), "p95": pct(0.95), "min": sorted_lat[0], "max": sorted_lat[-1],
            "cache_size_end_of_run": len(shared_cache),
        }
    return fold_reports, {"total_elapsed_seconds": round(total_load_elapsed, 1), "scoring_latency": latency_stats}


def decide(fold_reports: list[dict]) -> dict[float, dict]:
    """Recommended adoption criteria (per this script's module docstring /
    the task's Phase 8 spec): positive mean TechnicalScore delta, >=4/5 folds
    non-negative, positive net session flips, no severe intent_override/
    boundary regression, evaluated per nonzero alpha independently."""
    decisions: dict[float, dict] = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            continue
        deltas = [
            f["per_alpha"][alpha]["headline"]["technical_score"] - f["per_alpha"][0.0]["headline"]["technical_score"]
            for f in fold_reports
        ]
        non_negative_folds = sum(1 for d in deltas if d >= -1e-9)
        net_flips = sum(f["per_alpha"][alpha]["net_flips"] for f in fold_reports)
        severe_regression = False
        for f in fold_reports:
            for scenario in ("intent_override", "boundary"):
                base_sm = f["per_alpha"][0.0]["scenario_metrics"].get(scenario)
                alpha_sm = f["per_alpha"][alpha]["scenario_metrics"].get(scenario)
                if base_sm and alpha_sm and alpha_sm["hit_rate_at_10"] < base_sm["hit_rate_at_10"] - 0.10:
                    severe_regression = True
        mean_delta = statistics.fmean(deltas)
        adopt = (
            mean_delta > 1e-9
            and non_negative_folds >= 4
            and net_flips > 0
            and not severe_regression
        )
        decisions[alpha] = {
            "mean_technical_score_delta": round(mean_delta, 6),
            "non_negative_folds": non_negative_folds,
            "net_flips": net_flips,
            "severe_intent_override_or_boundary_regression": severe_regression,
            "adopt": adopt,
        }
    return decisions


def select_alpha(decisions: dict[float, dict]) -> float | None:
    candidates = [a for a, d in decisions.items() if d["adopt"]]
    if not candidates:
        return None
    return max(candidates, key=lambda a: decisions[a]["mean_technical_score_delta"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    print(f"Cross-encoder model: {config_mod.FROZEN_CROSS_ENCODER_MODEL} (AGENT_SHOPPER_CROSS_ENCODER_MODEL)")
    print(f"Depth={config_mod.FROZEN_CROSS_ENCODER_DEPTH}  RRF_K={config_mod.RRF_K}  Alphas={ALPHAS}\n")

    start = time.time()
    fold_reports, meta = run_cv(str(ROOT / args.catalog), str(ROOT / args.dataset), args.folds, args.seed)
    elapsed = time.time() - start

    print(f"\n=== Cross-encoder CV ({args.folds}-fold, {elapsed:.1f}s total) ===")
    decisions = decide(fold_reports)
    for alpha, d in decisions.items():
        print(
            f"alpha={alpha:.2f}: mean TechScore delta={d['mean_technical_score_delta']:+.4f}  "
            f"non-negative folds={d['non_negative_folds']}/{args.folds}  net flips={d['net_flips']:+d}  "
            f"severe intent_override/boundary regression={d['severe_intent_override_or_boundary_regression']}  "
            f"-> {'ADOPT' if d['adopt'] else 'DO NOT ADOPT'}"
        )
    selected = select_alpha(decisions)
    print(f"\nSelected alpha: {selected if selected is not None else 'NONE -- revert/remove the production integration'}")

    if meta["scoring_latency"]:
        ls = meta["scoring_latency"]
        print(
            f"\nScoring latency across {ls['n_calls']} real model calls: mean={ls['mean']:.3f}s "
            f"p50={ls['p50']:.3f}s p95={ls['p95']:.3f}s min={ls['min']:.3f}s max={ls['max']:.3f}s "
            f"(score cache ended with {ls['cache_size_end_of_run']} entries)"
        )

    log_path = ROOT / "cross_encoder_cv_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": args.label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": config_mod.FROZEN_CROSS_ENCODER_MODEL, "elapsed_seconds": round(elapsed, 1),
            "n_folds": args.folds, "seed": args.seed, "alphas": ALPHAS,
            "fold_reports": fold_reports, "decisions": decisions, "selected_alpha": selected, "meta": meta,
        }) + "\n")
    print(f"\nAppended run to {log_path}")


if __name__ == "__main__":
    main()

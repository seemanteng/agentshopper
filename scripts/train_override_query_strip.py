"""Validates config.OVERRIDE_QUERY_STRIP_ENABLED via stratified k-fold
cross-validation across the 200-session public dev set -- same discipline as
scripts/train_reranker_weights.py, adapted for a toggle rather than a fitted
weight (there is nothing to fit here: the boilerplate-stripping logic in
context.py is deterministic, so this script's only job is an honest,
fold-by-fold A/B comparison of the flag off vs on via the real evaluator
loop).

Background: scripts/diagnose_intent_override.py's per-route Recall@10/50/100
breakdown showed intent_override sessions recall the target into the fused
pool only 23.6% of the time post-override, vs. a session-level never-recalled
rate of 40% (12/30) -- far worse than buying/browsing (~10-11%). README's
"What we tried" already traces this to evaluator dialogue-scaffolding
("Actually, ignore my earlier preference...", "Ask me about one specific
attribute.") diluting the BM25/TF-IDF query text. A blanket fix (strip that
scaffolding from *every* turn's query) was already tried and reverted: it
fixed intent_override but regressed buying and boundary's MRR enough to
net-regress TechnicalScore overall.

This script validates a narrower version: strip the same scaffolding, but
only on turns in sessions where state.has_overridden() is already True (see
context.build_query_text's `session_has_overridden` param) -- i.e. never
touches buying/browsing/boundary sessions that haven't had an override,
which is exactly the scope the earlier attempt's regression came from.

Usage:
    python3 scripts/train_override_query_strip.py [--folds 5] [--seed 42] [--label my-run]
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
import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.train_reranker_weights import build_folds  # noqa: E402
from starter.agent import Agent  # noqa: E402


def _run_variant(agent: Agent, strip_enabled: bool, samples: list[dict], catalog_ids, categories, products) -> dict:
    """Runs `agent` over `samples` via the real evaluate() loop, with
    config.OVERRIDE_QUERY_STRIP_ENABLED forced to `strip_enabled` for the
    duration. Patches dialog_policy_mod's own bound name (not config's),
    same reasoning train_reranker_weights._force_heuristic_only gives for
    patching dialog_policy_mod.active_provider rather than llm_client's --
    `from agent_shopper.config import OVERRIDE_QUERY_STRIP_ENABLED` in
    context.py binds a private copy of the value at import time, so it must
    be patched where it's actually read (agent_shopper.context), not on the
    config module object."""
    import agent_shopper.context as context_mod
    with patch.object(context_mod, "OVERRIDE_QUERY_STRIP_ENABLED", strip_enabled):
        return evaluate(agent, samples, catalog_ids, categories, products)


def _headline(result: dict) -> dict:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "recommended_technical_score": result["recommended_technical_score"],
        "scenario_metrics": result.get("scenario_metrics"),
    }


def run_cv(catalog_path: str, dataset_path: str, n_splits: int, seed: int) -> list[dict]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    folds = build_folds(samples, n_splits=n_splits, seed=seed)
    agent = Agent(catalog_path)  # one Agent (BM25/TF-IDF indices) for the whole CV run

    fold_reports = []
    for i, (_train_samples, holdout_samples) in enumerate(folds, start=1):
        # No fitting step -- the flag is a deterministic on/off, not a
        # learned parameter -- so unlike train_reranker_weights.py there is
        # no training-fold pass here, only the held-out A/B comparison.
        baseline_result = _run_variant(agent, False, holdout_samples, catalog_ids, categories, products)
        stripped_result = _run_variant(agent, True, holdout_samples, catalog_ids, categories, products)

        baseline_hits = {s["sample_id"]: s["hit"] for s in baseline_result["sessions"]}
        stripped_hits = {s["sample_id"]: s["hit"] for s in stripped_result["sessions"]}
        flips_up = sum(1 for sid in baseline_hits if not baseline_hits[sid] and stripped_hits[sid])
        flips_down = sum(1 for sid in baseline_hits if baseline_hits[sid] and not stripped_hits[sid])

        fold_reports.append({
            "fold": i,
            "n_holdout": len(holdout_samples),
            "baseline": _headline(baseline_result),
            "stripped": _headline(stripped_result),
            "technical_score_delta": round(
                stripped_result["recommended_technical_score"] - baseline_result["recommended_technical_score"], 6
            ),
            "hit_flips_up": flips_up,
            "hit_flips_down": flips_down,
        })

    return fold_reports


def decide(fold_reports: list[dict]) -> dict:
    deltas = [f["technical_score_delta"] for f in fold_reports]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    net_flips = sum(f["hit_flips_up"] - f["hit_flips_down"] for f in fold_reports)
    # Same adoption bar as train_reranker_weights.decide(): a clear majority
    # win with a net-positive flip count across the whole CV run.
    adopt = wins > losses and wins >= (len(deltas) + 1) // 2 and net_flips > 0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_technical_score_delta": round(statistics.fmean(deltas), 6),
        "net_hit_flips": net_flips,
        "adopt": adopt,
    }


def _scenario_deltas(fold_reports: list[dict]) -> dict[str, dict]:
    """Aggregates each scenario_type's hit_rate_at_10 across folds (summed
    numerator/denominator, not averaged fold percentages, so folds with
    different holdout sizes weight correctly) -- this is what actually
    matters for the adoption decision: the earlier blanket-strip attempt's
    problem was a regression hidden inside per-scenario numbers that the
    aggregate alone wouldn't have shown until it was too late."""
    scenarios: dict[str, dict[str, list[float]]] = {}
    for f in fold_reports:
        for variant in ("baseline", "stripped"):
            sm = f[variant]["scenario_metrics"] or {}
            for scenario, metrics in sm.items():
                scenarios.setdefault(scenario, {"baseline_hr": [], "stripped_hr": []})
                scenarios[scenario][f"{variant}_hr"].append(metrics["hit_rate_at_10"])
    return {
        scenario: {
            "baseline_hit_rate_at_10": round(statistics.fmean(v["baseline_hr"]), 4) if v["baseline_hr"] else None,
            "stripped_hit_rate_at_10": round(statistics.fmean(v["stripped_hr"]), 4) if v["stripped_hr"] else None,
        }
        for scenario, v in scenarios.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    catalog_path = str(ROOT / args.catalog)
    dataset_path = str(ROOT / args.dataset)

    start = time.time()
    fold_reports = run_cv(catalog_path, dataset_path, args.folds, args.seed)
    decision = decide(fold_reports)
    scenario_deltas = _scenario_deltas(fold_reports)
    elapsed = time.time() - start

    print(f"\n=== Override query-strip CV ({args.folds}-fold, {elapsed:.1f}s) ===")
    for f in fold_reports:
        print(
            f"fold {f['fold']}: n_holdout={f['n_holdout']:3d}  "
            f"TechnicalScore {f['baseline']['recommended_technical_score']:.4f} -> "
            f"{f['stripped']['recommended_technical_score']:.4f}  "
            f"(delta {f['technical_score_delta']:+.4f}, flips +{f['hit_flips_up']}/-{f['hit_flips_down']})"
        )

    print(
        f"\n{decision['wins']} wins / {decision['losses']} losses / {decision['ties']} ties across "
        f"{args.folds} folds  mean TechnicalScore delta {decision['mean_technical_score_delta']:+.4f}  "
        f"net hit flips {decision['net_hit_flips']:+d}"
    )
    print("\nPer-scenario Hit Rate@10 (mean across folds' holdout sets):")
    for scenario in sorted(scenario_deltas):
        d = scenario_deltas[scenario]
        print(f"  {scenario:16s} {d['baseline_hit_rate_at_10']} -> {d['stripped_hit_rate_at_10']}")
    print(f"\nDecision: {'ADOPT' if decision['adopt'] else 'DO NOT ADOPT'} (see module docstring for the rule)")

    log_path = ROOT / "override_query_strip_cv_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "n_folds": args.folds,
            "seed": args.seed,
            "folds": fold_reports,
            "decision": decision,
            "scenario_deltas": scenario_deltas,
        }) + "\n")
    print(f"\nAppended run to {log_path}")


if __name__ == "__main__":
    main()

"""Fits HeuristicReranker's 5 linear weights via logistic regression,
validated by stratified k-fold cross-validation across the 200-session
public dev set -- NOT a naive fit-and-eval on the same sessions, which would
produce a flattering number that may not transfer to the hidden 800-session
set (README's "What we tried" already shows small scenario buckets like
intent_override, n=30, swinging hard on a couple of session flips).

Pipeline, once per fold:
  1. Run the current default agent (heuristic-only, deterministic, free)
     over the fold's *training* sessions, recording every candidate's 5
     feature values (agent_shopper.reranker._feature_vector) plus whether it
     was that session's true target -- via the same reranker-monkeypatch
     technique scripts/diagnose_retrieval.py already uses, so agent behavior
     during logging is byte-identical to a normal evaluator run.
  2. Fit sklearn.linear_model.LogisticRegression(class_weight="balanced") on
     those rows -> the fold's learned replacement for
     config.HEURISTIC_RERANK_WEIGHTS (same linear-formula shape, no
     intercept, so it drops straight into HeuristicReranker(weights=...)).
  3. Re-run the *actual* interactive evaluator loop on the fold's held-out
     sessions twice -- once with the baseline hand-tuned weights, once with
     the fold's learned weights -- and compare Hit Rate@10/MRR/MTTC/
     TechnicalScore. This step is not optional: the simulated conversation
     is stateful (evaluator.local_evaluator.customer_reply adapts to the
     agent's own ask_attribute from prior turns), so an offline
     classification score on logged rows cannot stand in for it.

Decision rule: adopt the learned weights only if they win (or are neutral)
in a clear majority of the k folds, with a net-positive hit-flip count
across all folds -- a mean improvement driven by 1-2 session flips in a
single fold is treated as noise, not a result, per the same discipline the
project's other tuning changes are held to (see README "What we tried").

This script never modifies agent_shopper/config.py itself -- if the CV
decision is ADOPT, it prints a final full-dataset fit for a human to review
and copy into config.py by hand, alongside a README "What we tried" entry
with the concrete numbers, matching how every other change in that section
was made.

Usage:
    python3 scripts/train_reranker_weights.py [--folds 5] [--seed 42] [--label my-run]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
import agent_shopper.reranker as reranker_mod  # noqa: E402
from agent_shopper.config import HEURISTIC_RERANK_WEIGHTS  # noqa: E402
from agent_shopper.reranker import _feature_vector, _minmax_normalize, _slots_from_summary  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402

FEATURE_NAMES = ("bm25", "vector", "attr_match", "rating", "price_fit")


@contextmanager
def _force_heuristic_only():
    """Ensures every reranker call made inside this block takes the free
    heuristic path, regardless of ambient OPENAI_API_KEY/ANTHROPIC_API_KEY.

    config.FORCE_HEURISTIC is *not* actually wired into the production
    per-turn engine choice in dialog_policy.process_turn -- it only gates
    the separate, unused reranker.get_reranker() -- so setting the env var
    alone does not guarantee the heuristic-only path there. Patching
    dialog_policy's own `active_provider` reference directly (not
    llm_client's, since `from x import y` binds a private copy of the name)
    is what actually forces it, and is required for this script's
    fit-vs-evaluate comparisons to be a clean, deterministic, zero-cost,
    apples-to-apples measurement of the weights alone.
    """
    with patch.object(dialog_policy_mod, "active_provider", return_value=None):
        yield


def build_folds(samples: list[dict], n_splits: int = 5, seed: int = 42) -> list[tuple[list[dict], list[dict]]]:
    """Stratified (by scenario_type) k-fold split over whole sessions -- each
    sample lands in exactly one fold's held-out set, so a fold's training
    rows and its held-out evaluation never share a session."""
    labels = [s["scenario_type"] for s in samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = list(range(len(samples)))
    return [
        ([samples[i] for i in train_idx], [samples[i] for i in holdout_idx])
        for train_idx, holdout_idx in skf.split(indices, labels)
    ]


def _log_training_rows(agent: Agent, samples: list[dict], catalog_ids, categories, products) -> list[dict]:
    """Runs `agent` over `samples` via the real evaluator loop, recording
    every candidate HeuristicReranker.rerank() actually saw each turn -- the
    exact 5-feature vector plus the is-target label -- without changing
    what the agent does (delegates to the real method after recording)."""
    rows: list[dict] = []
    real_rerank = reranker_mod.HeuristicReranker.rerank
    target_holder = {"target": None}

    def recording_rerank(self, ctx, candidates, top_k):
        if candidates:
            target = target_holder["target"]
            bm25_raw = {i: c.route_scores.get("keyword", 0.0) for i, c in enumerate(candidates)}
            vector_raw = {i: c.route_scores.get("vector", 0.0) for i, c in enumerate(candidates)}
            bm25_norm = _minmax_normalize(bm25_raw)
            vector_norm = _minmax_normalize(vector_raw)
            slots = _slots_from_summary(ctx.session.slot_summary)
            for i, candidate in enumerate(candidates):
                features = _feature_vector(
                    candidate.product, slots, bm25_norm.get(i, 0.0), vector_norm.get(i, 0.0),
                    ctx.profile.rating_floor_hint,
                )
                rows.append({**features, "label": int(candidate.product.parent_asin == target)})
        return real_rerank(self, ctx, candidates, top_k)

    with _force_heuristic_only(), patch.object(reranker_mod.HeuristicReranker, "rerank", recording_rerank):
        for sample in samples:
            target_holder["target"] = str(sample["ground_truth"]["parent_asin"])
            evaluate(agent, [sample], catalog_ids, categories, products)

    return rows


def fit_weights(rows: list[dict]) -> dict[str, float]:
    """Fits LogisticRegression(class_weight='balanced') on the logged rows
    and returns the fitted coefficients as a replacement weights dict --
    same linear-formula shape as HEURISTIC_RERANK_WEIGHTS (no intercept:
    HeuristicReranker's score is a raw ranking score, never passed through a
    sigmoid, so an intercept would only shift every candidate in a turn by
    the same constant and never change rank order)."""
    y = np.array([row["label"] for row in rows])
    if len(set(y.tolist())) < 2:
        # Degenerate fold (e.g. the target never entered the candidate pool
        # in any training session) -- fall back to the hand-tuned weights
        # rather than fitting a meaningless single-class model.
        return dict(HEURISTIC_RERANK_WEIGHTS)
    X = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])
    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)
    return {name: float(coef) for name, coef in zip(FEATURE_NAMES, model.coef_[0])}


def _run_variant(agent: Agent, weights: dict[str, float] | None, samples: list[dict], catalog_ids, categories, products) -> dict:
    """Runs `agent` over `samples` via the real evaluate() loop, with
    HeuristicReranker's weights overridden to `weights` for the duration
    (None = untouched, i.e. the hand-tuned config default). `agent` is
    reused rather than freshly constructed here -- building one (BM25/TF-IDF
    indices over the 50k catalog) costs ~15s on its own, and agent.reset()
    already fully re-initializes all per-session state between sessions, the
    same reuse pattern scripts/run_local_eval.py's own evaluator run relies
    on."""
    if weights is None:
        with _force_heuristic_only():
            return evaluate(agent, samples, catalog_ids, categories, products)

    original_init = reranker_mod.HeuristicReranker.__init__

    def patched_init(self, weights_arg=None):
        original_init(self, weights_arg if weights_arg is not None else weights)

    with _force_heuristic_only(), patch.object(reranker_mod.HeuristicReranker, "__init__", patched_init):
        return evaluate(agent, samples, catalog_ids, categories, products)


def _headline(result: dict) -> dict:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "recommended_technical_score": result["recommended_technical_score"],
    }


def run_cv(catalog_path: str, dataset_path: str, n_splits: int, seed: int) -> list[dict]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    folds = build_folds(samples, n_splits=n_splits, seed=seed)
    # One Agent (and its BM25/TF-IDF indices) for the whole CV run -- see
    # _run_variant's docstring for why reuse across sessions is safe.
    agent = Agent(catalog_path)

    fold_reports = []
    for i, (train_samples, holdout_samples) in enumerate(folds, start=1):
        rows = _log_training_rows(agent, train_samples, catalog_ids, categories, products)
        learned_weights = fit_weights(rows)

        baseline_result = _run_variant(agent, None, holdout_samples, catalog_ids, categories, products)
        learned_result = _run_variant(agent, learned_weights, holdout_samples, catalog_ids, categories, products)

        baseline_hits = {s["sample_id"]: s["hit"] for s in baseline_result["sessions"]}
        learned_hits = {s["sample_id"]: s["hit"] for s in learned_result["sessions"]}
        flips_up = sum(1 for sid in baseline_hits if not baseline_hits[sid] and learned_hits[sid])
        flips_down = sum(1 for sid in baseline_hits if baseline_hits[sid] and not learned_hits[sid])

        fold_reports.append({
            "fold": i,
            "n_train": len(train_samples),
            "n_holdout": len(holdout_samples),
            "learned_weights": learned_weights,
            "baseline": _headline(baseline_result),
            "learned": _headline(learned_result),
            "technical_score_delta": round(
                learned_result["recommended_technical_score"] - baseline_result["recommended_technical_score"], 6
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
    # Adopt only on a clear majority win with a net-positive flip count
    # across the whole CV run -- a mean improvement driven by 1-2 flips in a
    # single fold is noise, not a result (see module docstring).
    adopt = wins > losses and wins >= (len(deltas) + 1) // 2 and net_flips > 0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_technical_score_delta": round(statistics.fmean(deltas), 6),
        "net_hit_flips": net_flips,
        "adopt": adopt,
    }


def fit_full_weights(catalog_path: str, dataset_path: str) -> dict[str, float]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    rows = _log_training_rows(Agent(catalog_path), samples, catalog_ids, categories, products)
    return fit_weights(rows)


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
    elapsed = time.time() - start

    print(f"\n=== Reranker weight CV ({args.folds}-fold, {elapsed:.1f}s) ===")
    for f in fold_reports:
        print(
            f"fold {f['fold']}: n_holdout={f['n_holdout']:3d}  "
            f"TechnicalScore {f['baseline']['recommended_technical_score']:.4f} -> "
            f"{f['learned']['recommended_technical_score']:.4f}  "
            f"(delta {f['technical_score_delta']:+.4f}, flips +{f['hit_flips_up']}/-{f['hit_flips_down']})"
        )
        print(f"  learned weights: {json.dumps({k: round(v, 4) for k, v in f['learned_weights'].items()})}")

    print(
        f"\n{decision['wins']} wins / {decision['losses']} losses / {decision['ties']} ties across "
        f"{args.folds} folds  mean TechnicalScore delta {decision['mean_technical_score_delta']:+.4f}  "
        f"net hit flips {decision['net_hit_flips']:+d}"
    )
    print(f"Decision: {'ADOPT' if decision['adopt'] else 'DO NOT ADOPT'} (see module docstring for the rule)")

    full_weights = None
    if decision["adopt"]:
        print("\nFitting final weights on the full dataset (for review before editing config.py)...")
        full_weights = fit_full_weights(catalog_path, dataset_path)
        print(f"Full-data learned weights:          {json.dumps({k: round(v, 4) for k, v in full_weights.items()})}")
        print(f"Current config.HEURISTIC_RERANK_WEIGHTS: {json.dumps(HEURISTIC_RERANK_WEIGHTS)}")

    log_path = ROOT / "reranker_cv_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "n_folds": args.folds,
            "seed": args.seed,
            "folds": fold_reports,
            "decision": decision,
            "full_data_weights": full_weights,
        }) + "\n")
    print(f"\nAppended run to {log_path}")


if __name__ == "__main__":
    main()

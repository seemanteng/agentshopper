"""Fits agent_shopper.override_model.OverrideModel's 5 weights via logistic
regression on real per-turn features, traced from the actual agent running
over the full 200-session public set (same monkeypatch-and-delegate
technique scripts/diagnose_retrieval.py and train_reranker_weights.py both
use, so agent behavior during logging is byte-identical to a normal
evaluator run).

Labels are NOT the contradiction-language regex re-applied to itself (that
would just be circular -- a classifier trained to reproduce its own input
feature learns nothing new). They're ground truth: `1` on the exact turn
`sample["behavior"]["override"]["turn"]` for intent_override sessions
(materialize_hidden_fields exposes this), `0` on every other turn across
all 200 sessions (all four scenario_types) -- so the model sees far more
negatives than positives (~30 positive turns out of ~1300-1600 total), and
`class_weight="balanced"` accounts for that the same way
train_reranker_weights.fit_weights does for HeuristicReranker's 5 features.

This model's real validation belongs to whatever downstream feature/policy
eventually consumes `override_probability` (a LambdaMART reranker attempt
was tried and rejected by CV -- see README's "What we tried" -- so nothing
currently does), not to this script. This script does a lighter-weight
check: 5-fold stratified CV (by scenario_type) purely on classification
metrics (AUC, precision/recall at 0.5) to confirm the fit is stable and not
wildly overfit to ~30 positives, then fits on the full dataset and prints
the coefficients to copy into config.py by hand -- same pattern
train_reranker_weights.py's --label/decide()/full-fit split already uses,
just without a HitRate@10-based adopt/reject gate (that gate belongs to
whatever script eventually consumes this feature end-to-end).

Usage:
    python3 scripts/train_override_model.py [--folds 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
from agent_shopper.override_model import FEATURE_NAMES  # noqa: E402
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


def _rows_to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])
    y = np.array([row["label"] for row in rows])
    return X, y


def fit_model(rows: list[dict]) -> tuple[dict[str, float], float]:
    X, y = _rows_to_xy(rows)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    weights = {name: float(coef) for name, coef in zip(FEATURE_NAMES, model.coef_[0])}
    return weights, float(model.intercept_[0])


def run_cv(catalog_path: str, dataset_path: str, n_splits: int, seed: int) -> list[dict]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    # Trace every session exactly once (features don't depend on the
    # override model's own weights -- process_turn only calls
    # OverrideModel().predict_proba() *after* _override_features returns,
    # and that prediction has no feedback into this turn's own features),
    # then split the already-collected (session_id-free) rows by session
    # membership for CV -- avoids re-running the agent per fold.
    rows_by_sample: dict[str, list[dict]] = {}
    real_override_features = dialog_policy_mod._override_features
    sample_holder = {"id": None}

    def tagging_override_features(old_slots, extracted, contradiction_this_turn, turn):
        features = real_override_features(old_slots, extracted, contradiction_this_turn, turn)
        rows_by_sample.setdefault(sample_holder["id"], []).append(features)
        return features

    labels = [s["scenario_type"] for s in samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(range(len(samples)), labels))

    with patch.object(dialog_policy_mod, "_override_features", tagging_override_features):
        for sample in samples:
            sample_holder["id"] = sample["sample_id"]
            session_id = f"trainov_{sample['sample_id']}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
            override = effective_behavior.get("override") or {}
            override_turn = int(override["turn"]) if sample["scenario_type"] == "intent_override" and override else None

            disclosed: set[str] = set()
            boundary_used = False
            # Matches evaluate()'s own initialization exactly (local_evaluator.py):
            # pre-override, "hitting" the eventual target is incidental, not a
            # real conversion, so only intent_override sessions start False.
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
            turn_labels: list[int] = []
            for turn in range(1, MAX_TURNS + 1):
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                turn_labels.append(1 if override_turn is not None and turn == override_turn else 0)
                # Stop exactly where evaluate() would (first real hit, or
                # MAX_TURNS) -- without this, later turns of an
                # already-converted session are synthetic continuations
                # that never happen in a real run, contaminating both the
                # feature distribution and the labels for this classifier.
                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                if override_applied and target in ranked:
                    break
                if turn == MAX_TURNS:
                    break
                if not override_applied and override_turn is not None and turn + 1 == override_turn:
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    user_message, boundary_used = customer_reply(
                        effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                    )
            for row, label in zip(rows_by_sample[sample["sample_id"]], turn_labels):
                row["label"] = label

    fold_reports = []
    for i, (train_idx, holdout_idx) in enumerate(fold_indices, start=1):
        train_rows = [row for idx in train_idx for row in rows_by_sample[samples[idx]["sample_id"]]]
        holdout_rows = [row for idx in holdout_idx for row in rows_by_sample[samples[idx]["sample_id"]]]
        weights, intercept = fit_model(train_rows)

        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        Xtr, ytr = _rows_to_xy(train_rows)
        model.fit(Xtr, ytr)
        Xho, yho = _rows_to_xy(holdout_rows)
        proba = model.predict_proba(Xho)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(yho, proba) if len(set(yho.tolist())) > 1 else float("nan")
        precision = precision_score(yho, preds, zero_division=0)
        recall = recall_score(yho, preds, zero_division=0)
        fold_reports.append({
            "fold": i, "n_train_turns": len(train_rows), "n_holdout_turns": len(holdout_rows),
            "n_holdout_positives": int(yho.sum()), "auc": None if auc != auc else round(float(auc), 4),
            "precision_at_0.5": round(float(precision), 4), "recall_at_0.5": round(float(recall), 4),
            "weights": weights, "intercept": round(intercept, 4),
        })
    return fold_reports, rows_by_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    catalog_path = str(ROOT / args.catalog)
    dataset_path = str(ROOT / args.dataset)

    start = time.time()
    fold_reports, rows_by_sample = run_cv(catalog_path, dataset_path, args.folds, args.seed)
    elapsed = time.time() - start

    print(f"\n=== Override model CV ({args.folds}-fold, {elapsed:.1f}s) ===")
    for f in fold_reports:
        print(
            f"fold {f['fold']}: n_holdout_turns={f['n_holdout_turns']:4d} "
            f"({f['n_holdout_positives']} positive)  AUC={f['auc']}  "
            f"precision@0.5={f['precision_at_0.5']}  recall@0.5={f['recall_at_0.5']}"
        )

    all_rows = [row for rows in rows_by_sample.values() for row in rows]
    n_pos = sum(r["label"] for r in all_rows)
    print(f"\nTotal turns: {len(all_rows)}  positive (true override turns): {n_pos}")

    full_weights, full_intercept = fit_model(all_rows)
    print(f"\nFull-data fit (for review before editing config.py):")
    print(f"  OVERRIDE_MODEL_WEIGHTS = {json.dumps({k: round(v, 4) for k, v in full_weights.items()})}")
    print(f"  OVERRIDE_MODEL_INTERCEPT = {round(full_intercept, 4)}")

    log_path = ROOT / "override_model_cv_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "n_folds": args.folds,
            "seed": args.seed,
            "folds": fold_reports,
            "full_data_weights": full_weights,
            "full_data_intercept": full_intercept,
            "n_total_turns": len(all_rows),
            "n_positive_turns": n_pos,
        }) + "\n")
    print(f"\nAppended run to {log_path}")


if __name__ == "__main__":
    main()

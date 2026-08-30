"""Small, hand-reasoned sweep over ROUTE_WEIGHTS' dense-route weight
(and DENSE_FIELD_WEIGHTS left fixed) -- NOT a grid search. A handful of
candidate configs, each run once on the full 200-session public set and
compared against the currently-shipped weights.

Deliberately not a large grid: picking a winner out of many candidates by
re-running against the same 200 public sessions repeatedly is its own
overfitting risk (the same reason train_reranker_weights.py insists on
k-fold CV rather than a naive fit-and-eval). Keeping this to a small number
of hand-reasoned alternatives, evaluated once each, keeps that risk small --
this is the same "hand-set priors, tuned against scripts/run_local_eval.py"
process every other non-learned weight in config.py already went through,
not a new fitting procedure.

Usage:
    python3 scripts/sweep_dense_weights.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
import agent_shopper.orchestrator as orchestrator_mod  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402

CURRENT = {
    "buying": {"category": 0.45, "keyword": 0.30, "vector": 0.10, "dense": 0.15},
    "browsing": {"category": 0.15, "keyword": 0.25, "vector": 0.30, "dense": 0.30},
}

CANDIDATES = {
    "current (shipped)": CURRENT,
    "dense-up (reallocate from vector, since step 1 showed vector/keyword overlap)": {
        "buying": {"category": 0.45, "keyword": 0.30, "vector": 0.05, "dense": 0.20},
        "browsing": {"category": 0.15, "keyword": 0.20, "vector": 0.20, "dense": 0.45},
    },
    "dense-down (more conservative, check if less dense weight keeps most of the gain)": {
        "buying": {"category": 0.45, "keyword": 0.33, "vector": 0.12, "dense": 0.10},
        "browsing": {"category": 0.15, "keyword": 0.30, "vector": 0.35, "dense": 0.20},
    },
}


def _headline(result: dict) -> dict:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "recommended_technical_score": result["recommended_technical_score"],
        "intent_override_hit_rate": result["scenario_metrics"]["intent_override"]["hit_rate_at_10"],
    }


def main() -> None:
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")
    agent = Agent(str(ROOT / "data/catalog.jsonl"))  # one Agent (indices incl. dense cache) reused across candidates

    print(f"\n=== Dense route weight sweep ({len(CANDIDATES)} candidates, full 200-session set each) ===")
    results = {}
    for name, weights in CANDIDATES.items():
        start = time.time()
        with patch.object(orchestrator_mod, "ROUTE_WEIGHTS", weights), \
                patch.object(dialog_policy_mod, "active_provider", return_value=None):
            result = evaluate(agent, samples, catalog_ids, categories, products)
        elapsed = time.time() - start
        h = _headline(result)
        results[name] = h
        print(
            f"\n{name} ({elapsed:.0f}s):\n"
            f"  TechnicalScore={h['recommended_technical_score']:.4f}  "
            f"HitRate@10={h['hit_rate_at_10']:.4f}  MRR={h['mrr']:.4f}  MTTC={h['mttc']:.2f}  "
            f"intent_override HitRate@10={h['intent_override_hit_rate']:.4f}"
        )

    best = max(results, key=lambda n: results[n]["recommended_technical_score"])
    print(f"\nBest by TechnicalScore: {best}")


if __name__ == "__main__":
    main()

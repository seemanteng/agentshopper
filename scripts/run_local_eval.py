"""Wraps evaluator/local_evaluator.py, then breaks results down by
scenario_type/difficulty_bucket/category_bucket and appends a summary row
to a gitignored eval_runs.jsonl so tuning iterations are comparable.

Usage:
    python3 scripts/run_local_eval.py [--label my-change] [--catalog ...] [--dataset ...]

Set AGENT_SHOPPER_FORCE_HEURISTIC=1 for fast, free local iteration (skips
every LLM call); leave unset (with an API key present) to also validate the
LLM reranker path occasionally.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    start = time.time()
    subprocess.run(
        [sys.executable, "-m", "evaluator.local_evaluator",
         "--catalog", args.catalog, "--dataset", args.dataset, "--output", args.output],
        cwd=ROOT, check=True,
    )
    elapsed = time.time() - start

    result = json.loads((ROOT / args.output).read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in (ROOT / args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {s["sample_id"]: s for s in samples}

    breakdown: dict[str, list[dict]] = defaultdict(list)
    for session in result["sessions"]:
        sample = by_id.get(session["sample_id"], {})
        for key in ("difficulty_bucket", "category_bucket"):
            if sample.get(key):
                breakdown[f"{key}:{sample[key]}"].append(session)

    def summarize(sessions: list[dict]) -> dict:
        if not sessions:
            return {}
        hit_rate = sum(s["hit"] for s in sessions) / len(sessions)
        mrr = sum(s["reciprocal_rank"] for s in sessions) / len(sessions)
        return {"n": len(sessions), "hit_rate_at_10": round(hit_rate, 4), "mrr": round(mrr, 4)}

    extra_breakdown = {k: summarize(v) for k, v in sorted(breakdown.items())}

    print(f"\n=== Agent Shopper local eval ({elapsed:.1f}s) ===")
    print(f"TechnicalScore: {result['recommended_technical_score']:.4f}  "
          f"HitRate@10: {result['hit_rate_at_10']:.4f}  MRR: {result['mrr']:.4f}  MTTC: {result['mttc']:.2f}")
    print("By scenario_type:")
    for name, metrics in result["scenario_metrics"].items():
        print(f"  {name:16s} n={metrics['sample_count']:3d}  hit_rate={metrics['hit_rate_at_10']:.4f}  "
              f"mrr={metrics['mrr']:.4f}  mttc={metrics['mttc']:.2f}")
    print("By difficulty/category bucket:")
    for name, metrics in extra_breakdown.items():
        print(f"  {name:24s} {metrics}")

    log_path = ROOT / "eval_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "technical_score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "scenario_metrics": result["scenario_metrics"],
            "bucket_breakdown": extra_breakdown,
        }) + "\n")
    print(f"\nAppended run to {log_path}")


if __name__ == "__main__":
    main()

"""The actual LLM-reranker-vs-heuristic benchmark: runs the real LLM
reranker path --runs times (default 3) over the full 200-session public set,
plus an optional single heuristic-only baseline pass, and reports mean/
stdev/min/max of TechnicalScore and friends across runs -- not a single-shot
A/B, because the LLM path is non-deterministic (unlike every other change in
this repo's "What we tried" history) and a single run can't distinguish a
real improvement from getting lucky on a few close calls.

Also reports, per run and per scenario_type (especially intent_override):
  - what fraction of turns actually invoked the LLM vs. fell back to
    heuristic, broken down by *why* (tight_pool/clarify_skip/last_turn/
    circuit_breaker/no_provider/eligible) -- so a win or loss is
    attributable to the right mechanism, not just "the LLM path" in the
    abstract, which might be mostly-heuristic-with-occasional-LLM.
  - LLM call outcomes (success/failed), with failures further broken down
    by cause (see llm_client.LLMUnavailable.cause_type).
  - real token usage and an estimated $ cost.

Run scripts/estimate_llm_cost.py first to sanity-check the projected spend
before running this for real.

Usage:
    python3 scripts/run_llm_benchmark.py --runs 3 --baseline --label "llm-vs-heuristic-2026-08-30"
    python3 scripts/run_llm_benchmark.py --runs 1 --sample 20   # fast/cheap dev-loop iteration on the harness itself
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
from agent_shopper.dialog_state import SessionState  # noqa: E402
from agent_shopper.llm_client import active_provider, resolve_model  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402

# Kept in sync with scripts/estimate_llm_cost.py's DEFAULT_PRICING by hand --
# not imported from there since scripts/ isn't a package in this repo (every
# script here is a standalone tool, matching the existing convention). $ per
# 1K tokens; verify against the provider's current pricing page.
DEFAULT_PRICING = {
    "anthropic": {"input_per_1k": 0.0010, "output_per_1k": 0.0050},  # Claude Haiku 4.5
    "openai": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},  # gpt-4o-mini
}

HEADLINE_METRICS = ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")
SCENARIO_METRICS = ("hit_rate_at_10", "mrr", "mttc")


def _new_bucket() -> dict:
    return {"route_reason": Counter(), "llm_outcome": Counter(), "llm_failure_reason": Counter(),
            "prompt_tokens": 0, "completion_tokens": 0}


def _make_hook():
    breakdown = {"overall": _new_bucket(), "by_scenario": defaultdict(_new_bucket)}

    def hook(sample: dict, state: SessionState) -> None:
        scenario = sample["scenario_type"]
        for entry in state.engine_trace:
            for bucket in (breakdown["overall"], breakdown["by_scenario"][scenario]):
                bucket["route_reason"][entry.route_reason] += 1
                if entry.llm_outcome:
                    bucket["llm_outcome"][entry.llm_outcome] += 1
                    if entry.llm_outcome == "failed" and entry.llm_failure_reason:
                        bucket["llm_failure_reason"][entry.llm_failure_reason] += 1
                bucket["prompt_tokens"] += entry.prompt_tokens
                bucket["completion_tokens"] += entry.completion_tokens

    return breakdown, hook


def _bucket_to_json(bucket: dict) -> dict:
    return {
        "route_reason": dict(bucket["route_reason"]),
        "llm_outcome": dict(bucket["llm_outcome"]),
        "llm_failure_reason": dict(bucket["llm_failure_reason"]),
        "prompt_tokens": bucket["prompt_tokens"],
        "completion_tokens": bucket["completion_tokens"],
    }


def _engine_breakdown_to_json(breakdown: dict) -> dict:
    return {
        "overall": _bucket_to_json(breakdown["overall"]),
        "by_scenario": {name: _bucket_to_json(b) for name, b in sorted(breakdown["by_scenario"].items())},
    }


def _print_engine_breakdown(breakdown: dict) -> None:
    overall = breakdown["overall"]
    total = sum(overall["route_reason"].values())
    eligible = overall["route_reason"].get("eligible", 0)
    print(f"  Engine routing ({total} turns, {eligible} eligible for LLM = {eligible / total:.1%}):")
    for reason, count in sorted(overall["route_reason"].items(), key=lambda kv: -kv[1]):
        print(f"    {reason:16s} {count:5d}")
    if overall["llm_outcome"]:
        print(f"  LLM call outcomes: {dict(overall['llm_outcome'])}")
    if overall["llm_failure_reason"]:
        print(f"  LLM failure reasons: {dict(overall['llm_failure_reason'])}")
    print(f"  Tokens this run: {overall['prompt_tokens']:,} prompt + {overall['completion_tokens']:,} completion")


def _run_once(agent: Agent, samples: list[dict], catalog_ids, categories, products, *, force_heuristic: bool) -> tuple[dict, dict, float]:
    breakdown, hook = _make_hook()
    start = time.time()
    ctx = patch.object(dialog_policy_mod, "FORCE_HEURISTIC", True) if force_heuristic else _nullcontext()
    with ctx:
        result = evaluate(agent, samples, catalog_ids, categories, products, session_hook=hook)
    elapsed = time.time() - start
    return result, breakdown, elapsed


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _estimate_cost(breakdown: dict, price_in: float, price_out: float) -> float:
    overall = breakdown["overall"]
    return (overall["prompt_tokens"] / 1000) * price_in + (overall["completion_tokens"] / 1000) * price_out


def _aggregate(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample", type=int, default=None, help="use only the first N sessions -- fast/cheap dev-loop iteration only; omit for the trusted full-200 number")
    parser.add_argument("--label", default="")
    parser.add_argument("--baseline", action="store_true", help="also run one heuristic-only pass (deterministic, so one run suffices) for a direct before/after delta")
    parser.add_argument("--price-input-per-1k", type=float, default=None)
    parser.add_argument("--price-output-per-1k", type=float, default=None)
    args = parser.parse_args()

    provider = active_provider()
    if provider is None:
        print("ERROR: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured -- this benchmark needs a real "
              "LLM provider to be meaningful. Set one and re-run.")
        raise SystemExit(1)

    price_in = args.price_input_per_1k
    price_out = args.price_output_per_1k
    if price_in is None or price_out is None:
        defaults = DEFAULT_PRICING.get(provider, DEFAULT_PRICING["anthropic"])
        price_in = price_in if price_in is not None else defaults["input_per_1k"]
        price_out = price_out if price_out is not None else defaults["output_per_1k"]

    model = resolve_model() if provider == "openai" else resolve_model("AGENT_SHOPPER_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    print(f"Provider: {provider}  Model: {model}")

    samples = load_jsonl(ROOT / args.dataset)
    if args.sample:
        samples = samples[: args.sample]
    catalog_ids, categories, products = catalog_index(ROOT / args.catalog)
    agent = Agent(str(ROOT / args.catalog))

    # --- Startup guard: confirm the LLM path actually works before committing
    # to the full N-run cost -- a silently-broken key/model would otherwise
    # produce an all-heuristic "LLM benchmark" with a meaningless zero-variance
    # result that looks superficially plausible.
    print("\nStartup check: confirming at least one real LLM call succeeds...")
    # A single session can easily land on zero LLM-eligible turns by chance
    # (clarify_skip/tight_pool/last_turn) -- that's not a probe failure, just
    # bad luck. Use a handful of sessions so an eligible turn is all but
    # guaranteed (~24% of turns are eligible on this dataset), and only treat
    # "had eligible turns but every one failed" as the real red flag.
    probe_n = min(8, len(samples))
    _, probe_breakdown, _ = _run_once(agent, samples[:probe_n], catalog_ids, categories, products, force_heuristic=False)
    probe_eligible = probe_breakdown["overall"]["route_reason"].get("eligible", 0)
    probe_success = probe_breakdown["overall"]["llm_outcome"].get("success", 0)
    if probe_eligible == 0:
        print(f"  No LLM-eligible turns turned up in the first {probe_n} sessions (all clarify_skip/tight_pool/"
              f"last_turn) -- inconclusive, not a failure. Proceeding; the real cost is unaffected by this probe.")
    elif probe_success == 0:
        print("ERROR: every LLM-eligible turn in the probe failed (check API key validity, model name, network "
              "access). Aborting before spending on the full run. Probe breakdown:")
        _print_engine_breakdown(probe_breakdown)
        raise SystemExit(1)
    else:
        print(f"  OK -- {probe_success}/{probe_eligible} probe LLM calls succeeded.\n")

    log_path = ROOT / "llm_benchmark_runs.jsonl"
    run_records = []

    print(f"=== LLM reranker benchmark: {args.runs} run(s) over {len(samples)} sessions ===")
    for run_index in range(1, args.runs + 1):
        result, breakdown, elapsed = _run_once(agent, samples, catalog_ids, categories, products, force_heuristic=False)
        cost = _estimate_cost(breakdown, price_in, price_out)
        print(f"\n--- Run {run_index}/{args.runs} ({elapsed:.0f}s) ---")
        print(f"  TechnicalScore={result['recommended_technical_score']:.4f}  "
              f"HitRate@10={result['hit_rate_at_10']:.4f}  MRR={result['mrr']:.4f}  MTTC={result['mttc']:.2f}")
        _print_engine_breakdown(breakdown)
        print(f"  Estimated cost this run: ${cost:.2f}")

        record = {
            "label": args.label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_index": run_index, "elapsed_seconds": round(elapsed, 1),
            "technical_score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"], "mrr": result["mrr"], "mttc": result["mttc"],
            "scenario_metrics": result["scenario_metrics"],
            "engine_breakdown": _engine_breakdown_to_json(breakdown),
            "token_usage": {"prompt_tokens": breakdown["overall"]["prompt_tokens"],
                             "completion_tokens": breakdown["overall"]["completion_tokens"]},
            "estimated_cost_usd": round(cost, 4),
        }
        run_records.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # --- Aggregate across runs -----------------------------------------------
    print(f"\n=== Aggregate across {args.runs} run(s) ===")
    aggregate = {"label": args.label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "aggregate": True,
                 "n_runs": args.runs, "headline": {}, "by_scenario": {}}
    for metric in HEADLINE_METRICS:
        values = [r[metric] if metric != "recommended_technical_score" else r["technical_score"] for r in run_records]
        agg = _aggregate(values)
        aggregate["headline"][metric] = agg
        print(f"  {metric:28s} mean={agg['mean']:.4f}  stdev={agg['stdev']:.4f}  min={agg['min']:.4f}  max={agg['max']:.4f}")

    scenario_names = sorted(run_records[0]["scenario_metrics"].keys()) if run_records else []
    for scenario in scenario_names:
        aggregate["by_scenario"][scenario] = {}
        print(f"\n  {scenario}:")
        for metric in SCENARIO_METRICS:
            values = [r["scenario_metrics"][scenario][metric] for r in run_records]
            agg = _aggregate(values)
            aggregate["by_scenario"][scenario][metric] = agg
            print(f"    {metric:12s} mean={agg['mean']:.4f}  stdev={agg['stdev']:.4f}  min={agg['min']:.4f}  max={agg['max']:.4f}")

    total_cost = sum(r["estimated_cost_usd"] for r in run_records)
    aggregate["total_estimated_cost_usd"] = round(total_cost, 4)
    print(f"\n  Total estimated cost across {args.runs} run(s): ${total_cost:.2f}")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(aggregate) + "\n")
    print(f"\nAppended {args.runs} run(s) + 1 aggregate record to {log_path}")

    # --- Optional heuristic-only baseline ------------------------------------
    if args.baseline:
        print("\n=== Baseline: heuristic-only (deterministic, one run + a repeat-run regression guard) ===")
        baseline_result, baseline_breakdown, baseline_elapsed = _run_once(
            agent, samples, catalog_ids, categories, products, force_heuristic=True,
        )
        # Phase 6 regression guard: heuristic-only must be byte-identical
        # run-to-run (no unseeded randomness anywhere in the pipeline -- see
        # the plan's Phase 6). If this ever fails, something introduced a
        # nondeterminism that would silently confound the LLM-run variance
        # measured above with an unrelated source of noise.
        repeat_result, _, _ = _run_once(agent, samples, catalog_ids, categories, products, force_heuristic=True)
        assert repeat_result["recommended_technical_score"] == baseline_result["recommended_technical_score"], (
            "Heuristic-only baseline was NOT byte-identical across two runs -- an unseeded source of "
            "nondeterminism has crept into the pipeline. This invalidates the assumption that the LLM run's "
            "variance (measured above) reflects LLM stochasticity alone. Investigate before trusting the "
            "aggregate numbers above."
        )
        print(f"  TechnicalScore={baseline_result['recommended_technical_score']:.4f}  "
              f"HitRate@10={baseline_result['hit_rate_at_10']:.4f}  MRR={baseline_result['mrr']:.4f}  "
              f"MTTC={baseline_result['mttc']:.2f}  ({baseline_elapsed:.0f}s)")
        print("  Regression guard passed: repeat heuristic-only run was byte-identical.")

        print("\n=== Before -> After (heuristic baseline -> mean of LLM runs) ===")
        for metric in HEADLINE_METRICS:
            baseline_value = baseline_result[metric] if metric != "recommended_technical_score" else baseline_result["recommended_technical_score"]
            llm_mean = aggregate["headline"][metric]["mean"]
            print(f"  {metric:28s} {baseline_value:.4f} -> {llm_mean:.4f}  (Δ {llm_mean - baseline_value:+.4f})")
        for scenario in scenario_names:
            print(f"\n  {scenario}:")
            for metric in SCENARIO_METRICS:
                baseline_value = baseline_result["scenario_metrics"][scenario][metric]
                llm_mean = aggregate["by_scenario"][scenario][metric]["mean"]
                print(f"    {metric:12s} {baseline_value:.4f} -> {llm_mean:.4f}  (Δ {llm_mean - baseline_value:+.4f})")

        baseline_record = {
            "label": f"{args.label}-baseline" if args.label else "baseline", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baseline": True, "technical_score": baseline_result["recommended_technical_score"],
            "hit_rate_at_10": baseline_result["hit_rate_at_10"], "mrr": baseline_result["mrr"], "mttc": baseline_result["mttc"],
            "scenario_metrics": baseline_result["scenario_metrics"], "elapsed_seconds": round(baseline_elapsed, 1),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(baseline_record) + "\n")


if __name__ == "__main__":
    main()

"""Estimates the $ cost of an LLM-reranker benchmark run *before* spending
anything real -- see scripts/run_llm_benchmark.py, the actual benchmark this
feeds into.

Two-phase design:

  1. Free "attempt count" dry run: monkeypatches reranker_mod.call_structured
     to return an instant, synthetic-but-valid response (zero network cost,
     zero latency) for every LLM-eligible turn, then runs the real
     evaluate() loop over the full public set. This is not an approximation
     of the routing logic -- it *is* the routing logic (orchestrator.
     decide_rerank_engine), so the resulting per-scenario_type attempt
     counts are exact, not estimated. Crucially, the synthetic response
     always "succeeds" (never raises LLMUnavailable), so the circuit
     breaker never trips during the dry run -- if it did, a session's later
     turns would silently fall back to "circuit_breaker" instead of
     "eligible", undercounting the very thing we're trying to measure. A
     *real* run can still trip the breaker occasionally; that's a separate,
     already-tracked outcome (see run_llm_benchmark.py's engine_breakdown),
     not something this estimator should bake into its projection.

  2. A small number of REAL calls (--real-call-sample, default 20) against a
     stratified slice of the dataset, to get empirical mean/stdev
     prompt/completion tokens per call -- more trustworthy than statically
     counting characters in the system prompt + candidate summaries, since
     it captures real structured-output overhead too. Capped by a
     call-counting wrapper around call_structured so cost never exceeds the
     requested sample size regardless of how many sessions get processed.

Cost projection = mean_tokens_per_call * attempt_count(scaled to --runs) *
price_per_token. Pricing defaults below are what scripts/run_llm_benchmark.py
and this file were built against (Anthropic Claude Haiku 4.5 pricing table
cached 2026-06-24; OpenAI gpt-4o-mini pricing from the provider's public
pricing page as of 2026-08) -- ALWAYS pass --price-input-per-1k/
--price-output-per-1k explicitly, or at minimum re-verify these against the
provider's current pricing page, before trusting a projection that's about
to inform real spend.

Usage:
    python3 scripts/estimate_llm_cost.py [--runs 3] [--real-call-sample 20]
        [--catalog data/catalog.jsonl] [--dataset data/public_set.jsonl]
        [--price-input-per-1k P] [--price-output-per-1k P]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
import agent_shopper.reranker as reranker_mod  # noqa: E402
from agent_shopper.dialog_state import SessionState  # noqa: E402
from agent_shopper.llm_client import LLMUnavailable, TokenUsage, active_provider, resolve_model  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402

# $ per 1K tokens. See module docstring -- verify against the provider's
# current pricing page before a large real run; these are a starting point,
# not a guarantee.
DEFAULT_PRICING = {
    "anthropic": {"input_per_1k": 0.0010, "output_per_1k": 0.0050},  # Claude Haiku 4.5
    "openai": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},  # gpt-4o-mini
}


def _fake_success_call_structured(system_prompt, payload, schema):
    """Zero-cost stand-in for llm_client.call_structured: returns an
    instantly-valid response for every candidate, so every LLM-eligible turn
    "succeeds" and the circuit breaker never trips -- see module docstring
    for why that matters for an accurate attempt count."""
    judgments = [
        reranker_mod.Judgment(index=c["index"], relevance_score=0.5)
        for c in payload["candidates"]
    ]
    return schema(judgments=judgments), TokenUsage()


def _budget_capped_call_structured(cap: int):
    """Wraps the real call_structured, allowing at most `cap` real dispatches
    -- once reached, further LLM-eligible turns raise LLMUnavailable (falling
    back to heuristic, same as any other LLM failure) so a real run can never
    exceed the requested sample size regardless of dataset size."""
    real_call_structured = reranker_mod.call_structured
    count = 0

    def wrapper(system_prompt, payload, schema):
        nonlocal count
        if count >= cap:
            raise LLMUnavailable("estimate_llm_cost sample budget exhausted", cause_type="budget_cap")
        count += 1
        return real_call_structured(system_prompt, payload, schema)

    return wrapper


def _make_hook(tally: dict[str, Counter], usage_rows: list[tuple[int, int]]):
    def hook(sample: dict, state: SessionState) -> None:
        scenario = sample["scenario_type"]
        for entry in state.engine_trace:
            tally["overall"][entry.route_reason] += 1
            tally[scenario][entry.route_reason] += 1
            if entry.llm_outcome == "success":
                usage_rows.append((entry.prompt_tokens, entry.completion_tokens))

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--runs", type=int, default=3, help="how many repeated benchmark runs to project cost for")
    parser.add_argument("--real-call-sample", type=int, default=20, help="number of REAL (paid) calls to sample for token stats")
    parser.add_argument("--price-input-per-1k", type=float, default=None)
    parser.add_argument("--price-output-per-1k", type=float, default=None)
    args = parser.parse_args()

    provider = active_provider()
    if provider is None:
        print("No OPENAI_API_KEY or ANTHROPIC_API_KEY configured -- cannot estimate a real run's cost.")
        print("(The dry-run attempt count below is still accurate and provider-independent.)")

    price_in = args.price_input_per_1k
    price_out = args.price_output_per_1k
    if price_in is None or price_out is None:
        defaults = DEFAULT_PRICING.get(provider or "anthropic")
        price_in = price_in if price_in is not None else defaults["input_per_1k"]
        price_out = price_out if price_out is not None else defaults["output_per_1k"]

    samples = load_jsonl(ROOT / args.dataset)
    catalog_ids, categories, products = catalog_index(ROOT / args.catalog)
    agent = Agent(str(ROOT / args.catalog))

    # --- Phase 1: free, exact attempt count over the full dataset ---------
    print(f"\n=== Phase 1: dry-run attempt count ({len(samples)} sessions, zero cost) ===")
    tally: dict[str, Counter] = defaultdict(Counter)
    dry_usage_rows: list[tuple[int, int]] = []
    start = time.time()
    # Force `llm_available` True in decide_rerank_engine regardless of
    # whether a real provider key is actually configured -- otherwise every
    # turn would route to "no_provider" instead of "eligible" whenever this
    # is run without a key, defeating the entire point of a free,
    # provider-independent attempt count. Phase 2 below still requires (and
    # checks for) a real key, since it makes real paid calls.
    with patch.object(reranker_mod, "call_structured", _fake_success_call_structured), \
            patch.object(dialog_policy_mod, "active_provider", return_value="openai"):
        evaluate(agent, samples, catalog_ids, categories, products, session_hook=_make_hook(tally, dry_usage_rows))
    elapsed = time.time() - start
    print(f"  ({elapsed:.0f}s)")

    overall = tally["overall"]
    total_turns = sum(overall.values())
    eligible = overall.get("eligible", 0)
    print(f"\n  Overall: {total_turns} turns, {eligible} eligible for an LLM call ({eligible / total_turns:.1%})")
    for reason, count in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:16s} {count:5d}  ({count / total_turns:.1%})")
    print("\n  Eligible-turn rate by scenario_type:")
    for scenario in sorted(tally):
        if scenario == "overall":
            continue
        c = tally[scenario]
        n = sum(c.values())
        print(f"    {scenario:16s} {c.get('eligible', 0):4d} / {n:4d} eligible ({c.get('eligible', 0) / n:.1%})")

    if provider is None:
        print("\nSkipping Phase 2 (real token sampling) and cost projection -- no provider configured.")
        return

    # --- Phase 2: a small number of REAL calls for empirical token stats --
    print(f"\n=== Phase 2: real-call token sample (up to {args.real_call_sample} real calls, provider={provider}) ===")
    real_tally: dict[str, Counter] = defaultdict(Counter)
    real_usage_rows: list[tuple[int, int]] = []
    start = time.time()
    with patch.object(reranker_mod, "call_structured", _budget_capped_call_structured(args.real_call_sample)):
        evaluate(agent, samples, catalog_ids, categories, products, session_hook=_make_hook(real_tally, real_usage_rows))
    elapsed = time.time() - start
    print(f"  ({elapsed:.0f}s, {len(real_usage_rows)} real calls succeeded)")

    if not real_usage_rows:
        print("\n  No real calls succeeded -- check the configured API key/model. Cannot project cost.")
        return

    prompt_tokens = [p for p, _ in real_usage_rows]
    completion_tokens = [c for _, c in real_usage_rows]
    mean_prompt = statistics.fmean(prompt_tokens)
    mean_completion = statistics.fmean(completion_tokens)
    stdev_prompt = statistics.stdev(prompt_tokens) if len(prompt_tokens) > 1 else 0.0
    stdev_completion = statistics.stdev(completion_tokens) if len(completion_tokens) > 1 else 0.0
    print(f"\n  Empirical tokens/call: prompt {mean_prompt:.0f} (±{stdev_prompt:.0f})  completion {mean_completion:.0f} (±{stdev_completion:.0f})")
    print(f"  Resolved model: {resolve_model() if provider == 'openai' else resolve_model('AGENT_SHOPPER_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}")

    # --- Projection ---------------------------------------------------------
    projected_calls = eligible * args.runs
    projected_prompt_tokens = projected_calls * mean_prompt
    projected_completion_tokens = projected_calls * mean_completion
    projected_cost = (projected_prompt_tokens / 1000) * price_in + (projected_completion_tokens / 1000) * price_out

    print(f"\n=== Projection for --runs {args.runs} over the full {len(samples)}-session set ===")
    print(f"  Pricing used: ${price_in:.5f}/1K input, ${price_out:.5f}/1K output (verify against the provider's current pricing page)")
    print(f"  Projected eligible LLM calls: {eligible} eligible/run * {args.runs} runs = {projected_calls}")
    print(f"  Projected tokens: {projected_prompt_tokens:,.0f} prompt + {projected_completion_tokens:,.0f} completion")
    print(f"  Projected cost: ${projected_cost:.2f}")
    print("\n  Note: not every turn calls the LLM -- see the route_reason breakdown above "
          "(clarify_skip/last_turn/tight_pool/circuit_breaker all take the free heuristic path).")


if __name__ == "__main__":
    main()

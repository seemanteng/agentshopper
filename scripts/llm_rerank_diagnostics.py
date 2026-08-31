"""Two diagnostics for the LLM reranker that a plain TechnicalScore
comparison can't surface on its own -- both real, paid LLM calls, kept small
and bounded on purpose:

  --dump-trace: writes every (payload sent, judgments received, top pick) for
      a sample of real reranker calls to a gitignored jsonl file, for a
      manual skim of whether the model is getting swayed by catalog text
      that reads like an instruction (the system prompt already tells it to
      treat title/features/description as untrusted data -- this is the
      first real chance to check that holds under load, not just assume it).
      Always includes every intent_override session (highest-risk track --
      most prone to contradictory-history-in-prompt confusion) plus a
      stratified sample of the rest.

  --position-bias: measures the LLM's sensitivity to candidate *order*,
      independent of content -- a well-documented failure mode in listwise
      LLM reranking ("lost in the middle"/primacy-recency bias), and
      something this codebase has never checked because candidates always
      arrive in fused-RRF order (see reranker.py). Captures real
      (ctx, candidates) pairs from one real pass, then replays a bounded
      sample of those exact turns through a second LLMReranker call with
      candidates shuffled by --shuffle-seed, and reports how often the top
      pick changes. Replays off frozen in-memory objects (not two full
      parallel session runs) specifically to avoid a different confound: a
      full shuffled-vs-unshuffled session replay would diverge after the
      first turn where the shuffled top pick differs (a different top pick
      changes what's shown, which changes the soft repeat-penalty and can
      change which turn hits/clarifies), contaminating later-turn
      comparisons with conversational drift instead of pure order effects.
      ALSO replays the same calls a second time with the original
      (unshuffled) order as a stochasticity control -- gpt-4o-mini is called
      with no temperature/seed pinning, so two independent real calls on
      identical input can legitimately disagree from ordinary sampling noise
      alone. The number that actually means something is (shuffled flip rate
      - control flip rate), not the raw shuffled flip rate.

shuffle_seed is a constructor-param-only feature on LLMReranker (see
reranker.py) -- inert in production, only ever set here.

Usage:
    python3 scripts/llm_rerank_diagnostics.py --dump-trace --trace-sample 40
    python3 scripts/llm_rerank_diagnostics.py --position-bias --shuffle-seed 7 --replay-sample 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.reranker as reranker_mod  # noqa: E402
from agent_shopper.dialog_state import SessionState  # noqa: E402
from agent_shopper.llm_client import active_provider  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


class CapturingLLMReranker(reranker_mod.LLMReranker):
    """Same behavior as LLMReranker, plus recording (ctx, candidates,
    payload, response_judgments, top_pick) for every call into a shared
    list -- installed in place of the real class via patch.object, which
    works because dialog_policy does a late `reranker_mod.LLMReranker()`
    module-attribute lookup rather than binding the name at import time."""

    def __init__(self, *args, capture_into: list | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capture_into = capture_into

    def rerank(self, ctx, candidates, top_k):
        ranked = super().rerank(ctx, candidates, top_k)
        if self._capture_into is not None:
            self._capture_into.append({
                "ctx": ctx,
                "candidates": list(candidates),
                "payload": self.last_payload,
                "response_judgments": self.last_response_judgments,
                "top_pick": ranked[0].product.parent_asin if ranked else None,
                "used_llm": self.last_call_used_llm,
            })
        return ranked


def _stratified_sample(samples: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Every intent_override session (highest-risk track) plus a seeded
    random sample of the rest, up to n total."""
    intent_override = [s for s in samples if s["scenario_type"] == "intent_override"]
    rest = [s for s in samples if s["scenario_type"] != "intent_override"]
    remaining = max(0, n - len(intent_override))
    rng = random.Random(seed)
    rest_sample = rng.sample(rest, min(remaining, len(rest)))
    return intent_override + rest_sample


def _tag_captures_by_sample(captures: list[dict]) -> tuple[list[dict], object]:
    """Returns (captures, session_hook) -- the hook stamps sample_id/
    scenario_type/turn_within_session onto every capture appended since the
    last call, exploiting evaluate()'s strictly sequential per-sample
    processing (no concurrency anywhere in this pipeline -- see README/plan
    Phase 6)."""
    cursor = {"start": 0}

    def hook(sample: dict, state: SessionState) -> None:
        end = len(captures)
        for i, entry in enumerate(captures[cursor["start"]:end], start=1):
            entry["sample_id"] = sample["sample_id"]
            entry["scenario_type"] = sample["scenario_type"]
            entry["turn_within_session"] = i
        cursor["start"] = end

    return captures, hook


def run_dump_trace(agent: Agent, samples: list[dict], catalog_ids, categories, products, args) -> None:
    subset = _stratified_sample(samples, args.trace_sample)
    print(f"\n=== --dump-trace: {len(subset)} sessions "
          f"({sum(1 for s in subset if s['scenario_type'] == 'intent_override')} intent_override + rest) ===")
    captures, hook = _tag_captures_by_sample([])
    with patch.object(reranker_mod, "LLMReranker", lambda *a, **kw: CapturingLLMReranker(*a, capture_into=captures, **kw)):
        evaluate(agent, subset, catalog_ids, categories, products, session_hook=hook)

    out_path = ROOT / "llm_rerank_trace_sample.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for entry in captures:
            if not entry["used_llm"]:
                continue  # fell back to heuristic this call -- nothing to skim
            handle.write(json.dumps({
                "sample_id": entry["sample_id"], "scenario_type": entry["scenario_type"],
                "turn_within_session": entry["turn_within_session"],
                "payload": entry["payload"], "response_judgments": entry["response_judgments"],
                "top_pick": entry["top_pick"],
            }) + "\n")
            written += 1
    print(f"Wrote {written} real LLM-call traces to {out_path}")
    print("Manually skim this file for signs the model's relevance_score/top_pick was swayed by "
          "catalog title/features/description text that reads like an instruction, rather than by "
          "genuine relevance to shopper_context -- focus on intent_override entries first.")


def run_position_bias(agent: Agent, samples: list[dict], catalog_ids, categories, products, args) -> None:
    subset = _stratified_sample(samples, max(args.replay_sample, args.trace_sample))
    print(f"\n=== --position-bias: capturing baseline over {len(subset)} sessions ===")
    baseline_captures, hook = _tag_captures_by_sample([])
    with patch.object(reranker_mod, "LLMReranker", lambda *a, **kw: CapturingLLMReranker(*a, capture_into=baseline_captures, **kw)):
        evaluate(agent, subset, catalog_ids, categories, products, session_hook=hook)

    real_calls = [c for c in baseline_captures if c["used_llm"]]
    print(f"  {len(real_calls)} real baseline calls captured")

    # Weight the replay sample toward intent_override, matching --dump-trace's priority.
    intent_override_calls = [c for c in real_calls if c["scenario_type"] == "intent_override"]
    rest_calls = [c for c in real_calls if c["scenario_type"] != "intent_override"]
    rng = random.Random(args.shuffle_seed)
    n_rest = max(0, args.replay_sample - len(intent_override_calls))
    replay_set = intent_override_calls + rng.sample(rest_calls, min(n_rest, len(rest_calls)))
    replay_set = replay_set[: args.replay_sample]
    print(f"  Replaying {len(replay_set)} of those calls with candidates shuffled (seed={args.shuffle_seed})...")

    # Control: replay the SAME calls with candidates left in their original
    # (unshuffled) order. gpt-4o-mini is called with no temperature/seed
    # pinning (see llm_client._call_openai), so two independent real calls on
    # identical input can legitimately return different judgments/top picks
    # from ordinary sampling noise alone -- without this control, a shuffled
    # flip rate can't be told apart from plain call-to-call inconsistency.
    # The *attributable* order effect is (shuffled flip rate - control flip
    # rate), not the raw shuffled flip rate on its own.
    def _replay(shuffle_seed: int | None) -> dict[str, list[int]]:
        by_scenario: dict[str, list[int]] = {}
        for call in replay_set:
            rr = CapturingLLMReranker(shuffle_seed=shuffle_seed)
            ranked = rr.rerank(call["ctx"], call["candidates"], top_k=len(call["candidates"]))
            if not rr.last_call_used_llm:
                continue  # real call failed on replay -- not a data point
            new_top = ranked[0].product.parent_asin if ranked else None
            by_scenario.setdefault(call["scenario_type"], []).append(int(new_top != call["top_pick"]))
        return by_scenario

    print("  Replaying again with the SAME (unshuffled) order -- stochasticity control...")
    control_by_scenario = _replay(None)
    print(f"  Replaying with candidates shuffled (seed={args.shuffle_seed})...")
    shuffled_by_scenario = _replay(args.shuffle_seed)

    def _rate(by_scenario: dict[str, list[int]]) -> tuple[int, int]:
        flips = sum(sum(v) for v in by_scenario.values())
        total = sum(len(v) for v in by_scenario.values())
        return flips, total

    control_flips, control_total = _rate(control_by_scenario)
    shuffled_flips, shuffled_total = _rate(shuffled_by_scenario)
    if control_total == 0 or shuffled_total == 0:
        print("  Not enough replay calls succeeded -- nothing to report.")
        return

    control_rate = control_flips / control_total
    shuffled_rate = shuffled_flips / shuffled_total
    print(f"\n=== Control (same order, repeat call): {control_flips}/{control_total} ({control_rate:.1%}) ===")
    for scenario, results in sorted(control_by_scenario.items()):
        print(f"  {scenario:16s} {sum(results):3d}/{len(results):3d}  ({sum(results) / len(results):.1%})")
    print(f"\n=== Shuffled (candidates reordered): {shuffled_flips}/{shuffled_total} ({shuffled_rate:.1%}) ===")
    for scenario, results in sorted(shuffled_by_scenario.items()):
        print(f"  {scenario:16s} {sum(results):3d}/{len(results):3d}  ({sum(results) / len(results):.1%})")

    attributable = shuffled_rate - control_rate
    print(f"\n=== Attributable order effect (shuffled - control): {attributable:+.1%} ===")
    if control_rate > 0.5:
        print("  NOTE: the control (same-order repeat call) flip rate is itself very high -- this points to "
              "the LLM reranker's top pick being highly unstable call-to-call regardless of order (likely no "
              "temperature/seed pinning + many close-relevance candidates), which is arguably a bigger concern "
              "for production trustworthiness than order sensitivity specifically.")
    elif attributable > 0.15:
        print("  FLAG: candidate reordering changes the top pick meaningfully more often than plain repeat-call "
              "noise does -- the LLM reranker shows real position sensitivity independent of content.")
    else:
        print("  Shuffled and control flip rates are close -- most of the raw flip rate is call-to-call "
              "stochasticity, not a genuine order effect.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--dump-trace", action="store_true")
    parser.add_argument("--position-bias", action="store_true")
    parser.add_argument("--trace-sample", type=int, default=40, help="sessions for --dump-trace (always includes every intent_override session)")
    parser.add_argument("--replay-sample", type=int, default=40, help="calls to replay for --position-bias")
    parser.add_argument("--shuffle-seed", type=int, default=7)
    args = parser.parse_args()

    if not args.dump_trace and not args.position_bias:
        parser.error("pass --dump-trace and/or --position-bias")

    if active_provider() is None:
        print("ERROR: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured -- both diagnostics need real LLM calls.")
        raise SystemExit(1)

    samples = load_jsonl(ROOT / args.dataset)
    catalog_ids, categories, products = catalog_index(ROOT / args.catalog)
    agent = Agent(str(ROOT / args.catalog))

    if args.dump_trace:
        run_dump_trace(agent, samples, catalog_ids, categories, products, args)
    if args.position_bias:
        run_position_bias(agent, samples, catalog_ids, categories, products, args)


if __name__ == "__main__":
    main()

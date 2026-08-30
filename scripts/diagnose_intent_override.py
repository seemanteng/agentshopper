"""Diagnostic: for intent_override sessions where the target ASIN was NEVER
recalled into any turn's candidate pool (the 63% majority of that track's
misses -- see scripts/diagnose_retrieval.py's per-scenario breakdown), trace
every retrieve() call and slot state turn-by-turn to find out *why*
retrieval never surfaces it, rather than guessing.

Read-only with respect to agent_shopper/: agent_shopper.retrieval.retrieve
is monkeypatched only for the duration of this script's own run, purely to
record its arguments/result before delegating to the real implementation
unchanged (same technique as scripts/diagnose_retrieval.py).

Usage:
    python3 scripts/diagnose_intent_override.py [--catalog data/catalog.jsonl] [--dataset data/public_set.jsonl]
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), so add the repo root ourselves

import agent_shopper.dialog_policy as dialog_policy_mod  # noqa: E402
import agent_shopper.retrieval as retrieval_mod  # noqa: E402
from agent_shopper.models import SlotSet  # noqa: E402
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


def _slots_repr(slots: SlotSet) -> str:
    parts = []
    for f in fields(slots):
        value = getattr(slots, f.name)
        if f.name == "hard_marked":
            if value:
                parts.append(f"hard_marked={sorted(value)}")
            continue
        if value not in (None, [], ()):
            parts.append(f"{f.name}={value!r}")
    return ", ".join(parts) if parts else "(empty)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = [s for s in load_jsonl(ROOT / args.dataset) if s["scenario_type"] == "intent_override"]
    catalog_ids, categories, products = catalog_index(ROOT / args.catalog)
    agent = Agent(str(ROOT / args.catalog))

    # sample_id -> ordered list of retrieve() call records for that session.
    traces: dict[str, list[dict]] = {}
    current: list[object] = [None, None, None, None]  # [sample_id, turn, target, override_applied]
    real_retrieve = retrieval_mod.retrieve
    recall_acc = new_recall_accumulator()  # only accumulated for override_applied turns -- see below
    recall_acc_n = 0
    asin_index_cache: dict[int, dict[str, int]] = {}

    def recording_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit=200, dense_index=None):
        nonlocal recall_acc_n
        result = real_retrieve(catalog, bm25_index, tfidf_index, query_text, effective_slots, plan, limit, dense_index)
        sample_id, turn, target, override_applied = current
        if sample_id is not None:
            target_in_pool = any(c.product.parent_asin == target for c in result.candidates)
            traces.setdefault(sample_id, []).append({
                "turn": turn,
                "query_text": query_text,
                "slots": _slots_repr(effective_slots),
                "gate_to_category": plan.gate_to_category,
                "hard_filter_slots": plan.hard_filter_slots,
                "pool_size": result.pool_size,
                "target_in_pool": target_in_pool,
            })
            # Only turns after the override message was actually sent are
            # diagnostic of the override problem -- a pre-override turn
            # recalling the (not-yet-relevant) eventual target is incidental,
            # same gating the hit/recall-turn tracking below already applies.
            if override_applied:
                cache_key = id(catalog)
                if cache_key not in asin_index_cache:
                    asin_index_cache[cache_key] = build_asin_index(catalog)
                ranks = route_ranks_for_target(
                    catalog, bm25_index, tfidf_index, query_text, effective_slots, plan,
                    target, result.candidates, asin_index_cache[cache_key], dense_index=dense_index,
                )
                accumulate(recall_acc, ranks)
                recall_acc_n += 1
        return result

    sessions_out: list[dict] = []
    with patch.object(retrieval_mod, "retrieve", recording_retrieve):
        for sample in samples:
            session_id = f"diagio_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = False
            user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
            hit_turn: int | None = None
            recall_turn: int | None = None
            override_turn: int | None = None
            messages: list[tuple[int, str]] = []

            for turn in range(1, MAX_TURNS + 1):
                current[0], current[1], current[2], current[3] = sample["sample_id"], turn, target, override_applied
                messages.append((turn, user_message))
                contradiction = dialog_policy_mod._has_contradiction_language(user_message)
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}

                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                turn_calls = [r for r in traces.get(sample["sample_id"], []) if r["turn"] == turn]
                if override_applied and recall_turn is None and any(r["target_in_pool"] for r in turn_calls):
                    recall_turn = turn
                if override_applied and target in ranked:
                    hit_turn = turn
                    break
                if turn == MAX_TURNS:
                    break
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    override_turn = turn + 1
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
                "hit_turn": hit_turn,
                "recall_turn": recall_turn,
                "override_turn": override_turn,
                "messages": messages,
                "contradiction_flags": {t: dialog_policy_mod._has_contradiction_language(m) for t, m in messages},
            })

    never_recalled = [s for s in sessions_out if s["hit_turn"] is None and s["recall_turn"] is None]
    print(f"{len(samples)} intent_override sessions -- {len(never_recalled)} never recalled the target at all.\n")

    report_recall_table(
        recall_acc, recall_acc_n,
        title=f"Per-route Recall@10/50/100 over {recall_acc_n} post-override retrieve() calls "
              f"across {len(samples)} intent_override sessions:",
    )

    for s in never_recalled:
        print("=" * 100)
        print(f"{s['sample_id']}  override_turn={s['override_turn']}")
        for turn, msg in s["messages"]:
            marker = " <-- override" if turn == s["override_turn"] else ""
            contradiction = s["contradiction_flags"][turn]
            print(f"  turn {turn}{marker}  contradiction_detected={contradiction}  message={msg!r}")
            for call in traces.get(s["sample_id"], []):
                if call["turn"] != turn:
                    continue
                print(
                    f"    retrieve(): gate={call['gate_to_category']!s:5s} "
                    f"hard_filter={call['hard_filter_slots']}  pool_size={call['pool_size']:3d}  "
                    f"target_in_pool={call['target_in_pool']}"
                )
                print(f"      query_text={call['query_text']!r}")
                print(f"      slots: {call['slots']}")
        print()


if __name__ == "__main__":
    main()

import unittest

from agent_shopper.config import BUYING_GATE_MIN_SLOTS, MAX_TURNS
from agent_shopper.models import SlotSet
from agent_shopper.orchestrator import (
    decide_rerank_engine,
    decide_routes,
    droppable_slots_by_tier,
    relax_gate,
    should_clarify,
    widen_budget,
)


class DecideRoutesTest(unittest.TestCase):
    def test_buying_with_enough_slots_gates(self) -> None:
        slots = SlotSet(**{f: "x" for f in ("color", "size", "material")[:BUYING_GATE_MIN_SLOTS]})
        plan, _ = decide_routes("buying", slots)
        self.assertTrue(plan.gate_to_category)

    def test_buying_with_too_few_slots_does_not_gate(self) -> None:
        plan, _ = decide_routes("buying", SlotSet())
        self.assertFalse(plan.gate_to_category)

    def test_browsing_never_gates_even_with_slots(self) -> None:
        slots = SlotSet(color="black", size="9", material="leather")
        plan, _ = decide_routes("browsing", slots)
        self.assertFalse(plan.gate_to_category)

    def test_track_weights_differ(self) -> None:
        buying_plan, _ = decide_routes("buying", SlotSet())
        browsing_plan, _ = decide_routes("browsing", SlotSet())
        self.assertGreater(buying_plan.weights["category"], browsing_plan.weights["category"])
        self.assertGreater(browsing_plan.weights["vector"], buying_plan.weights["vector"])

    def test_single_hard_marked_slot_does_not_gate_alone(self) -> None:
        # Reverted after an eval run showed this regressed the buying
        # scenario (see the comment in orchestrator.decide_routes) -- a
        # single hard-marked slot still gets enforced via hard_filter_slots,
        # but doesn't flip on the full category-gate machinery by itself.
        slots = SlotSet(budget=(0, 50.0), hard_marked=frozenset({"budget"}))
        plan, _ = decide_routes("buying", slots)
        self.assertFalse(plan.gate_to_category)
        self.assertEqual(plan.hard_filter_slots, ("budget",))

    def test_hard_filter_slots_populated_when_not_gated(self) -> None:
        slots = SlotSet(budget=(0, 100.0), hard_marked=frozenset({"budget"}))
        plan, _ = decide_routes("browsing", slots)
        self.assertFalse(plan.gate_to_category)
        self.assertEqual(plan.hard_filter_slots, ("budget",))

    def test_hard_filter_slots_empty_when_gated(self) -> None:
        slots = SlotSet(**{f: "x" for f in ("color", "size", "material")[:BUYING_GATE_MIN_SLOTS]},
                         hard_marked=frozenset({"color"}))
        plan, _ = decide_routes("buying", slots)
        self.assertTrue(plan.gate_to_category)
        self.assertEqual(plan.hard_filter_slots, ())


class RelaxGateTest(unittest.TestCase):
    def test_drops_lowest_priority_slot_first(self) -> None:
        slots = SlotSet(category="shoes", feature=["waterproof"], style="casual")
        relaxed, dropped, was_hard = relax_gate(slots)
        self.assertEqual(dropped, "feature")
        self.assertFalse(was_hard)
        self.assertEqual(relaxed.feature, [])
        self.assertEqual(relaxed.category, "shoes")  # untouched

    def test_never_drops_category(self) -> None:
        slots = SlotSet(category="shoes")
        self.assertIsNone(relax_gate(slots))

    def test_returns_none_when_only_category_filled(self) -> None:
        self.assertIsNone(relax_gate(SlotSet(category="shoes")))

    def test_prefers_soft_over_hard_marked_slot(self) -> None:
        slots = SlotSet(category="shoes", style="casual", budget=(0, 50.0), hard_marked=frozenset({"budget"}))
        relaxed, dropped, was_hard = relax_gate(slots)
        self.assertEqual(dropped, "style")
        self.assertFalse(was_hard)
        self.assertEqual(relaxed.budget, (0, 50.0))  # hard-marked slot left untouched

    def test_falls_back_to_hard_when_nothing_soft_left(self) -> None:
        slots = SlotSet(category="shoes", budget=(0, 50.0), hard_marked=frozenset({"budget"}))
        relaxed, dropped, was_hard = relax_gate(slots)
        self.assertEqual(dropped, "budget")
        self.assertTrue(was_hard)
        self.assertIsNone(relaxed.budget)


class DroppableSlotsByTierTest(unittest.TestCase):
    def test_splits_soft_and_hard_preserving_priority_order(self) -> None:
        slots = SlotSet(category="shoes", feature=["waterproof"], style="casual",
                         budget=(0.0, 50.0), hard_marked=frozenset({"budget"}))
        soft, hard = droppable_slots_by_tier(slots)
        self.assertEqual(soft, ["feature", "style"])
        self.assertEqual(hard, ["budget"])

    def test_empty_tiers_when_nothing_droppable(self) -> None:
        soft, hard = droppable_slots_by_tier(SlotSet(category="shoes"))
        self.assertEqual((soft, hard), ([], []))


class WidenBudgetTest(unittest.TestCase):
    def test_widens_both_sides(self) -> None:
        widened = widen_budget(SlotSet(budget=(20.0, 40.0)))
        self.assertLess(widened.budget[0], 20.0)
        self.assertGreater(widened.budget[1], 40.0)

    def test_no_budget_returns_none(self) -> None:
        self.assertIsNone(widen_budget(SlotSet()))

    def test_handles_one_sided_budget(self) -> None:
        widened = widen_budget(SlotSet(budget=(None, 50.0)))
        self.assertIsNone(widened.budget[0])
        self.assertGreater(widened.budget[1], 50.0)


class ShouldClarifyTest(unittest.TestCase):
    def test_never_clarifies_in_endgame(self) -> None:
        self.assertFalse(should_clarify(pool_size=200, filled_hard_slots=0, turns_remaining=2, best_split_score=0.9))

    def test_never_clarifies_on_tight_pool(self) -> None:
        self.assertFalse(should_clarify(pool_size=5, filled_hard_slots=0, turns_remaining=5, best_split_score=0.9))

    def test_never_clarifies_below_split_threshold(self) -> None:
        self.assertFalse(should_clarify(pool_size=200, filled_hard_slots=0, turns_remaining=5, best_split_score=0.01))

    def test_clarifies_when_over_general_and_good_split(self) -> None:
        self.assertTrue(should_clarify(pool_size=200, filled_hard_slots=0, turns_remaining=5, best_split_score=0.5))

    def test_never_clarifies_once_enough_slots_filled(self) -> None:
        self.assertFalse(should_clarify(pool_size=200, filled_hard_slots=6, turns_remaining=5, best_split_score=0.9))


class DecideRerankEngineTest(unittest.TestCase):
    """decide_rerank_engine returns (engine, reason) -- reason is what
    dialog_policy logs into SessionState.engine_trace for the
    LLM-invocation-rate breakdown (see scripts/run_llm_benchmark.py)."""

    def test_tight_pool_uses_heuristic(self) -> None:
        self.assertEqual(
            decide_rerank_engine(5, turn=1, max_turns=MAX_TURNS, llm_available=True, llm_disabled=False),
            ("heuristic", "tight_pool"),
        )

    def test_last_turn_always_heuristic(self) -> None:
        self.assertEqual(
            decide_rerank_engine(50, turn=MAX_TURNS, max_turns=MAX_TURNS, llm_available=True, llm_disabled=False),
            ("heuristic", "last_turn"),
        )

    def test_no_key_falls_back_to_heuristic(self) -> None:
        self.assertEqual(
            decide_rerank_engine(50, turn=1, max_turns=MAX_TURNS, llm_available=False, llm_disabled=False),
            ("heuristic", "no_provider"),
        )

    def test_circuit_breaker_forces_heuristic(self) -> None:
        self.assertEqual(
            decide_rerank_engine(50, turn=1, max_turns=MAX_TURNS, llm_available=True, llm_disabled=True),
            ("heuristic", "circuit_breaker"),
        )

    def test_uses_llm_when_available_and_pool_large(self) -> None:
        self.assertEqual(
            decide_rerank_engine(50, turn=1, max_turns=MAX_TURNS, llm_available=True, llm_disabled=False),
            ("llm", "eligible"),
        )

    def test_clarify_turn_forces_heuristic_even_with_llm_available(self) -> None:
        self.assertEqual(
            decide_rerank_engine(50, turn=1, max_turns=MAX_TURNS, llm_available=True, llm_disabled=False, do_clarify=True),
            ("heuristic", "clarify_skip"),
        )

    def test_do_clarify_defaults_to_false(self) -> None:
        # Every pre-existing call site (positional args only) must keep
        # getting "llm" here, unchanged -- do_clarify defaults off.
        self.assertEqual(
            decide_rerank_engine(50, turn=1, max_turns=MAX_TURNS, llm_available=True, llm_disabled=False),
            ("llm", "eligible"),
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from agent_shopper import dialog_policy as dialog_policy_mod
from agent_shopper.bm25_index import BM25Index
from agent_shopper.context import distill_profile
from agent_shopper.dialog_policy import choose_clarify_attribute, merge_slot_updates, process_turn
from agent_shopper.dialog_state import SessionState
from agent_shopper.models import Candidate, SlotSet
from agent_shopper.tfidf_index import TfidfIndex
from tests.agent_shopper.fixtures import make_catalog


def _state(slots: SlotSet = None) -> SessionState:
    return SessionState(
        session_id="s1", user_profile={}, distilled_profile=distill_profile({}),
        slots=slots or SlotSet(),
    )


class AccumulationVsOverrideTest(unittest.TestCase):
    def test_first_mention_is_plain_accumulation(self) -> None:
        state = _state()
        merge_slot_updates(state, {"color": "black"}, "I like black", turn=1)
        self.assertEqual(state.slots.color, "black")
        self.assertEqual(state.override_events, [])

    def test_conjunction_accumulates_into_list(self) -> None:
        state = _state(SlotSet(color="black"))
        merge_slot_updates(state, {"color": "white"}, "black or white is fine", turn=2)
        self.assertEqual(set(state.slots.color), {"black", "white"})
        self.assertEqual(state.override_events, [])

    def test_differing_point_slot_with_no_cue_overrides(self) -> None:
        state = _state(SlotSet(color="black"))
        merge_slot_updates(state, {"color": "red"}, "I want it in red", turn=2)
        self.assertEqual(state.slots.color, "red")
        self.assertEqual(len(state.override_events), 1)
        self.assertEqual(state.override_events[0].slot, "color")

    def test_explicit_contradiction_language_forces_override(self) -> None:
        state = _state(SlotSet(color="black", category="shoes"))
        merge_slot_updates(state, {"category": "earrings"}, "Actually, ignore my earlier preference. What I need is: earrings.", turn=3)
        self.assertEqual(state.slots.category, "earrings")
        self.assertTrue(state.has_overridden())

    def test_override_turn_now_replaces_stale_material_via_new_vocabulary(self) -> None:
        # Regression case for the intent_override slot-extraction gap: once
        # extract_slots() recognizes a catalog-style override value (see
        # slots.MATERIALS' "faux fur" addition), the stale pre-override
        # material actually gets replaced instead of silently surviving --
        # see scripts/diagnose_intent_override.py.
        state = _state(SlotSet(material="cotton"))
        merge_slot_updates(
            state, {"material": "faux fur"},
            "Actually, ignore my earlier preference. What I need is: Faux Fur.", turn=4,
        )
        self.assertEqual(state.slots.material, "faux fur")
        self.assertTrue(state.has_overridden())

    def test_category_refinement_is_accumulation_not_override(self) -> None:
        state = _state(SlotSet(category="shoes"))
        merge_slot_updates(state, {"category": "running shoes"}, "specifically running shoes", turn=2)
        self.assertEqual(state.slots.category, "running shoes")
        self.assertEqual(state.override_events, [])

    def test_category_conflict_clears_style_and_feature(self) -> None:
        state = _state(SlotSet(category="shoes", style="casual", feature=["waterproof"]))
        merge_slot_updates(state, {"category": "earrings"}, "I want earrings instead", turn=3)
        self.assertEqual(state.slots.category, "earrings")
        self.assertIsNone(state.slots.style)
        self.assertEqual(state.slots.feature, [])

    def test_same_department_override_preserves_style_and_feature(self) -> None:
        # "shoes" -> "boots" are both footwear -- style/feature should carry over.
        state = _state(SlotSet(category="shoes", style="casual", feature=["waterproof"]))
        state.clarified_attributes = {"style"}
        merge_slot_updates(state, {"category": "boots"}, "actually boots", turn=2)
        self.assertEqual(state.slots.style, "casual")
        self.assertEqual(state.slots.feature, ["waterproof"])
        self.assertIn("style", state.clarified_attributes)

    def test_different_department_override_clears_and_reopens_clarified_attributes(self) -> None:
        state = _state(SlotSet(category="shoes", style="casual", feature=["waterproof"]))
        state.clarified_attributes = {"style", "feature", "color"}
        merge_slot_updates(state, {"category": "earrings"}, "I want earrings instead", turn=3)
        self.assertIsNone(state.slots.style)
        self.assertEqual(state.slots.feature, [])
        self.assertNotIn("style", state.clarified_attributes)
        self.assertNotIn("feature", state.clarified_attributes)
        self.assertIn("color", state.clarified_attributes)  # untouched, unrelated slot

    def test_budget_overlap_tightens_range(self) -> None:
        state = _state(SlotSet(budget=(0.0, 100.0)))
        merge_slot_updates(state, {"budget": (20.0, 50.0)}, "maybe $20-50", turn=2)
        self.assertEqual(state.slots.budget, (20.0, 50.0))
        self.assertEqual(state.override_events, [])

    def test_budget_non_overlap_overrides(self) -> None:
        state = _state(SlotSet(budget=(0.0, 20.0)))
        merge_slot_updates(state, {"budget": (50.0, 100.0)}, "actually more like $50-100", turn=2)
        self.assertEqual(state.slots.budget, (50.0, 100.0))
        self.assertEqual(len(state.override_events), 1)

    def test_repeated_same_value_is_noop(self) -> None:
        state = _state(SlotSet(color="black"))
        merge_slot_updates(state, {"color": "black"}, "black is good", turn=2)
        self.assertEqual(state.slots.color, "black")
        self.assertEqual(state.override_events, [])

    def test_feature_always_accumulates(self) -> None:
        state = _state(SlotSet(feature=["waterproof"]))
        merge_slot_updates(state, {"feature": ["lightweight"]}, "also lightweight", turn=2)
        self.assertEqual(state.slots.feature, ["waterproof", "lightweight"])

    def test_hard_constraint_language_marks_slot_hard(self) -> None:
        state = _state()
        merge_slot_updates(state, {"budget": (0.0, 50.0)}, "My absolute max is $50, no more.", turn=1)
        self.assertIn("budget", state.slots.hard_marked)

    def test_soft_language_does_not_mark_hard(self) -> None:
        state = _state()
        merge_slot_updates(state, {"color": "black"}, "I like black", turn=1)
        self.assertNotIn("color", state.slots.hard_marked)

    def test_cross_domain_size_forces_override_not_conjunction(self) -> None:
        state = _state(SlotSet(size="M"))
        merge_slot_updates(state, {"size": "9"}, "size 9 or maybe something else", turn=2)
        self.assertEqual(state.slots.size, "9")

    def test_same_domain_size_conjunction_still_accumulates(self) -> None:
        state = _state(SlotSet(size="8"))
        merge_slot_updates(state, {"size": "9"}, "size 8 or 9 both work", turn=2)
        self.assertEqual(set(state.slots.size), {"8", "9"})

    def test_hard_cue_with_multiple_slots_extracted_marks_none(self) -> None:
        # Attribution is ambiguous when a hard-constraint cue and more than
        # one extracted slot appear in the same message -- skip marking
        # rather than guessing wrong on every extracted slot.
        state = _state()
        merge_slot_updates(
            state, {"budget": (None, 50.0), "material": "cotton"},
            "It must be under $50, maybe cotton.", turn=1,
        )
        self.assertEqual(state.slots.hard_marked, frozenset())

    def test_simulator_key_requirement_phrasing_does_not_mark_hard(self) -> None:
        # evaluator.local_evaluator.initial_message's buying-scenario template
        # ("A key requirement is: ...") always fires on turn 1 of nearly
        # every buying session -- an eval run confirmed that including this
        # phrasing in _HARD_CONSTRAINT_RE regressed buying-scenario HitRate@10
        # from 0.51 to 0.41 (a single, possibly-noisy, turn-1 extraction
        # getting hard-filtered too early). Deliberately excluded.
        state = _state()
        merge_slot_updates(state, {"material": "leather"}, "I'm looking for boots. A key requirement is: leather.", turn=1)
        self.assertNotIn("material", state.slots.hard_marked)

    def test_simulator_what_matters_is_phrasing_does_not_mark_hard(self) -> None:
        # Matches evaluator.local_evaluator.customer_reply's disclosure template,
        # which mixes hard_constraints and soft_preferences -- not a reliable signal.
        state = _state()
        merge_slot_updates(state, {"color": "blue"}, "For that, what matters is: color: blue.", turn=2)
        self.assertNotIn("color", state.slots.hard_marked)


class ChooseClarifyAttributeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()

    def test_single_candidate_pool_falls_back_to_feature_catchall(self) -> None:
        # A pool of one product has no entropy for any structured attribute
        # to exploit, so the fixed-score "feature" catch-all (matching the
        # simulator's own default classify_constraint bucket) should win.
        candidates = [Candidate(product=self.catalog.by_asin["B005"])]
        attr, score = choose_clarify_attribute(candidates, SlotSet(), set())
        self.assertEqual(attr, "feature")
        self.assertEqual(score, 0.3)

    def test_skips_already_filled_slots(self) -> None:
        slots = SlotSet(material="cotton", color="black", size="small", style="casual", budget=(0, 100))
        candidates = [Candidate(product=p) for p in self.catalog.products]
        attr, _ = choose_clarify_attribute(candidates, slots, set())
        self.assertNotIn(attr, ("material", "color", "size", "style", "budget"))

    def test_skips_already_clarified_attributes(self) -> None:
        candidates = [Candidate(product=p) for p in self.catalog.products]
        attr, _ = choose_clarify_attribute(
            candidates, SlotSet(), {"budget", "material", "color", "size", "style", "use_case", "feature"}
        )
        self.assertIsNone(attr)

    def test_never_returns_category_or_brand(self) -> None:
        candidates = [Candidate(product=p) for p in self.catalog.products]
        attr, _ = choose_clarify_attribute(candidates, SlotSet(), set())
        self.assertNotIn(attr, ("category", "brand"))

    def test_high_diversity_pool_beats_low_diversity(self) -> None:
        # Materials vary widely across the fixture catalog -> should score well.
        candidates = [Candidate(product=p) for p in self.catalog.products]
        attr, score = choose_clarify_attribute(candidates, SlotSet(), set())
        self.assertIsNotNone(attr)
        self.assertGreater(score, 0.0)


class ClarifyRelevanceTest(unittest.TestCase):
    def test_department_relevance_boosts_size_for_footwear(self) -> None:
        boosted = dialog_policy_mod._clarify_relevance("size", "boots")
        neutral = dialog_policy_mod._clarify_relevance("color", "boots")
        self.assertGreater(boosted, neutral)

    def test_neutral_for_unknown_category(self) -> None:
        self.assertEqual(dialog_policy_mod._clarify_relevance("size", "gadgets"), 1.0)

    def test_neutral_when_no_category_known(self) -> None:
        self.assertEqual(dialog_policy_mod._clarify_relevance("size", None), 1.0)


class ForcedRelaxClarifyTest(unittest.TestCase):
    """When every soft-preference slot has already been relaxed and the pool
    is still empty, the only thing left to drop is a hard-marked slot -- the
    turn should ask about it rather than silently dropping it."""

    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.bm25 = BM25Index(self.catalog)
        self.tfidf = TfidfIndex(self.catalog)
        # Cross-encoder reranking is on by default (see config.py) -- this
        # test exercises the heuristic clarify-relaxation path specifically
        # and must not trigger a real 256MB model load. See
        # tests/agent_shopper/test_cross_encoder_reranker.py for the
        # cross-encoder's own coverage.
        patcher = patch.object(dialog_policy_mod, "FROZEN_CROSS_ENCODER_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_asks_about_the_hard_marked_slot_instead_of_dropping_it(self) -> None:
        state = _state(SlotSet(
            category="necklace", material="silver",
            budget=(0.0, 1.0),  # no fixture product is this cheap -- guarantees a zero pool
            hard_marked=frozenset({"budget"}),
        ))
        response = process_turn(self.catalog, self.bm25, self.tfidf, state, "I need to buy something now", turn=1, top_k=5)
        self.assertEqual(response["ask_attribute"], "budget")
        self.assertIn("budget", response["message"].lower())


class EngineTraceAndUsageTest(unittest.TestCase):
    """No API key is configured in the test environment, so every turn here
    takes the heuristic path -- covers the "usage" dict and
    SessionState.engine_trace being populated (zeroed/no-LLM-outcome) on a
    turn that never calls out to an LLM, per dialog_policy.process_turn."""

    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.bm25 = BM25Index(self.catalog)
        self.tfidf = TfidfIndex(self.catalog)
        # See ForcedRelaxClarifyTest.setUp's comment -- same reasoning.
        patcher = patch.object(dialog_policy_mod, "FROZEN_CROSS_ENCODER_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_usage_is_zero_on_heuristic_turn(self) -> None:
        state = _state()
        response = process_turn(self.catalog, self.bm25, self.tfidf, state, "running shoes", turn=1, top_k=5)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_engine_trace_records_one_entry_per_turn(self) -> None:
        state = _state()
        process_turn(self.catalog, self.bm25, self.tfidf, state, "running shoes", turn=1, top_k=5)
        self.assertEqual(len(state.engine_trace), 1)
        entry = state.engine_trace[0]
        self.assertEqual(entry.engine, "heuristic")
        # The fixture catalog has only 8 products, so pool_size is well under
        # TIGHT_POOL_SIZE regardless of provider availability -- that branch
        # is checked first in decide_rerank_engine.
        self.assertEqual(entry.route_reason, "tight_pool")
        self.assertIsNone(entry.llm_outcome)


if __name__ == "__main__":
    unittest.main()

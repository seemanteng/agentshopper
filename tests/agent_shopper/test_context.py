import unittest

from agent_shopper.context import build_query_text, distill_profile, distill_session
from agent_shopper.dialog_state import ShownRecord
from agent_shopper.models import SlotSet


class DistillProfileTest(unittest.TestCase):
    def test_purchase_frequency_bucketing(self) -> None:
        low = distill_profile({"purchase_frequency": "1-2 prior purchases"})
        mid = distill_profile({"purchase_frequency": "3-4 prior purchases"})
        high = distill_profile({"purchase_frequency": "5+ prior purchases"})
        self.assertLess(low.decisiveness_prior, mid.decisiveness_prior)
        self.assertLess(mid.decisiveness_prior, high.decisiveness_prior)

    def test_missing_profile_fields_use_defaults(self) -> None:
        profile = distill_profile({})
        self.assertEqual(profile.preference_tags, [])
        self.assertEqual(profile.summary_short, "")

    def test_picky_rating_style_bumps_decisiveness_and_rating_floor(self) -> None:
        picky = distill_profile({"purchase_frequency": "3-4 prior purchases", "rating_style": "very critical"})
        neutral = distill_profile({"purchase_frequency": "3-4 prior purchases", "rating_style": "usually positive"})
        self.assertGreater(picky.decisiveness_prior, neutral.decisiveness_prior)
        self.assertGreater(picky.rating_floor_hint, neutral.rating_floor_hint)

    def test_summary_truncated_to_200_chars(self) -> None:
        profile = distill_profile({"summary": "x" * 500})
        self.assertEqual(len(profile.summary_short), 200)

    def test_preference_tags_kept_verbatim(self) -> None:
        profile = distill_profile({"preference_tags": ["fit", "comfort"]})
        self.assertEqual(profile.preference_tags, ["fit", "comfort"])


class QueryTextTest(unittest.TestCase):
    def test_query_text_includes_slots_and_message(self) -> None:
        slots = SlotSet(category="shoes", color="black")
        text = build_query_text(slots, "looking for something warm")
        self.assertIn("shoes", text)
        self.assertIn("black", text)
        self.assertIn("warm", text)

    def test_query_text_deduplicates(self) -> None:
        slots = SlotSet(category="shoes")
        text = build_query_text(slots, "shoes shoes shoes")
        self.assertEqual(text.split().count("shoes"), 1)

    def test_query_text_bounded_at_40_tokens(self) -> None:
        slots = SlotSet(feature=[f"feature{i}" for i in range(60)])
        text = build_query_text(slots, "")
        self.assertLessEqual(len(text.split()), 40)


class DistillSessionTest(unittest.TestCase):
    def test_turns_remaining_counts_down(self) -> None:
        early = distill_session(SlotSet(), "hi", turn=1, turns_since_progress=0, track="buying")
        late = distill_session(SlotSet(), "hi", turn=9, turns_since_progress=0, track="buying")
        self.assertGreater(early.turns_remaining, late.turns_remaining)

    def test_stuck_flag_set_after_threshold(self) -> None:
        not_stuck = distill_session(SlotSet(), "hi", turn=3, turns_since_progress=1, track="buying")
        stuck = distill_session(SlotSet(), "hi", turn=3, turns_since_progress=2, track="buying")
        self.assertFalse(not_stuck.stuck)
        self.assertTrue(stuck.stuck)

    def test_slot_summary_excludes_unfilled(self) -> None:
        session = distill_session(SlotSet(color="black"), "hi", turn=1, turns_since_progress=0, track="buying")
        self.assertEqual(session.slot_summary, {"color": "black"})

    def test_shown_asin_counts_empty_with_no_history(self) -> None:
        session = distill_session(SlotSet(), "hi", turn=2, turns_since_progress=0, track="buying")
        self.assertEqual(session.shown_asin_counts, {})

    def test_shown_asin_counts_populated(self) -> None:
        history = [
            ShownRecord(turn=1, parent_asin="B001"),
            ShownRecord(turn=2, parent_asin="B001"),
            ShownRecord(turn=2, parent_asin="B002"),
        ]
        session = distill_session(SlotSet(), "hi", turn=3, turns_since_progress=0, track="buying", shown_history=history)
        self.assertEqual(session.shown_asin_counts, {"B001": 2, "B002": 1})

    def test_shown_asin_counts_excludes_current_and_future_turns(self) -> None:
        history = [ShownRecord(turn=2, parent_asin="B001")]
        session = distill_session(SlotSet(), "hi", turn=2, turns_since_progress=0, track="buying", shown_history=history)
        self.assertEqual(session.shown_asin_counts, {})


if __name__ == "__main__":
    unittest.main()

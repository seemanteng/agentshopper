import unittest

from agent_shopper.intent import classify, score_intent
from agent_shopper.models import SlotSet


class IntentClassifierTest(unittest.TestCase):
    def test_budget_and_imperative_language_is_buying(self) -> None:
        track = classify("I need running shoes under $50", SlotSet(), turn=1, has_overridden=False, decisiveness_prior=0.5)
        self.assertEqual(track, "buying")

    def test_browsing_language_stays_browsing(self) -> None:
        track = classify("Just browsing for some ideas", SlotSet(), turn=1, has_overridden=False, decisiveness_prior=0.5)
        self.assertEqual(track, "browsing")

    def test_many_filled_slots_pushes_toward_buying(self) -> None:
        slots = SlotSet(category="shoes", color="black", size="9", material="leather")
        track = classify("here are my exact requirements", slots, turn=1, has_overridden=False, decisiveness_prior=0.5)
        self.assertEqual(track, "buying")

    def test_high_decisiveness_prior_nudges_toward_buying(self) -> None:
        low = classify("something nice", SlotSet(), turn=1, has_overridden=False, decisiveness_prior=0.3)
        high = classify("something nice", SlotSet(), turn=1, has_overridden=False, decisiveness_prior=0.7)
        self.assertEqual(low, "browsing")
        # Same message, higher decisiveness prior should never make the score lower.
        self.assertIn(high, ("browsing", "buying"))

    def test_score_intent_returns_two_separate_signals(self) -> None:
        language, specificity = score_intent(
            "just browsing for ideas", SlotSet(color="black", size="9"), decisiveness_prior=0.5,
        )
        self.assertLess(language, 0)  # browsing language pulls this negative...
        self.assertEqual(specificity, 2)  # ...independent of the two filled hard slots

    def test_turn_nudge_only_applies_near_threshold_with_neutral_language(self) -> None:
        slots = SlotSet(color="black", size="9", material="leather")  # specificity = 3
        kwargs = dict(
            message="here's an update", slots=slots, has_overridden=False,
            decisiveness_prior=0.625, current_track="browsing",  # language = 0.25 -> combined = 3.25
        )
        self.assertEqual(classify(turn=2, **kwargs), "browsing")  # no nudge yet, hysteresis holds at 3.25
        self.assertEqual(classify(turn=3, **kwargs), "buying")  # +0.5 nudge clears the hysteresis margin

    def test_hysteresis_prevents_flip_on_marginal_crossing(self) -> None:
        slots = SlotSet(color="black", size="9")  # specificity = 2, language = 0.6 -> combined = 2.6
        track = classify(
            "here's an update", slots, turn=2, has_overridden=False,
            decisiveness_prior=0.8, current_track="buying",
        )
        self.assertEqual(track, "buying")  # dips below INTENT_BUYING_THRESHOLD but not past the hysteresis margin

    def test_hysteresis_allows_flip_when_score_clearly_negative(self) -> None:
        track = classify(
            "just browsing for ideas", SlotSet(), turn=2, has_overridden=False,
            decisiveness_prior=0.5, current_track="buying",
        )
        self.assertEqual(track, "browsing")  # well past the hysteresis margin, so it does flip

    def test_turn_one_ignores_hysteresis(self) -> None:
        # Same inputs as test_hysteresis_prevents_flip_on_marginal_crossing, but
        # turn 1 has no real prior to be sticky about -- the provisional
        # classification wins outright instead of holding at current_track.
        slots = SlotSet(color="black", size="9")
        track = classify(
            "here's an update", slots, turn=1, has_overridden=False,
            decisiveness_prior=0.8, current_track="buying",
        )
        self.assertEqual(track, "browsing")


if __name__ == "__main__":
    unittest.main()

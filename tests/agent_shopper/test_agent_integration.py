import unittest
from unittest.mock import patch

from agent_shopper.agent import Agent
from agent_shopper.llm_client import LLMUnavailable
from tests.agent_shopper.fixtures import write_catalog_file

VALID_ASK_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other", None,
}


def _assert_valid_response(test: unittest.TestCase, response: dict) -> None:
    test.assertIsInstance(response, dict)
    test.assertIsInstance(response["message"], str)
    test.assertIn(response["ask_attribute"], VALID_ASK_ATTRIBUTES)
    test.assertIsInstance(response["recommendations"], list)
    test.assertLessEqual(len(response["recommendations"]), 100)
    for rec in response["recommendations"]:
        test.assertIn("parent_asin", rec)


class AgentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_path = write_catalog_file()
        self.addCleanup(self.catalog_path.unlink, missing_ok=True)
        self.agent = Agent(self.catalog_path)

    def test_reset_then_respond_returns_valid_schema(self) -> None:
        self.agent.reset("s1", {"purchase_frequency": "3-4 prior purchases", "rating_style": "usually positive",
                                 "preference_tags": ["comfort"], "average_prior_rating": 4.5, "summary": "likes shoes"})
        response = self.agent.respond("s1", "I need running shoes under $60", turn=1, top_k=10)
        _assert_valid_response(self, response)

    def test_respond_before_reset_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("unknown", "hi", turn=1, top_k=10)

    def test_full_scripted_buying_session_never_raises(self) -> None:
        self.agent.reset("buying-session", {"purchase_frequency": "1-2 prior purchases", "rating_style": "neutral", "preference_tags": []})
        messages = [
            "I'm looking for shoes. A key requirement is: leather.",
            "I'd like them in black, under $100.",
            "Size 9 please.",
            "Athletic style for running.",
        ]
        for turn, message in enumerate(messages, start=1):
            response = self.agent.respond("buying-session", message, turn, 10)
            _assert_valid_response(self, response)

    def test_intent_override_session_clears_and_rewrites(self) -> None:
        self.agent.reset("override-session", {"purchase_frequency": "3-4 prior purchases", "rating_style": "usually positive", "preference_tags": []})
        self.agent.respond("override-session", "I'm looking for shoes. I like casual style.", 1, 10)
        response = self.agent.respond(
            "override-session",
            "Actually, ignore my earlier preference. What I need is: leather boots.",
            2, 10,
        )
        _assert_valid_response(self, response)
        state = self.agent.sessions.get("override-session")
        self.assertTrue(state.has_overridden())

    def test_llm_client_mocked_to_always_raise_never_crashes_agent(self) -> None:
        self.agent.reset("llm-fail-session", {"purchase_frequency": "5+ prior purchases", "rating_style": "critical", "preference_tags": ["comfort"]})
        with patch("agent_shopper.reranker.call_structured", side_effect=LLMUnavailable("boom")), \
             patch("agent_shopper.reranker.active_provider", return_value="openai"), \
             patch("agent_shopper.dialog_policy.active_provider", return_value="openai"):
            for turn in range(1, 4):
                response = self.agent.respond("llm-fail-session", "I need a red dress under $80", turn, 10)
                _assert_valid_response(self, response)

    def test_ten_turn_session_completes_within_turn_budget(self) -> None:
        self.agent.reset("boundary-session", {"purchase_frequency": "1-2 prior purchases", "rating_style": "neutral", "preference_tags": []})
        message = "I'm just browsing, not sure what I want yet."
        for turn in range(1, 11):
            response = self.agent.respond("boundary-session", message, turn, 10)
            _assert_valid_response(self, response)
            message = "Those options are not quite right yet. Ask me about one specific attribute."


if __name__ == "__main__":
    unittest.main()

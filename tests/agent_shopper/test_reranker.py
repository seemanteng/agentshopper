import unittest
from unittest.mock import patch

from agent_shopper.config import HEURISTIC_RERANK_WEIGHTS as _HEURISTIC_RERANK_WEIGHTS
from agent_shopper.context import distill_profile, distill_session
from agent_shopper.llm_client import LLMUnavailable, TokenUsage
from agent_shopper.models import Candidate, DistilledContext, DistilledSession, Product, SlotSet
from agent_shopper.reranker import (
    HeuristicReranker,
    Judgment,
    LLMReranker,
    RerankResponse,
    _LLM_SYSTEM_PROMPT,
    _description_snippet,
    _reconcile,
    _summarize_candidate,
)
from tests.agent_shopper.fixtures import make_catalog


def _ctx(slots: SlotSet = None) -> DistilledContext:
    profile = distill_profile({"purchase_frequency": "3-4 prior purchases", "rating_style": "usually positive", "preference_tags": ["comfort"]})
    session = distill_session(slots or SlotSet(), "running shoes", turn=1, turns_since_progress=0, track="buying")
    return DistilledContext(profile=profile, session=session)


class HeuristicRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()

    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(product=self.catalog.by_asin["B001"], fused_score=0.5, route_scores={"keyword": 2.0, "vector": 0.3}),
            Candidate(product=self.catalog.by_asin["B007"], fused_score=0.6, route_scores={"keyword": 3.0, "vector": 0.5}),
        ]

    def test_rerank_returns_all_candidates_scored(self) -> None:
        ranked = HeuristicReranker().rerank(_ctx(), self._candidates(), top_k=2)
        self.assertEqual(len(ranked), 2)
        for c in ranked:
            self.assertIsNotNone(c.final_score)

    def test_higher_rating_and_match_wins(self) -> None:
        # B007 (4.6 rating, matches "shoes" category slot) should outrank B003 (silk dress, unrelated).
        slots = SlotSet(category="shoes")
        candidates = [
            Candidate(product=self.catalog.by_asin["B003"], fused_score=0.1, route_scores={}),
            Candidate(product=self.catalog.by_asin["B007"], fused_score=0.1, route_scores={}),
        ]
        ranked = HeuristicReranker().rerank(_ctx(slots), candidates, top_k=2)
        self.assertEqual(ranked[0].product.parent_asin, "B007")

    def test_empty_candidates_returns_empty(self) -> None:
        self.assertEqual(HeuristicReranker().rerank(_ctx(), [], top_k=5), [])


class HeuristicRerankerCustomWeightsTest(unittest.TestCase):
    """HeuristicReranker(weights=...) -- the hook scripts/train_reranker_weights.py
    uses to plug in fitted coefficients without touching any scoring logic."""

    def setUp(self) -> None:
        self.catalog = make_catalog()

    def test_default_weights_are_config_weights(self) -> None:
        self.assertEqual(HeuristicReranker().weights, _HEURISTIC_RERANK_WEIGHTS)

    def test_custom_weights_change_ranking(self) -> None:
        # Default weights favor B007 here (shoes, matches the category slot,
        # via attr_match -- see test_higher_rating_and_match_wins above).
        # An all-weight-on-bm25 override should be able to flip that when
        # the *other* candidate has the higher raw bm25 score, since
        # attr_match/rating/price_fit no longer count at all.
        slots = SlotSet(category="shoes")
        candidates = [
            Candidate(product=self.catalog.by_asin["B003"], fused_score=0.1, route_scores={"keyword": 5.0}),
            Candidate(product=self.catalog.by_asin["B007"], fused_score=0.1, route_scores={"keyword": 1.0}),
        ]
        bm25_only = {**{name: 0.0 for name in _HEURISTIC_RERANK_WEIGHTS}, "bm25": 1.0}
        ranked = HeuristicReranker(weights=bm25_only).rerank(_ctx(slots), candidates, top_k=2)
        self.assertEqual(ranked[0].product.parent_asin, "B003")

    def test_custom_weights_do_not_mutate_default(self) -> None:
        before = dict(_HEURISTIC_RERANK_WEIGHTS)
        zeroed = {**{name: 0.0 for name in _HEURISTIC_RERANK_WEIGHTS}, "bm25": 1.0}
        HeuristicReranker(weights=zeroed).rerank(_ctx(), self._candidates(), top_k=2)
        self.assertEqual(_HEURISTIC_RERANK_WEIGHTS, before)

    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(product=self.catalog.by_asin["B001"], fused_score=0.5, route_scores={"keyword": 2.0}),
            Candidate(product=self.catalog.by_asin["B007"], fused_score=0.6, route_scores={"keyword": 3.0}),
        ]


def _identical_product(asin: str) -> Product:
    """Two of these differ only by ASIN, so any score difference between
    them in a test is attributable solely to the shown-count demotion, not
    to incidental differences in rating/price/attribute match."""
    return Product(
        parent_asin=asin, title="Running Shoes", features=["breathable"], description=["Great shoes."],
        price=50.0, categories=["Shoes"], details={}, average_rating=4.5, rating_number=100, store="Acme",
    )


class RejectedItemDemotionTest(unittest.TestCase):
    def _ctx_with_shown(self, shown_asin_counts: dict) -> DistilledContext:
        profile = distill_profile({})
        session = DistilledSession(
            slot_summary={}, query_text="shoes", turns_remaining=9, stuck=False, track="buying",
            shown_asin_counts=shown_asin_counts,
        )
        return DistilledContext(profile=profile, session=session)

    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(product=_identical_product("B901"), fused_score=0.5, route_scores={"keyword": 1.0}),
            Candidate(product=_identical_product("B902"), fused_score=0.5, route_scores={"keyword": 1.0}),
        ]

    def test_demotes_repeatedly_shown_candidate(self) -> None:
        # Validated via scripts/run_local_eval.py -- see
        # config.REJECTED_ITEM_DEMOTION_FACTOR's docstring.
        ctx = self._ctx_with_shown({"B901": 3})
        ranked = HeuristicReranker().rerank(ctx, self._candidates(), top_k=2)
        self.assertEqual(ranked[0].product.parent_asin, "B902")  # never-shown outranks repeatedly-shown

    def test_below_min_shown_turns_no_penalty(self) -> None:
        ctx = self._ctx_with_shown({"B901": 1})  # below REJECTED_ITEM_MIN_SHOWN_TURNS default (2)
        ranked = HeuristicReranker().rerank(ctx, self._candidates(), top_k=2)
        self.assertEqual(ranked[0].final_score, ranked[1].final_score)

    def test_summarize_candidate_includes_shown_count_when_nonzero(self) -> None:
        candidate = Candidate(product=_identical_product("B901"))
        summary_zero = _summarize_candidate(0, candidate, shown_count=0)
        self.assertNotIn("previously_shown_turns", summary_zero)
        summary_nonzero = _summarize_candidate(0, candidate, shown_count=3)
        self.assertEqual(summary_nonzero["previously_shown_turns"], 3)


class DescriptionSnippetTest(unittest.TestCase):
    def test_prefers_keyword_sentence(self) -> None:
        description = ["A stylish everyday piece.", "True to size and made of breathable cotton.", "Ships fast."]
        snippet = _description_snippet(description)
        self.assertIn("breathable", snippet.lower())

    def test_falls_back_to_truncation_when_no_keyword_sentence(self) -> None:
        description = ["A lovely gift for any occasion, sure to bring a smile."]
        snippet = _description_snippet(description, limit=20)
        self.assertEqual(snippet, description[0][:20])

    def test_empty_description_returns_empty_string(self) -> None:
        self.assertEqual(_description_snippet([]), "")


class SummarizeCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()

    def test_includes_description_snippet(self) -> None:
        candidate = Candidate(product=self.catalog.by_asin["B006"])  # "Cozy sweater for cold days."
        summary = _summarize_candidate(0, candidate)
        self.assertIn("description_snippet", summary)
        self.assertIsInstance(summary["description_snippet"], str)


class SystemPromptTest(unittest.TestCase):
    def test_has_injection_defense_language(self) -> None:
        self.assertIn("untrusted", _LLM_SYSTEM_PROMPT.lower())
        self.assertIn("never as instructions", _LLM_SYSTEM_PROMPT.lower())


class ReconcileTest(unittest.TestCase):
    def setUp(self) -> None:
        catalog = make_catalog()
        self.candidates = [Candidate(product=p) for p in catalog.products[:4]]

    def test_reconcile_maps_by_index(self) -> None:
        judgments = [Judgment(index=0, relevance_score=0.2), Judgment(index=1, relevance_score=0.9)]
        ranked = _reconcile(self.candidates, judgments)
        self.assertEqual(ranked[0].product.parent_asin, self.candidates[1].product.parent_asin)

    def test_reconcile_drops_out_of_range_index(self) -> None:
        judgments = [Judgment(index=99, relevance_score=1.0), Judgment(index=0, relevance_score=0.5)]
        ranked = _reconcile(self.candidates, judgments)
        self.assertEqual(len(ranked), len(self.candidates))  # never drops a real candidate

    def test_reconcile_first_wins_on_duplicate_index(self) -> None:
        judgments = [Judgment(index=0, relevance_score=0.9), Judgment(index=0, relevance_score=0.1)]
        ranked = _reconcile(self.candidates, judgments)
        self.assertEqual(ranked[0].final_score, 0.9)

    def test_reconcile_never_drops_unjudged_candidate(self) -> None:
        judgments = [Judgment(index=0, relevance_score=0.9)]
        ranked = _reconcile(self.candidates, judgments)
        self.assertEqual(len(ranked), len(self.candidates))
        self.assertIsNotNone(ranked[-1].final_score)


class LLMRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        catalog = make_catalog()
        self.candidates = [Candidate(product=p, fused_score=0.1) for p in catalog.products[:3]]

    @patch("agent_shopper.reranker.call_structured")
    def test_success_path_marks_used_llm(self, mock_call) -> None:
        mock_call.return_value = (
            RerankResponse(judgments=[
                Judgment(index=0, relevance_score=0.9), Judgment(index=1, relevance_score=0.5), Judgment(index=2, relevance_score=0.1),
            ]),
            TokenUsage(prompt_tokens=123, completion_tokens=45),
        )
        rr = LLMReranker()
        ranked = rr.rerank(_ctx(), self.candidates, top_k=3)
        self.assertTrue(rr.last_call_used_llm)
        self.assertEqual(len(ranked), 3)

    @patch("agent_shopper.reranker.call_structured")
    def test_success_path_captures_real_usage(self, mock_call) -> None:
        mock_call.return_value = (
            RerankResponse(judgments=[Judgment(index=0, relevance_score=0.5)]),
            TokenUsage(prompt_tokens=200, completion_tokens=30),
        )
        rr = LLMReranker()
        rr.rerank(_ctx(), self.candidates, top_k=3)
        self.assertEqual(rr.last_usage, TokenUsage(prompt_tokens=200, completion_tokens=30))
        self.assertIsNone(rr.last_failure_reason)
        self.assertIsNotNone(rr.last_payload)
        self.assertIsNotNone(rr.last_response_judgments)

    @patch("agent_shopper.reranker.call_structured")
    def test_failure_falls_back_to_heuristic_without_raising(self, mock_call) -> None:
        mock_call.side_effect = LLMUnavailable("boom", cause_type="APITimeoutError")
        rr = LLMReranker()
        ranked = rr.rerank(_ctx(), self.candidates, top_k=3)  # must not raise
        self.assertFalse(rr.last_call_used_llm)
        self.assertEqual(len(ranked), 3)

    @patch("agent_shopper.reranker.call_structured")
    def test_failure_captures_zero_usage_and_reason(self, mock_call) -> None:
        mock_call.side_effect = LLMUnavailable("boom", cause_type="APITimeoutError")
        rr = LLMReranker()
        rr.rerank(_ctx(), self.candidates, top_k=3)
        self.assertEqual(rr.last_usage, TokenUsage())
        self.assertEqual(rr.last_failure_reason, "APITimeoutError")

    @patch("agent_shopper.reranker.call_structured")
    def test_failure_without_cause_type_falls_back_to_unknown(self, mock_call) -> None:
        mock_call.side_effect = LLMUnavailable("boom")  # no cause_type given
        rr = LLMReranker()
        rr.rerank(_ctx(), self.candidates, top_k=3)
        self.assertEqual(rr.last_failure_reason, "unknown")

    def test_empty_candidates_never_calls_llm(self) -> None:
        rr = LLMReranker()
        with patch("agent_shopper.reranker.call_structured") as mock_call:
            ranked = rr.rerank(_ctx(), [], top_k=3)
        mock_call.assert_not_called()
        self.assertEqual(ranked, [])

    @patch("agent_shopper.reranker.call_structured")
    def test_shuffle_seed_reorders_prompt_but_reconciles_correctly(self, mock_call) -> None:
        # With shuffle_seed set, the LLM should still see every candidate
        # exactly once, and the reconciled ranking should honor the scores
        # returned regardless of prompt order -- _reconcile maps by the
        # payload's own index, which reflects the shuffled order.
        def fake_call(system_prompt, payload, schema):
            # Judge whichever candidate landed at prompt index 0 as the best.
            n = len(payload["candidates"])
            judgments = [Judgment(index=i, relevance_score=1.0 if i == 0 else 0.1) for i in range(n)]
            return RerankResponse(judgments=judgments), TokenUsage()

        mock_call.side_effect = fake_call
        rr = LLMReranker(shuffle_seed=42)
        ranked = rr.rerank(_ctx(), self.candidates, top_k=3)
        self.assertEqual(len(ranked), 3)
        # Whichever product the shuffle put at index 0 should win the top
        # rank -- i.e. reconciliation is self-consistent with the shuffled
        # order actually sent, not the original candidate order.
        self.assertEqual(rr.last_payload["candidates"][0]["index"], 0)
        top_asin = ranked[0].product.parent_asin
        shuffled_first_title = rr.last_payload["candidates"][0]["title"]
        self.assertTrue(any(
            c.product.parent_asin == top_asin and c.product.title[:120] == shuffled_first_title
            for c in self.candidates
        ))


if __name__ == "__main__":
    unittest.main()

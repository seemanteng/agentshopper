import unittest

from agent_shopper.bm25_index import BM25Index
from agent_shopper.models import RoutePlan, SlotSet
from agent_shopper.retrieval import retrieve
from agent_shopper.tfidf_index import TfidfIndex
from tests.agent_shopper.fixtures import make_catalog


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.bm25 = BM25Index(self.catalog)
        self.tfidf = TfidfIndex(self.catalog)

    def test_gated_buying_restricts_to_category_filter(self) -> None:
        slots = SlotSet(category="shoes", material="nylon")
        plan = RoutePlan(weights={"category": 0.5, "keyword": 0.35, "vector": 0.15}, gate_to_category=True)
        result = retrieve(self.catalog, self.bm25, self.tfidf, "running shoes nylon", slots, plan)
        asins = {c.product.parent_asin for c in result.candidates}
        self.assertEqual(asins, {"B007"})
        self.assertEqual(result.pool_size, 1)
        self.assertTrue(result.gated)

    def test_gated_empty_pool_returns_no_candidates(self) -> None:
        slots = SlotSet(category="shoes", material="silk")  # no product matches both
        plan = RoutePlan(weights={"category": 0.5, "keyword": 0.35, "vector": 0.15}, gate_to_category=True)
        result = retrieve(self.catalog, self.bm25, self.tfidf, "silk shoes", slots, plan)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.pool_size, 0)

    def test_ungated_browsing_covers_full_catalog(self) -> None:
        plan = RoutePlan(weights={"category": 0.2, "keyword": 0.35, "vector": 0.45}, gate_to_category=False)
        result = retrieve(self.catalog, self.bm25, self.tfidf, "shoes", SlotSet(), plan)
        self.assertFalse(result.gated)
        self.assertGreater(len(result.candidates), 1)

    def test_fused_scores_are_descending(self) -> None:
        plan = RoutePlan(weights={"category": 0.2, "keyword": 0.35, "vector": 0.45}, gate_to_category=False)
        result = retrieve(self.catalog, self.bm25, self.tfidf, "running shoes", SlotSet(), plan)
        scores = [c.fused_score for c in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hard_marked_slot_filters_even_when_ungated(self) -> None:
        # Browsing-track (never gated) but the budget was stated as a hard
        # requirement -- it should still hard-filter the fused pool, not
        # just soft-rank it.
        slots = SlotSet(budget=(0.0, 20.0), hard_marked=frozenset({"budget"}))
        plan = RoutePlan(
            weights={"category": 0.2, "keyword": 0.35, "vector": 0.45},
            gate_to_category=False,
            hard_filter_slots=("budget",),
        )
        result = retrieve(self.catalog, self.bm25, self.tfidf, "cotton shirt", slots, plan)
        self.assertFalse(result.gated)
        asins = {c.product.parent_asin for c in result.candidates}
        self.assertIn("B004", asins)  # $15 cotton t-shirt, within the hard budget
        for c in result.candidates:
            self.assertLessEqual(c.product.price, 20.0)


if __name__ == "__main__":
    unittest.main()

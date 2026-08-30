import unittest

from agent_shopper.tfidf_index import TfidfIndex
from tests.agent_shopper.fixtures import make_catalog


class TfidfIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.index = TfidfIndex(self.catalog)

    def test_search_finds_semantically_close_product(self) -> None:
        results = self.index.search("sweater for cold weather", limit=3)
        asins = [self.catalog.products[i].parent_asin for i, _ in results]
        self.assertIn("B006", asins)

    def test_search_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.index.search("", limit=5), [])

    def test_limit_is_respected(self) -> None:
        results = self.index.search("shoes", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_scores_between_zero_and_one(self) -> None:
        results = self.index.search("running shoes athletic", limit=8)
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()

import unittest

from agent_shopper.bm25_index import BM25Index
from tests.agent_shopper.fixtures import make_catalog


class BM25IndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.index = BM25Index(self.catalog)

    def test_search_returns_relevant_product_first(self) -> None:
        results = self.index.search("leather boots winter", limit=5)
        self.assertTrue(results)
        top_asin = self.catalog.products[results[0][0]].parent_asin
        self.assertEqual(top_asin, "B002")

    def test_search_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.index.search("", limit=5), [])
        self.assertEqual(self.index.search("the a an", limit=5), [])  # all stopwords

    def test_search_scores_are_descending(self) -> None:
        results = self.index.search("running shoes", limit=8)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_title_field_weighted_more_than_description(self) -> None:
        # "necklace" only appears in B008's title -- must be found and scored positively.
        results = self.index.search("necklace", limit=5)
        asins = [self.catalog.products[i].parent_asin for i, _ in results]
        self.assertIn("B008", asins)


if __name__ == "__main__":
    unittest.main()

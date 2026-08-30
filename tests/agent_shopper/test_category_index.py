import unittest

from agent_shopper.category_index import (
    attr_match_fraction,
    filter_products,
    matches_budget,
    matches_category,
)
from agent_shopper.models import SlotSet
from tests.agent_shopper.fixtures import make_catalog


class CategoryIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()

    def test_matches_category_substring(self) -> None:
        product = self.catalog.by_asin["B001"]
        self.assertTrue(matches_category(product, "shoes"))
        self.assertFalse(matches_category(product, "jewelry"))

    def test_matches_budget_excludes_out_of_range(self) -> None:
        product = self.catalog.by_asin["B002"]  # price 89.0
        self.assertTrue(matches_budget(product, (50.0, 100.0)))
        self.assertFalse(matches_budget(product, (0.0, 50.0)))

    def test_matches_budget_excludes_unpriced_product(self) -> None:
        product = self.catalog.by_asin["B001"]
        object.__setattr__(product, "price", None)
        self.assertFalse(matches_budget(product, (0.0, 50.0)))

    def test_filter_products_ands_all_filled_slots(self) -> None:
        slots = SlotSet(category="shoes", material="nylon")
        ids = filter_products(self.catalog, slots)
        asins = {self.catalog.products[i].parent_asin for i in ids}
        self.assertEqual(asins, {"B007"})

    def test_filter_products_no_slots_returns_everything(self) -> None:
        ids = filter_products(self.catalog, SlotSet())
        self.assertEqual(len(ids), len(self.catalog.products))

    def test_attr_match_fraction_partial_match(self) -> None:
        slots = SlotSet(category="shoes", color="black")  # B002 matches category, not color explicitly stated as "black"... but title says Black
        product = self.catalog.by_asin["B002"]
        fraction = attr_match_fraction(product, slots)
        self.assertGreaterEqual(fraction, 0.5)

    def test_attr_match_fraction_no_filled_slots_is_zero(self) -> None:
        product = self.catalog.by_asin["B001"]
        self.assertEqual(attr_match_fraction(product, SlotSet()), 0.0)


if __name__ == "__main__":
    unittest.main()

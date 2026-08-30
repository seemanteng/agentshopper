import unittest

from agent_shopper.text_utils import flatten_text, terms, unique_terms


class TextUtilsTest(unittest.TestCase):
    def test_flatten_text_handles_list_dict_none(self) -> None:
        self.assertEqual(flatten_text(None), "")
        self.assertEqual(flatten_text(["a", "b"]), "a b")
        self.assertEqual(flatten_text({"Department": "Mens"}), "Department Mens")
        self.assertEqual(flatten_text("plain"), "plain")

    def test_terms_drops_stopwords_and_short_tokens(self) -> None:
        result = terms("I am looking for a blue running shoe")
        self.assertIn("blue", result)
        self.assertIn("running", result)
        self.assertNotIn("a", result)
        self.assertNotIn("for", result)

    def test_unique_terms_dedupes_preserving_order_and_limit(self) -> None:
        result = unique_terms("blue shoe blue shoe red shoe", limit=2)
        self.assertEqual(result, ["blue", "shoe"])


if __name__ == "__main__":
    unittest.main()

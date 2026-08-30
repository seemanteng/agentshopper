import unittest

from agent_shopper.slots import extract_slots, parse_budget, same_department, size_domain


class BudgetParsingTest(unittest.TestCase):
    def test_range(self) -> None:
        self.assertEqual(parse_budget("I'd like something $20-40"), (20.0, 40.0))

    def test_under(self) -> None:
        self.assertEqual(parse_budget("under $50 please"), (None, 50.0))

    def test_over(self) -> None:
        self.assertEqual(parse_budget("at least $30"), (30.0, None))

    def test_around_widens_both_sides(self) -> None:
        lo, hi = parse_budget("budget around $30")
        self.assertLess(lo, 30.0)
        self.assertGreater(hi, 30.0)

    def test_bare_dollar_amount(self) -> None:
        lo, hi = parse_budget("I have $40 to spend")
        self.assertLess(lo, 40.0)
        self.assertGreater(hi, 40.0)

    def test_no_budget_mentioned(self) -> None:
        self.assertIsNone(parse_budget("I like blue shoes"))


class ExtractSlotsTest(unittest.TestCase):
    def test_extracts_material_and_color(self) -> None:
        extracted = extract_slots("I want something in leather, preferably black.")
        self.assertEqual(extracted["material"], "leather")
        self.assertEqual(extracted["color"], "black")

    def test_extracts_use_case_and_category(self) -> None:
        extracted = extract_slots("Looking for running shoes for hiking.")
        self.assertEqual(extracted["category"], "shoes")
        self.assertEqual(extracted["use_case"], "hiking")

    def test_extracts_budget(self) -> None:
        extracted = extract_slots("Something under $50 would be great.")
        self.assertEqual(extracted["budget"], (None, 50.0))

    def test_no_false_positive_on_unrelated_message(self) -> None:
        extracted = extract_slots("Those options are not quite right yet.")
        self.assertEqual(extracted, {})

    def test_multiple_colors_returned_as_list(self) -> None:
        extracted = extract_slots("I like it in black or white.")
        self.assertIn("color", extracted)
        self.assertIsInstance(extracted["color"], list)
        self.assertEqual(set(extracted["color"]), {"black", "white"})


class MaterialsExpansionTest(unittest.TestCase):
    def test_extracts_faux_fur(self) -> None:
        self.assertEqual(extract_slots("I'd like it in faux fur.")["material"], "faux fur")

    def test_extracts_acrylic(self) -> None:
        self.assertEqual(extract_slots("100% Acrylic please.")["material"], "acrylic")


class IntentOverrideRegressionTest(unittest.TestCase):
    """Regression cases for the 7 real intent_override sessions in
    data/public_set.jsonl whose override message extracted zero slots
    before this change (see scripts/diagnose_intent_override.py). Each
    message is the evaluator's literal override template --
    evaluator/local_evaluator.py's behavior_for(): "Actually, ignore my
    earlier preference. What I need is: {new_value}."

    Only "faux fur"/"acrylic" (public_0072/public_0125) are asserted as
    fixed -- confirmed via scripts/run_local_eval.py session-level
    before/after diff to be a real, isolated win. A wider vocabulary pass
    ("textile", "synthetic", plus a new FEATURES slot for "water
    resistant"/"hand wash"/etc., covering the other 4 sessions below) was
    tried and reverted: those words also match ordinary customer_reply()
    attribute-disclosure text elsewhere (see slots.MATERIALS' comment),
    regressing 3 unrelated sessions for zero net gain on the sessions they
    targeted. The remaining assertions below document that reverted state
    on purpose, as a guard against re-adding those words without re-running
    the same before/after validation."""

    _TEMPLATE = "Actually, ignore my earlier preference. What I need is: {value}."

    def test_public_0072_faux_fur(self) -> None:
        extracted = extract_slots(self._TEMPLATE.format(value="Faux Fur"))
        self.assertEqual(extracted.get("material"), "faux fur")

    def test_public_0125_acrylic(self) -> None:
        extracted = extract_slots(self._TEMPLATE.format(value="100% Acrylic"))
        self.assertEqual(extracted.get("material"), "acrylic")

    def test_public_0003_water_resistant_deliberately_unextracted(self) -> None:
        self.assertEqual(extract_slots(self._TEMPLATE.format(value="Water Resistant")), {})

    def test_public_0023_hand_wash_only_deliberately_unextracted(self) -> None:
        self.assertEqual(extract_slots(self._TEMPLATE.format(value="Hand Wash Only")), {})

    def test_public_0038_textile_deliberately_unextracted(self) -> None:
        self.assertEqual(extract_slots(self._TEMPLATE.format(value="Textile")), {})

    def test_public_0068_imported_deliberately_unextracted(self) -> None:
        # "Imported" is near-universal catalog boilerplate carrying no
        # discriminative value -- never a vocabulary candidate.
        self.assertEqual(extract_slots(self._TEMPLATE.format(value="Imported")), {})

    def test_public_0186_synthetic_deliberately_unextracted(self) -> None:
        self.assertEqual(extract_slots(self._TEMPLATE.format(value="100% Synthetic")), {})


class SameDepartmentTest(unittest.TestCase):
    def test_true_for_related_categories(self) -> None:
        self.assertTrue(same_department("shoes", "boots"))

    def test_false_for_unrelated_categories(self) -> None:
        self.assertFalse(same_department("shoes", "earrings"))

    def test_false_for_unknown_word(self) -> None:
        self.assertFalse(same_department("shoes", "gadgets"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(same_department("Shoes", "BOOTS"))


class SizeDomainTest(unittest.TestCase):
    def test_letter(self) -> None:
        self.assertEqual(size_domain("medium"), "letter")
        self.assertEqual(size_domain("M"), "letter")

    def test_numeric(self) -> None:
        self.assertEqual(size_domain("9"), "numeric")

    def test_numeric_with_country_suffix(self) -> None:
        self.assertEqual(size_domain("9 us"), "numeric")

    def test_unknown(self) -> None:
        self.assertEqual(size_domain("xyz-size"), "unknown")


if __name__ == "__main__":
    unittest.main()

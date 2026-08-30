"""Pure unit tests for scripts/train_reranker_weights.py's fold-building and
fitting logic -- fast, no catalog/agent involved. The full CV pipeline
(scripts/train_reranker_weights.py itself) is exercised manually via
`python3 scripts/train_reranker_weights.py`, not in the regular unit-test
suite, since it runs the full agent across the public dev set."""

import unittest

from scripts.train_reranker_weights import FEATURE_NAMES, build_folds, decide, fit_weights

SCENARIO_COUNTS = {"buying": 80, "browsing": 80, "intent_override": 30, "boundary": 10}


def _fake_samples() -> list[dict]:
    samples = []
    for scenario, count in SCENARIO_COUNTS.items():
        for i in range(count):
            samples.append({"sample_id": f"{scenario}_{i}", "scenario_type": scenario})
    return samples


class BuildFoldsTest(unittest.TestCase):
    def test_every_sample_is_a_holdout_exactly_once(self) -> None:
        samples = _fake_samples()
        folds = build_folds(samples, n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)
        seen_as_holdout: list[str] = []
        for _, holdout in folds:
            seen_as_holdout.extend(s["sample_id"] for s in holdout)
        self.assertCountEqual(seen_as_holdout, [s["sample_id"] for s in samples])

    def test_no_sample_id_leaks_between_a_folds_train_and_holdout(self) -> None:
        samples = _fake_samples()
        for train, holdout in build_folds(samples, n_splits=5, seed=42):
            train_ids = {s["sample_id"] for s in train}
            holdout_ids = {s["sample_id"] for s in holdout}
            self.assertEqual(train_ids & holdout_ids, set())
            self.assertEqual(len(train) + len(holdout), len(samples))

    def test_stratification_keeps_scenario_proportions_roughly_even(self) -> None:
        samples = _fake_samples()
        for _, holdout in build_folds(samples, n_splits=5, seed=42):
            counts = {}
            for s in holdout:
                counts[s["scenario_type"]] = counts.get(s["scenario_type"], 0) + 1
            # 5-fold split of 80/80/30/10 should land close to 16/16/6/2 per
            # fold -- allow slack for StratifiedKFold's rounding rather than
            # asserting exact counts.
            for scenario, total in SCENARIO_COUNTS.items():
                expected = total / 5
                self.assertLessEqual(abs(counts.get(scenario, 0) - expected), 2)

    def test_deterministic_given_same_seed(self) -> None:
        samples = _fake_samples()
        folds_a = build_folds(samples, n_splits=5, seed=42)
        folds_b = build_folds(samples, n_splits=5, seed=42)
        ids_a = [[s["sample_id"] for s in holdout] for _, holdout in folds_a]
        ids_b = [[s["sample_id"] for s in holdout] for _, holdout in folds_b]
        self.assertEqual(ids_a, ids_b)


class FitWeightsTest(unittest.TestCase):
    def test_falls_back_to_config_weights_on_single_class(self) -> None:
        from agent_shopper.config import HEURISTIC_RERANK_WEIGHTS

        rows = [{**{name: 0.5 for name in FEATURE_NAMES}, "label": 0} for _ in range(5)]
        self.assertEqual(fit_weights(rows), dict(HEURISTIC_RERANK_WEIGHTS))

    def test_learns_positive_weight_on_a_perfectly_predictive_feature(self) -> None:
        rows = []
        for i in range(30):
            is_target = i % 3 == 0
            row = {name: 0.1 for name in FEATURE_NAMES}
            row["attr_match"] = 0.9 if is_target else 0.1
            row["label"] = int(is_target)
            rows.append(row)
        weights = fit_weights(rows)
        self.assertEqual(set(weights), set(FEATURE_NAMES))
        self.assertGreater(weights["attr_match"], 0.0)


class DecideTest(unittest.TestCase):
    def _fold(self, delta: float, flips_up: int = 0, flips_down: int = 0) -> dict:
        return {"technical_score_delta": delta, "hit_flips_up": flips_up, "hit_flips_down": flips_down}

    def test_adopts_on_clear_consistent_win(self) -> None:
        folds = [self._fold(0.01, flips_up=2) for _ in range(4)] + [self._fold(0.0)]
        decision = decide(folds)
        self.assertTrue(decision["adopt"])

    def test_does_not_adopt_on_single_fold_driven_improvement(self) -> None:
        # One big win, rest losses/ties -- exactly the "1-2 flips in one
        # fold is noise" case the decision rule exists to catch.
        folds = [self._fold(0.20, flips_up=5)] + [self._fold(-0.01, flips_down=1) for _ in range(4)]
        decision = decide(folds)
        self.assertFalse(decision["adopt"])

    def test_does_not_adopt_on_net_negative_flips_even_with_majority_wins(self) -> None:
        folds = [self._fold(0.001, flips_up=0, flips_down=1) for _ in range(3)] + [self._fold(-0.01) for _ in range(2)]
        decision = decide(folds)
        self.assertFalse(decision["adopt"])


if __name__ == "__main__":
    unittest.main()

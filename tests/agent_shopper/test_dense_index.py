"""Unit tests for the dense-route safety net (agent_shopper.dense_index):
offline enforcement and graceful degradation on model-load failure. Never
downloads or loads the real sentence-transformers model -- uses a fake
`sentence_transformers` module in sys.modules, same technique
tests/agent_shopper/test_cross_encoder_reranker.py already uses for the
sibling CrossEncoder loader."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent_shopper.dense_index import DenseIndex
from tests.agent_shopper.fixtures import make_catalog


class _FakeSTModule:
    """Minimal stand-in for the `sentence_transformers` module, exposing
    only the `SentenceTransformer` name dense_index._get_model imports."""

    def __init__(self, ctor) -> None:
        self.SentenceTransformer = ctor


class _FakeModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True):
        # Deterministic, content-independent vectors -- shape is all that
        # matters for these tests, not semantic quality.
        return np.ones((len(texts), self.dim), dtype=np.float32)


class DenseIndexSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _cache_path(self) -> Path:
        return Path(self._tmpdir.name) / "cache.npz"

    def test_local_files_only_sets_hf_offline_env_vars(self) -> None:
        def fake_ctor(*args, **kwargs):
            self.assertTrue(kwargs.get("local_files_only"))
            return _FakeModel()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(fake_ctor)}):
                DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=True)
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_local_files_only_false_does_not_force_offline_env_vars(self) -> None:
        def fake_ctor(*args, **kwargs):
            self.assertFalse(kwargs.get("local_files_only"))
            return _FakeModel()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(fake_ctor)}):
                DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=False)
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
            self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

    def test_load_failure_does_not_crash_construction(self) -> None:
        def raising_ctor(*args, **kwargs):
            raise OSError("simulated: model not found locally and local_files_only=True")

        with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(raising_ctor)}):
            # Must not raise -- this is called once at Agent() construction
            # and reused for every session; a crash here would fail the
            # entire submission, not just this one route.
            index = DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=True)
        self.assertTrue(index._model_load_failed)

    def test_search_returns_empty_list_after_load_failure_not_raise(self) -> None:
        def raising_ctor(*args, **kwargs):
            raise OSError("simulated failure")

        with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(raising_ctor)}):
            index = DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=True)
            results = index.search("running shoes", limit=10)
        self.assertEqual(results, [])

    def test_search_does_not_retry_a_known_failed_load(self) -> None:
        calls = {"n": 0}

        def raising_ctor(*args, **kwargs):
            calls["n"] += 1
            raise OSError("simulated failure")

        with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(raising_ctor)}):
            index = DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=True)
            index.search("running shoes", limit=10)
            index.search("leather boots", limit=10)
            index.search("silk dress", limit=10)
        # One attempt at construction time; search() must not re-attempt the
        # load on every call once it's known to be broken.
        self.assertEqual(calls["n"], 1)

    def test_successful_load_still_works_normally(self) -> None:
        def fake_ctor(*args, **kwargs):
            return _FakeModel(dim=4)

        with patch.dict("sys.modules", {"sentence_transformers": _FakeSTModule(fake_ctor)}):
            index = DenseIndex(self.catalog, cache_path=self._cache_path(), local_files_only=True)
            results = index.search("running shoes", limit=5)
        self.assertFalse(index._model_load_failed)
        self.assertIsInstance(results, list)
        for doc_index, score in results:
            self.assertIsInstance(doc_index, int)
            self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()

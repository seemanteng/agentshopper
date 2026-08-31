"""Unit tests for the dense-route safety net (agent_shopper.dense_index):
offline enforcement and graceful degradation on model-load failure. Never
downloads or loads the real sentence-transformers model -- uses a fake
`sentence_transformers` module in sys.modules, same technique
tests/agent_shopper/test_cross_encoder_reranker.py already uses for the
sibling CrossEncoder loader."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent_shopper.dense_index import DenseIndex, _model_cache_slug
from tests.agent_shopper.fixtures import make_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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


class ModelCacheSlugTest(unittest.TestCase):
    """_model_cache_slug is what keeps DenseIndex.default_cache_path portable
    once DENSE_MODEL_NAME defaults to an absolute local path rather than a
    Hugging Face id -- see dense_index.py's own comment on why embedding the
    full path directly would be both ugly and checkout-location-sensitive."""

    def test_hf_id_uses_the_slug_after_the_slash(self) -> None:
        self.assertEqual(_model_cache_slug("sentence-transformers/all-MiniLM-L6-v2"), "sentence-transformers_all-MiniLM-L6-v2")

    def test_existing_local_directory_uses_only_its_own_name(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            packaged = Path(d) / "models" / "dense" / "all-MiniLM-L6-v2"
            packaged.mkdir(parents=True)
            # Not the full path, and not sensitive to where `d` happens to be.
            self.assertEqual(_model_cache_slug(str(packaged)), "all-MiniLM-L6-v2")

    def test_nonexistent_path_falls_back_to_hf_id_style_slugging(self) -> None:
        # A path-shaped string that doesn't actually exist on disk (e.g. an
        # env-var override to a directory that hasn't been created yet) is
        # treated like an id, not silently misread as a real local checkout.
        self.assertEqual(_model_cache_slug("/nonexistent/models/dense/all-MiniLM-L6-v2"), "_nonexistent_models_dense_all-MiniLM-L6-v2")


class ProductionDefaultsTest(unittest.TestCase):
    """Proves what a genuinely fresh, zero-`AGENT_SHOPPER_*`-env-var judge
    process actually resolves DENSE_MODEL_NAME to -- mirrors
    test_cross_encoder_reranker.py's own ProductionDefaultsTest exactly,
    same reasoning: a subprocess with a scrubbed environment is what actually
    answers "what does a judge's process see," not an in-process reload."""

    def _run_in_subprocess(self, cwd: Path, env_overrides: dict[str, str] | None = None) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_SHOPPER_")}
        env.update(env_overrides or {})
        result = subprocess.run(
            [sys.executable, "-c", "import agent_shopper.config as c; print(c.DENSE_MODEL_NAME); print(c.DENSE_MODEL_LOCAL_FILES_ONLY)"],
            cwd=str(cwd), env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=f"subprocess failed: {result.stderr}")
        return result.stdout

    def test_zero_env_var_default_resolves_to_the_packaged_checkpoint(self) -> None:
        stdout = self._run_in_subprocess(REPO_ROOT)
        model_name, local_only = stdout.strip().splitlines()
        self.assertEqual(local_only, "True")
        # Module-relative, not CWD-relative: must resolve to an absolute path
        # under this repo's models/dense/ regardless of subprocess CWD.
        self.assertTrue(model_name.startswith(str(REPO_ROOT)), msg=model_name)
        self.assertIn("models/dense/all-MiniLM-L6-v2", model_name)
        self.assertTrue(Path(model_name).is_dir(), msg=f"packaged checkpoint missing at {model_name}")
        self.assertTrue((Path(model_name) / "manifest.json").is_file())

    def test_model_path_is_cwd_independent(self) -> None:
        # Same subprocess check from a working directory outside the
        # repository entirely -- proves the Path(__file__).resolve()-based
        # default doesn't depend on the caller's CWD.
        stdout = self._run_in_subprocess(Path(tempfile.gettempdir()), {"PYTHONPATH": str(REPO_ROOT)})
        model_name = stdout.strip().splitlines()[0]
        self.assertTrue(model_name.startswith(str(REPO_ROOT)), msg=model_name)

    def test_env_var_override_still_works(self) -> None:
        stdout = self._run_in_subprocess(REPO_ROOT, {"AGENT_SHOPPER_DENSE_MODEL": "sentence-transformers/all-MiniLM-L6-v2"})
        model_name = stdout.strip().splitlines()[0]
        self.assertEqual(model_name, "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()

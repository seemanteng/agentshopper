"""Field-aware dense semantic retrieval -- the "dense" route.

A frozen sentence-embedding model (no fine-tuning -- the competition spec
puts full base-model training out of scope but explicitly allows dense
retrieval with a frozen/local encoder), embedding each product's
title/attributes/category-path/description *separately* rather than one
concatenated document -- concatenation would bury a short but decisive field
(e.g. a one-word material mention) under a much longer description, the same
reasoning tfidf_index.py's field weighting and BM25_FIELD_WEIGHTS both
already apply via literal token repetition; dense embeddings don't have that
trick available, so separate per-field vectors combined by
config.DENSE_FIELD_WEIGHTS is the equivalent for this route.

Motivation: scripts/diagnose_retrieval.py's per-route Recall@10/50/100
breakdown showed the existing keyword (BM25) and vector (TF-IDF) routes are
highly overlapping and both cap out around 15-21% R@100 -- fusing two
lexical routes that already agree adds little. A dense route catches
paraphrase/synonym matches both miss entirely (e.g. "sneaker" query against
a "running shoe" title with zero token overlap).

Embeddings are computed once per catalog and cached to disk under
config.DENSE_CACHE_DIR, keyed by catalog size + first/last ASIN + model name
-- re-embedding 50k products x 4 fields on every Agent() construction would
cost real time on every eval/CV run that builds a fresh Agent (see
scripts/train_reranker_weights.py's _run_variant docstring on why Agent
construction cost matters to these scripts). The sentence-transformers/torch
import itself is deferred to first actual use (model load or cache miss),
not import time -- constructing a DenseIndex over an already-cached catalog
never needs torch loaded into the process at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from agent_shopper.catalog import Catalog
from agent_shopper.config import DENSE_CACHE_DIR, DENSE_FIELD_WEIGHTS, DENSE_MODEL_NAME
from agent_shopper.models import Product

_FIELD_NAMES = ("title", "attributes", "category", "description")


def _field_text(product: Product, field: str) -> str:
    if field == "title":
        return product.title
    if field == "attributes":
        return " ".join([*product.features, *(f"{k} {v}" for k, v in product.details.items())])
    if field == "category":
        return " ".join(product.categories)
    if field == "description":
        # First couple of sentences only, not the full blob -- a short
        # snippet is plenty for an embedding and keeps encode() fast; the
        # full description is exactly the kind of long, often-generic text
        # this route's field weighting deliberately downweights (see
        # config.DENSE_FIELD_WEIGHTS' comment).
        return " ".join(product.description[:2])
    raise ValueError(field)  # pragma: no cover -- _FIELD_NAMES is the only caller


def _catalog_fingerprint(catalog: Catalog) -> str:
    """Cheap catalog identity check for cache validity -- count plus first/
    last ASIN. Not a full content hash (hashing every title/feature/
    description of 50k products on every Agent() construction would itself
    cost real time, working against the whole point of caching); a changed
    catalog that happens to keep the same size and first/last ASIN would
    silently reuse a stale cache, but the competition catalog is fixed and
    read-only for the duration of a run, so this is a non-issue in practice."""
    h = hashlib.sha1()
    h.update(str(len(catalog.products)).encode())
    if catalog.products:
        h.update(catalog.products[0].parent_asin.encode())
        h.update(catalog.products[-1].parent_asin.encode())
    return h.hexdigest()[:12]


def default_cache_path(catalog: Catalog, model_name: str = DENSE_MODEL_NAME) -> Path:
    safe_model = model_name.replace("/", "_")
    return Path(DENSE_CACHE_DIR) / f".dense_cache_{safe_model}_{_catalog_fingerprint(catalog)}.npz"


class DenseIndex:
    def __init__(
        self, catalog: Catalog, model_name: str = DENSE_MODEL_NAME, cache_path: Path | str | None = None,
    ) -> None:
        self.catalog = catalog
        self.model_name = model_name
        self._model = None  # lazy -- see module docstring
        self._field_matrices: dict[str, np.ndarray] = {}
        self._cache_path = Path(cache_path) if cache_path is not None else default_cache_path(catalog, model_name)
        self._load_or_build()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # deferred: see module docstring

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_cache(self) -> bool:
        if not self._cache_path.exists():
            return False
        try:
            data = np.load(self._cache_path)
            if not all(field in data for field in _FIELD_NAMES):
                return False
            if data[_FIELD_NAMES[0]].shape[0] != len(self.catalog.products):
                return False
        except Exception:
            return False  # corrupt/unreadable cache -- rebuild rather than crash startup
        self._field_matrices = {field: data[field] for field in _FIELD_NAMES}
        return True

    def _load_or_build(self) -> None:
        if not self.catalog.products:
            self._field_matrices = {field: np.zeros((0, 1), dtype=np.float32) for field in _FIELD_NAMES}
            return
        if self._load_cache():
            return
        model = self._get_model()
        matrices: dict[str, np.ndarray] = {}
        for field in _FIELD_NAMES:
            texts = [_field_text(p, field) or " " for p in self.catalog.products]
            embeddings = model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            matrices[field] = np.asarray(embeddings, dtype=np.float32)
        self._field_matrices = matrices
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._cache_path, **matrices)
        except OSError:
            pass  # caching is a pure speed optimization -- never fail startup over a write error

    def search(self, query_text: str, limit: int) -> list[tuple[int, float]]:
        if not self.catalog.products or not query_text.strip():
            return []
        model = self._get_model()
        query_vec = np.asarray(
            model.encode([query_text], normalize_embeddings=True), dtype=np.float32,
        )[0]
        combined = np.zeros(len(self.catalog.products), dtype=np.float32)
        for field, weight in DENSE_FIELD_WEIGHTS.items():
            if weight <= 0:
                continue
            combined += weight * (self._field_matrices[field] @ query_vec)
        # argpartition for the top-N, then sort just that slice -- same
        # approach tfidf_index.TfidfIndex.search uses, avoids a full
        # O(n log n) sort over all 50k products per query.
        if limit >= len(combined):
            top_indices = combined.argsort()[::-1]
        else:
            partitioned = np.argpartition(-combined, limit)[:limit]
            top_indices = partitioned[np.argsort(-combined[partitioned])]
        return [(int(i), float(combined[i])) for i in top_indices if combined[i] > 0][:limit]

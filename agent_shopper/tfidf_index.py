"""TF-IDF + cosine similarity vector route.

A lightweight, in-memory, CPU-only stand-in for dense embeddings: no model
download, no GPU, no external vector DB -- just a sparse matrix fit once over
the 50k-product catalog. It catches near-synonym n-grams that raw BM25 term
matching misses (e.g. "sneaker" query against a "running shoe" title).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent_shopper.catalog import Catalog
from agent_shopper.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from agent_shopper.models import Product


def _document_text(product: Product) -> str:
    return " ".join([
        product.title,
        " ".join(product.features),
        " ".join(product.description),
        " ".join(product.categories),
        product.store,
    ])


class TfidfIndex:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
        )
        docs = [_document_text(p) for p in catalog.products]
        self.matrix = self.vectorizer.fit_transform(docs) if docs else None

    def search(self, query_text: str, limit: int) -> list[tuple[int, float]]:
        if self.matrix is None or not query_text.strip():
            return []
        query_vec = self.vectorizer.transform([query_text])
        if query_vec.nnz == 0:
            return []
        sims = cosine_similarity(query_vec, self.matrix)[0]
        # argpartition for the top-N, then sort just that slice -- avoids a
        # full O(n log n) sort over all 50k products per query.
        if limit >= len(sims):
            top_indices = sims.argsort()[::-1]
        else:
            import numpy as np

            partitioned = np.argpartition(-sims, limit)[:limit]
            top_indices = partitioned[np.argsort(-sims[partitioned])]
        return [(int(i), float(sims[i])) for i in top_indices if sims[i] > 0][:limit]

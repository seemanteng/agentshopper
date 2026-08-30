"""Hand-rolled BM25 keyword route.

Kept dependency-free (no rank_bm25) so field weighting and IDF caching are
fully under our control. Field weighting is applied by literal token
repetition when each document is built, since the BM25 formula itself has no
native per-field weight knob.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from agent_shopper.catalog import Catalog
from agent_shopper.config import BM25_B, BM25_FIELD_WEIGHTS, BM25_K1
from agent_shopper.models import Product
from agent_shopper.text_utils import terms


def _document_text(product: Product) -> list[str]:
    weighted: list[str] = []
    weighted.extend(terms(product.title) * BM25_FIELD_WEIGHTS["title"])
    weighted.extend(terms(" ".join(product.features)) * BM25_FIELD_WEIGHTS["features"])
    weighted.extend(terms(" ".join(product.description)) * BM25_FIELD_WEIGHTS["description"])
    weighted.extend(terms(product.store) * BM25_FIELD_WEIGHTS["store"])
    return weighted


class BM25Index:
    """Okapi BM25 over the catalog's title/features/description/store text."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.k1 = BM25_K1
        self.b = BM25_B
        self._doc_term_counts: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._df: dict[str, int] = defaultdict(int)
        self._idf: dict[str, float] = {}
        # Real inverted index, built once: term -> [doc_index, ...]. Looking
        # this up per query keeps search O(matching docs), not O(catalog).
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        n = len(self.catalog.products)
        for doc_index, product in enumerate(self.catalog.products):
            tokens = _document_text(product)
            counts = Counter(tokens)
            self._doc_term_counts.append(counts)
            self._doc_lengths.append(len(tokens))
            for term in counts:
                self._df[term] += 1
                self._postings[term].append(doc_index)
        self.avgdl = (sum(self._doc_lengths) / n) if n else 0.0
        for term, df in self._df.items():
            # Standard Robertson-Sparck Jones IDF with a +1 floor to stay
            # non-negative for very common terms.
            self._idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], doc_index: int) -> float:
        counts = self._doc_term_counts[doc_index]
        dl = self._doc_lengths[doc_index]
        if dl == 0 or not query_terms:
            return 0.0
        score = 0.0
        for term in query_terms:
            tf = counts.get(term)
            if not tf:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query_text: str, limit: int) -> list[tuple[int, float]]:
        """Returns [(doc_index, score), ...] sorted descending, non-zero only."""
        query_terms = list(dict.fromkeys(terms(query_text)))
        if not query_terms:
            return []
        # Only score documents containing at least one query term -- via the
        # precomputed inverted index, this is O(matching docs), not
        # O(catalog_size).
        candidate_docs: set[int] = set()
        for term in query_terms:
            candidate_docs.update(self._postings.get(term, ()))
        scored = [(i, self.score(query_terms, i)) for i in candidate_docs]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

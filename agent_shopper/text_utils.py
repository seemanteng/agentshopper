"""Shared tokenization used by BM25, TF-IDF, and slot extraction."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "need", "needs", "am", "im", "have", "has", "do", "does", "any", "your",
})


def flatten_text(value: object) -> str:
    """Turn a catalog field (str, list, dict, None) into one lowercase-able
    string, the same shape the official evaluator's searchable_text() uses."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def terms(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = tokenize(text)
    if not drop_stopwords:
        return toks
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def unique_terms(text: str, limit: int | None = None) -> list[str]:
    out = list(dict.fromkeys(terms(text)))
    return out[:limit] if limit else out

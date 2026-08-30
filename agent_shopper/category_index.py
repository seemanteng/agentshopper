"""Structured category/attribute/price filter -- the "category route".

This is the high-precision track for Buying intent: an AND-filter over
categories, free-text attribute mentions (material/color/size/brand/style/
use_case), and a parsed price window, rather than any ranked-retrieval
scoring. It also exposes `attr_match_fraction`, a soft per-product match
score reused by the heuristic reranker and the clarify-attribute entropy
calculation.
"""

from __future__ import annotations

from agent_shopper.catalog import Catalog
from agent_shopper.models import Product, SlotSet

# Slots that are matched as "does this text appear somewhere in the
# product's searchable text" rather than structurally. use_case/style/brand
# have no dedicated catalog field, so free-text containment is the only
# option available without training a classifier.
_TEXT_MATCH_SLOTS = ("material", "color", "size", "style", "brand", "use_case")

# The catalog's categories[0] is always this generic department root (see
# evaluator/local_evaluator.py's own coarse_category, which excludes the
# same strings). Matching against it would make e.g. "jewelry" or "shoes"
# match nearly every product in a Clothing_Shoes_and_Jewelry-only catalog.
_EXCLUDED_CATEGORY_VALUES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def _product_text(product: Product) -> str:
    return " ".join([
        product.title.lower(),
        " ".join(product.features).lower(),
        " ".join(product.description).lower(),
        " ".join(f"{k} {v}" for k, v in product.details.items()).lower(),
    ])


def _values(slot_value: object) -> list[str]:
    if slot_value is None:
        return []
    if isinstance(slot_value, list):
        return [str(v) for v in slot_value]
    return [str(slot_value)]


def matches_category(product: Product, value: str) -> bool:
    needle = value.lower().strip()
    if not needle:
        return True
    for cat in product.categories:
        cat_l = cat.lower()
        if cat_l in _EXCLUDED_CATEGORY_VALUES:
            continue
        if needle in cat_l:
            return True
    return needle in product.title.lower()


def matches_text_slot(product: Product, values: list[str], text: str | None = None) -> bool:
    if not values:
        return True
    haystack = text if text is not None else _product_text(product)
    return any(v.lower().strip() in haystack for v in values if v.strip())


def matches_budget(product: Product, budget: tuple[float | None, float | None] | None) -> bool:
    if budget is None:
        return True
    if product.price is None:
        return False
    lo, hi = budget
    if lo is not None and product.price < lo:
        return False
    if hi is not None and product.price > hi:
        return False
    return True


def filter_products(catalog: Catalog, slots: SlotSet, skip_slots: tuple[str, ...] = ()) -> list[int]:
    """Returns doc indices (into catalog.products) satisfying every filled,
    non-skipped slot. `skip_slots` lets the orchestrator relax the gate by
    dropping specific slots without touching SessionState."""
    checks: list[tuple[str, object]] = []
    if slots.category and "category" not in skip_slots:
        checks.append(("category", slots.category))
    for name in _TEXT_MATCH_SLOTS:
        if name in skip_slots:
            continue
        value = getattr(slots, name)
        if value:
            checks.append((name, _values(value)))
    if slots.budget and "budget" not in skip_slots:
        checks.append(("budget", slots.budget))

    if not checks:
        return list(range(len(catalog.products)))

    result = []
    for index, product in enumerate(catalog.products):
        ok = True
        text = _product_text(product)
        for name, value in checks:
            if name == "category":
                ok = matches_category(product, value)
            elif name == "budget":
                ok = matches_budget(product, value)
            else:
                ok = matches_text_slot(product, value, text)
            if not ok:
                break
        if ok:
            result.append(index)
    return result


def rank_by_rating(catalog: Catalog, doc_indices: list[int], limit: int) -> list[tuple[int, float]]:
    """Secondary ranking for the category route's exact-match set: higher
    rating (shrunk by rating_number confidence) first."""
    scored = []
    for i in doc_indices:
        product = catalog.products[i]
        rating = product.average_rating or 0.0
        confidence = min(1.0, (product.rating_number or 0) / 20.0)
        scored.append((i, 0.5 + confidence * (rating / 5.0 - 0.5) if product.average_rating else 0.4))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def attr_match_fraction(product: Product, slots: SlotSet) -> float:
    """Soft match score in [0, 1]: fraction of filled slots this product
    satisfies. Used by the heuristic reranker and clarify-attribute scoring,
    independent of the hard AND-filter above."""
    filled = slots.filled_slots()
    if not filled:
        return 0.0
    text = _product_text(product)
    hits = 0
    for name in filled:
        if name == "category":
            hits += int(matches_category(product, slots.category))
        elif name == "feature":
            hits += int(matches_text_slot(product, slots.feature, text))
        elif name == "budget":
            hits += int(matches_budget(product, slots.budget))
        else:
            hits += int(matches_text_slot(product, _values(getattr(slots, name)), text))
    return hits / len(filled)

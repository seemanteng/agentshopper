"""Pillar II: extracts slot values from a single turn's user message.

Purely rule/keyword based (no training in scope). This module only
*extracts what's new this turn*; merging that into a session's accumulated
SlotSet -- including the Accumulation vs Override decision -- is
dialog_policy.merge_slot's job, not this module's.
"""

from __future__ import annotations

import re

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "suede", "canvas", "linen", "cashmere",
    "velvet", "mesh", "rubber", "polyurethane", "faux leather", "satin",
    # "faux fur"/"acrylic" close 2 of the 7 intent_override sessions
    # (data/public_set.jsonl public_0072/public_0125) whose override
    # message extracted zero slots before this change -- see scripts/
    # diagnose_intent_override.py. Confirmed via scripts/run_local_eval.py
    # session-level before/after diff: public_0072 flips MISS->HIT,
    # public_0125's rank improves 10->3, with no regression elsewhere in
    # the 200-session public set. A broader vocabulary pass ("textile",
    # "synthetic", and a new FEATURES slot for "water resistant"/"hand
    # wash"/etc.) was tried alongside these two and reverted: every one of
    # those additions also matches ordinary customer_reply() attribute-
    # disclosure text (it draws from the same catalog-snippet mechanism
    # intent_card() uses for the override's new_value -- see evaluator/
    # local_evaluator.py), so it silently changed slot state on
    # buying/browsing/boundary turns that previously extracted nothing,
    # regressing 3 unrelated sessions (public_0035, public_0084,
    # public_0138) for zero net gain on the override sessions it targeted.
    "faux fur", "acrylic",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "tan", "gold",
    "silver", "multicolor", "teal", "maroon", "ivory", "cream", "burgundy",
)
SIZE_WORDS = ("small", "medium", "large", "petite", "plus size", "wide", "narrow", "extra large", "extra small")
SIZE_RE = re.compile(r"\b(?:size\s*)?(x{0,2}[sml]|\d{1,2}(?:\.\d)?(?:\s*(?:us|uk|eu))?)\b", re.I)

# Coarse size "domain" classifier -- letter (S/M/L/...) vs numeric
# (shoe/waist/etc). Used only to decide whether two size mentions are the
# *same kind* of measurement (dialog_policy._same_size_domain), never to
# convert between them -- the competition simulator discloses size
# constraints as whatever literal string the target product's own catalog
# text contains, never a normalized/converted value, so no US/UK/EU
# shoe-size equivalence table is needed.
_LETTER_SIZES = set(SIZE_WORDS) | {"xxs", "xs", "s", "m", "l", "xl", "xxl"}
_NUMERIC_SIZE_RE = re.compile(r"^\d{1,2}(?:\.\d)?(?:\s*(?:us|uk|eu))?$", re.I)


def size_domain(value: object) -> str:
    """Returns "letter", "numeric", or "unknown" for a size value."""
    text = str(value).strip().lower()
    if text in _LETTER_SIZES:
        return "letter"
    if _NUMERIC_SIZE_RE.match(text):
        return "numeric"
    return "unknown"


STYLES = (
    "casual", "formal", "athletic", "vintage", "classic", "modern", "boho",
    "bohemian", "minimalist", "sporty", "elegant", "chic", "trendy",
    "slim fit", "loose fit", "relaxed fit", "fitted", "oversized",
)
USE_CASES = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "wedding",
    "travel", "yoga", "office", "school", "beach", "party", "everyday",
    "workout", "casual wear", "formal wear", "summer",
)
CATEGORY_WORDS = (
    "earrings", "necklace", "bracelet", "ring", "jewelry", "watch",
    "shoes", "sneakers", "boots", "sandals", "heels", "flats",
    "dress", "shirt", "t-shirt", "blouse", "pants", "jeans", "shorts",
    "jacket", "coat", "sweater", "hoodie", "skirt", "socks", "hat",
    "bag", "handbag", "belt", "scarf", "gloves", "swimsuit", "leggings",
)

# Coarse "department" bucket per category word -- used only to decide
# whether a category override should invalidate style/feature (see
# dialog_policy._clear_incompatible_attributes), not for retrieval/
# matching. A closed table over this module's own fixed CATEGORY_WORDS
# vocabulary, not the unbounded catalog category tree -- extract_slots can
# only ever emit one of these words for "category" (find_word returns a
# single best match). Covers all 33 CATEGORY_WORDS exhaustively.
CATEGORY_DEPARTMENT = {
    "earrings": "jewelry", "necklace": "jewelry", "bracelet": "jewelry",
    "ring": "jewelry", "jewelry": "jewelry", "watch": "jewelry",
    "shoes": "footwear", "sneakers": "footwear", "boots": "footwear",
    "sandals": "footwear", "heels": "footwear", "flats": "footwear",
    "dress": "apparel", "shirt": "apparel", "t-shirt": "apparel", "blouse": "apparel",
    "pants": "apparel", "jeans": "apparel", "shorts": "apparel", "jacket": "apparel",
    "coat": "apparel", "sweater": "apparel", "hoodie": "apparel", "skirt": "apparel",
    "leggings": "apparel", "swimsuit": "apparel",
    "socks": "accessories", "hat": "accessories", "belt": "accessories",
    "scarf": "accessories", "gloves": "accessories",
    "bag": "bags", "handbag": "bags",
}


def same_department(old_category: str, new_category: str) -> bool:
    """Whether two extracted category words plausibly share style/feature
    applicability. Unknown words (outside CATEGORY_WORDS) conservatively
    default to "different" -- matches the always-clear behavior used when
    we can't classify a category word at all."""
    old_dept = CATEGORY_DEPARTMENT.get(old_category.lower())
    new_dept = CATEGORY_DEPARTMENT.get(new_category.lower())
    return old_dept is not None and old_dept == new_dept


BRAND_RE = re.compile(r"\bbrand(?:\s*[:=]|\s+is)?\s+([A-Za-z0-9&'\- ]{2,30})", re.I)
BRAND_BY_RE = re.compile(r"\bby\s+([A-Z][A-Za-z0-9&'\- ]{1,29})\b")

_RANGE_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_UNDER_RE = re.compile(r"(?:under|less than|no more than|below|at most|no more than)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_OVER_RE = re.compile(r"(?:over|more than|at least|above)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_AROUND_RE = re.compile(r"(?:around|about|approximately)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_BARE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def parse_budget(text: str) -> tuple[float | None, float | None] | None:
    lowered = text.lower()
    m = _RANGE_RE.search(lowered)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b))
    m = _UNDER_RE.search(lowered)
    if m:
        return (None, float(m.group(1)))
    m = _OVER_RE.search(lowered)
    if m:
        return (float(m.group(1)), None)
    m = _AROUND_RE.search(lowered)
    if m:
        v = float(m.group(1))
        return (round(v * 0.8, 2), round(v * 1.2, 2))
    m = _BARE_RE.search(text)
    if m:
        v = float(m.group(1))
        return (round(v * 0.8, 2), round(v * 1.2, 2))
    return None


def find_word(text: str, vocabulary: tuple[str, ...]) -> str | None:
    """Finds the best single vocabulary match in `text`.

    Multi-word phrases ("faux leather") are checked first, longest first, so
    a compound term always beats a shorter word it happens to contain. If no
    phrase matches, the *last-mentioned* single word wins -- e.g. "running
    shoes for hiking" should extract "hiking" (the actual stated purpose),
    not "running" merely because it appears first.
    """
    lowered = text.lower()
    multi_word = [w for w in vocabulary if " " in w]
    for word in sorted(multi_word, key=len, reverse=True):
        if re.search(_word_pattern(word), lowered):
            return word

    best: str | None = None
    best_pos = -1
    for word in vocabulary:
        if " " in word:
            continue
        match = re.search(_word_pattern(word), lowered)
        if match and match.start() > best_pos:
            best, best_pos = word, match.start()
    return best


def _word_pattern(word: str) -> str:
    # Tolerates simple English plurals ("belt"/"belts", "watch"/"watches")
    # so generated dialog text ("I'm looking for Watches...") still matches
    # our singular-form vocabulary lists.
    return rf"\b{re.escape(word)}(?:e?s)?\b"


def _find_all_words(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [w for w in vocabulary if re.search(_word_pattern(w), lowered)]


def extract_slots(message: str) -> dict[str, object]:
    """Returns only the slots detected in this message (values in the same
    shape SlotSet expects: str for point slots, list[str] for 'feature')."""
    extracted: dict[str, object] = {}

    budget = parse_budget(message)
    if budget is not None:
        extracted["budget"] = budget

    material = find_word(message, MATERIALS)
    if material:
        extracted["material"] = material

    colors = _find_all_words(message, COLORS)
    if colors:
        extracted["color"] = colors[0] if len(colors) == 1 else colors

    size = find_word(message, SIZE_WORDS)
    if not size:
        m = SIZE_RE.search(message)
        if m and re.search(r"\bsize\b", message, re.I):
            size = m.group(1)
    if size:
        extracted["size"] = size

    style = find_word(message, STYLES)
    if style:
        extracted["style"] = style

    use_case = find_word(message, USE_CASES)
    if use_case:
        extracted["use_case"] = use_case

    category = find_word(message, CATEGORY_WORDS)
    if category:
        extracted["category"] = category

    brand_match = BRAND_RE.search(message) or BRAND_BY_RE.search(message)
    if brand_match:
        extracted["brand"] = brand_match.group(1).strip()

    return extracted

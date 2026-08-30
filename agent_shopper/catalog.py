"""Loads catalog.jsonl once and normalizes it into Product rows.

Pure in-memory, read-only, no mutation of the source file -- per the
competition's "catalog is strictly read-only" constraint.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_shopper.models import Product


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def _as_dict(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
    return {}


def parse_product(row: dict) -> Product:
    return Product(
        parent_asin=str(row["parent_asin"]),
        title=str(row.get("title") or ""),
        features=_as_list(row.get("features")),
        description=_as_list(row.get("description")),
        price=_to_float(row.get("price")),
        categories=_as_list(row.get("categories")),
        details=_as_dict(row.get("details")),
        average_rating=_to_float(row.get("average_rating")),
        rating_number=_to_int(row.get("rating_number")),
        store=str(row.get("store") or ""),
    )


class Catalog:
    """In-memory catalog: a list of Product plus an id->index lookup."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.path = Path(catalog_path)
        self.products: list[Product] = []
        self.by_asin: dict[str, Product] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = parse_product(json.loads(line))
                self.products.append(product)
                self.by_asin[product.parent_asin] = product

    def __len__(self) -> int:
        return len(self.products)

    def get(self, parent_asin: str) -> Product | None:
        return self.by_asin.get(parent_asin)

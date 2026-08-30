"""Small shared fixture catalog for agent_shopper unit tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_shopper.catalog import Catalog

SAMPLE_PRODUCTS = [
    {
        "parent_asin": "B001", "title": "Blue Cotton Running Shoes for Men",
        "features": ["Breathable mesh upper", "Cotton lining"], "description": ["Great for daily running."],
        "price": 45.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic"],
        "details": {"Department": "Mens"}, "average_rating": 4.5, "rating_number": 120, "store": "TrailRun",
    },
    {
        "parent_asin": "B002", "title": "Black Leather Boots for Winter",
        "features": ["Genuine leather", "Warm lining"], "description": ["Sturdy winter boots."],
        "price": 89.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Boots"],
        "details": {"Department": "Mens"}, "average_rating": 4.2, "rating_number": 50, "store": "BootCo",
    },
    {
        "parent_asin": "B003", "title": "Red Silk Evening Dress",
        "features": ["100% silk", "Elegant formal wear"], "description": ["Perfect for a wedding."],
        "price": 60.0, "categories": ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
        "details": {"Department": "Womens"}, "average_rating": 4.0, "rating_number": 30, "store": "Elegance",
    },
    {
        "parent_asin": "B004", "title": "White Cotton Casual T-Shirt",
        "features": ["100% cotton", "Casual everyday wear"], "description": ["Soft and breathable."],
        "price": 15.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
        "details": {"Department": "Mens"}, "average_rating": 3.8, "rating_number": 200, "store": "BasicWear",
    },
    {
        "parent_asin": "B005", "title": "Gold Hoop Earrings Lightweight",
        "features": ["Hypoallergenic stainless steel", "Gold plated"], "description": ["Great gift for her."],
        "price": 25.0, "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings"],
        "details": {"Department": "Womens"}, "average_rating": 4.7, "rating_number": 500, "store": "Spirit Hoops",
    },
    {
        "parent_asin": "B006", "title": "Green Wool Sweater Warm Winter",
        "features": ["100% wool", "Warm winter wear"], "description": ["Cozy sweater for cold days."],
        "price": 55.0, "categories": ["Clothing, Shoes & Jewelry", "Women", "Sweaters"],
        "details": {"Department": "Womens"}, "average_rating": 4.1, "rating_number": 80, "store": "Cozy",
    },
    {
        "parent_asin": "B007", "title": "Black Nylon Running Shoes Athletic",
        "features": ["Lightweight nylon", "Great for running"], "description": ["Athletic performance shoes."],
        "price": 70.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic"],
        "details": {"Department": "Mens"}, "average_rating": 4.6, "rating_number": 300, "store": "TrailRun",
    },
    {
        "parent_asin": "B008", "title": "Silver Necklace Minimalist Style",
        "features": ["Sterling silver", "Minimalist design"], "description": ["Elegant everyday necklace."],
        "price": 40.0, "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
        "details": {"Department": "Womens"}, "average_rating": 4.3, "rating_number": 60, "store": "Silverline",
    },
]


def write_catalog_file(products: list[dict] = SAMPLE_PRODUCTS) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in products:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


def make_catalog(products: list[dict] = SAMPLE_PRODUCTS) -> Catalog:
    path = write_catalog_file(products)
    try:
        return Catalog(path)
    finally:
        path.unlink(missing_ok=True)

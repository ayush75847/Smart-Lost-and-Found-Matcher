from __future__ import annotations

import re

from algorithms.pattern_recognition import normalize_text

ITEMS = {
    "wallet": "wallet",
    "purse": "wallet",
    "phone": "phone",
    "cellphone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "backpack": "backpack",
    "rucksack": "backpack",
    "bag": "bag",
    "headphones": "headphones",
    "earbuds": "earbuds",
    "keys": "keys",
    "key": "keys",
    "laptop": "laptop",
    "notebook": "notebook",
    "umbrella": "umbrella",
    "bottle": "water bottle",
    "watch": "watch",
    "glasses": "glasses",
    "spectacles": "glasses",
}
CATEGORIES = {
    "wallet": "Accessories",
    "phone": "Electronics",
    "backpack": "Bags",
    "bag": "Bags",
    "headphones": "Electronics",
    "earbuds": "Electronics",
    "keys": "Accessories",
    "laptop": "Electronics",
    "notebook": "Stationery",
    "umbrella": "Accessories",
    "water bottle": "Drinkware",
    "watch": "Accessories",
    "glasses": "Accessories",
}
COLORS = {
    "black": "Black",
    "dark": "Dark",
    "white": "White",
    "blue": "Blue",
    "navy": "Navy",
    "red": "Red",
    "green": "Green",
    "yellow": "Yellow",
    "brown": "Brown",
    "gray": "Gray",
    "grey": "Gray",
    "pink": "Pink",
    "purple": "Purple",
}
MATERIALS = {
    "leather": "Leather",
    "leatherette": "Leatherette",
    "canvas": "Canvas",
    "denim": "Denim",
    "steel": "Steel",
    "stainless": "Steel",
    "plastic": "Plastic",
    "cotton": "Cotton",
    "wooden": "Wood",
    "wood": "Wood",
}
BRANDS = [
    "wildcraft", "samsung", "apple", "oneplus", "boat", "sony", "jbl",
    "adidas", "nike", "puma", "hp", "dell", "lenovo", "soundway",
]


def _find_keyword(text: str, values: dict[str, str]) -> str | None:
    for key, value in values.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            return value
    return None


def analyze_description(description: str) -> dict:
    text = normalize_text(description)
    item_name = _find_keyword(text, ITEMS)
    category = CATEGORIES.get(item_name) if item_name else None
    color = _find_keyword(text, COLORS)
    material = _find_keyword(text, MATERIALS)
    brand = next(
        (brand.title() for brand in BRANDS if re.search(rf"\b{re.escape(brand)}\b", text)),
        None,
    )
    location = None
    match = re.search(
        r"\b(?:near|outside|inside|at|beside|by|around|in)\s+(?:the\s+)?"
        r"([a-z0-9][a-z0-9\s'-]{1,100}?)(?:[.!?,]|$)",
        text,
    )
    if match:
        location = re.sub(r"\s+", " ", match.group(1)).strip()
        location = re.sub(r"\b(?:yesterday|today|last night|this morning)$", "", location).strip()
        if not location:
            location = None

    attributes = {
        "item_name": item_name,
        "category": category,
        "color": color,
        "material": material,
        "brand": brand,
        "location": location,
    }
    found = sum(value is not None for value in attributes.values())
    notes = [
        "Extracted with deterministic Python rules; no paid API or language model was used.",
        "Unrecognized or missing attributes are left blank rather than inferred.",
    ]
    if found == 0:
        notes.append("No supported item attributes were recognized in this description.")
    return {"source": "deterministic_rules", "attributes": attributes, "confidence_notes": notes}

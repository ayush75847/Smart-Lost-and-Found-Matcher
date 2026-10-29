from __future__ import annotations

import re
from difflib import SequenceMatcher


STOPWORDS = {
    "a", "about", "an", "and", "at", "by", "for", "found", "from", "in",
    "inside", "is", "it", "lost", "near", "of", "on", "outside", "the",
    "to", "was", "with", "my", "this", "that", "item",
}

SYNONYMS = {
    "dark": "black",
    "ebony": "black",
    "jetblack": "black",
    "navy": "blue",
    "grey": "gray",
    "cellphone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "spectacles": "glasses",
    "eyeglasses": "glasses",
    "rucksack": "backpack",
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.casefold().strip()
    value = re.sub(r"[\u2018\u2019]", "'", value)
    value = re.sub(r"[^a-z0-9\s'-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str | None) -> set[str]:
    words = normalize_text(value).split()
    result: set[str] = set()
    for word in words:
        token = re.sub(r"(^[-']+|[-']+$)", "", word)
        if not token or token in STOPWORDS:
            continue
        result.add(SYNONYMS.get(token, token))
    return result


def text_similarity(left: str | None, right: str | None) -> float | None:
    """Return a bounded, explainable lexical similarity; missing is unavailable."""
    if not normalize_text(left) or not normalize_text(right):
        return None

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return None
    if left_tokens == right_tokens:
        return 1.0

    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    fuzzy = SequenceMatcher(
        None, normalize_text(left), normalize_text(right), autojunk=False
    ).ratio()
    return round(max(jaccard, fuzzy if fuzzy >= 0.56 else 0.0), 4)


def pattern_labels(lost: dict, found: dict) -> tuple[list[str], list[str]]:
    """Describe attribute-level agreements without revealing private features."""
    labels = (
        ("category", "Category"),
        ("name", "Item name"),
        ("color", "Color"),
        ("brand", "Brand"),
        ("location", "Location"),
        ("description", "Description"),
    )
    matched: list[str] = []
    different: list[str] = []
    for key, label in labels:
        left = lost.get(key)
        right = found.get(key)
        similarity = text_similarity(str(left) if left else None, str(right) if right else None)
        if similarity is None:
            continue
        if similarity >= 0.45:
            matched.append(label)
        else:
            different.append(label)
    return matched, different

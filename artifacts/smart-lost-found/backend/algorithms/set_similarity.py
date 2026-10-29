from __future__ import annotations

from collections.abc import Iterable

from algorithms.pattern_recognition import tokenize


def _as_tokens(values: str | Iterable[str]) -> set[str]:
    if isinstance(values, str):
        return tokenize(values)
    tokens: set[str] = set()
    for value in values:
        tokens.update(tokenize(str(value)))
    return tokens


def jaccard_details(
    set_a: str | Iterable[str], set_b: str | Iterable[str]
) -> dict[str, object]:
    """Use actual normalized Python sets; two empty sets have no evidence."""
    a = _as_tokens(set_a)
    b = _as_tokens(set_b)
    intersection = a & b
    union = a | b
    score = (len(intersection) / len(union)) if union else None
    return {
        "set_a": sorted(a),
        "set_b": sorted(b),
        "intersection": sorted(intersection),
        "union": sorted(union),
        "similarity": round(score, 4) if score is not None else None,
    }


def jaccard_similarity(set_a: str | Iterable[str], set_b: str | Iterable[str]) -> float | None:
    return jaccard_details(set_a, set_b)["similarity"]  # type: ignore[return-value]

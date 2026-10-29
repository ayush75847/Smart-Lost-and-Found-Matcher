from __future__ import annotations

from datetime import date, datetime
from typing import Any

from algorithms.pattern_recognition import text_similarity
from algorithms.set_similarity import jaccard_details
from schemas import DEFAULT_WEIGHTS


LABELS = {
    "category": "Category",
    "name": "Item name",
    "color": "Color",
    "brand": "Brand",
    "location": "Location",
    "date": "Date",
    "description": "Description",
}


def _value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def date_similarity(left: Any, right: Any, window_days: int = 30) -> float | None:
    lost_date = _as_date(left)
    found_date = _as_date(right)
    if lost_date is None or found_date is None:
        return None
    delta = abs((found_date - lost_date).days)
    return round(max(0.0, 1.0 - delta / max(window_days, 1)), 4)


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    supplied = weights or DEFAULT_WEIGHTS
    values = {key: max(0.0, float(value)) for key, value in supplied.items() if key in LABELS}
    for key, value in DEFAULT_WEIGHTS.items():
        values.setdefault(key, value)
    total = sum(values.values())
    if total <= 0:
        raise ValueError("At least one comparison weight must be greater than zero.")
    return {key: value / total for key, value in values.items()}


def score_pair(
    lost: Any,
    found: Any,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    normalized = normalize_weights(weights)
    similarities: dict[str, float | None] = {
        "category": text_similarity(_value(lost, "category"), _value(found, "category")),
        "name": text_similarity(_value(lost, "name"), _value(found, "name")),
        "color": text_similarity(_value(lost, "color"), _value(found, "color")),
        "brand": text_similarity(_value(lost, "brand"), _value(found, "brand")),
        "location": text_similarity(_value(lost, "location"), _value(found, "location")),
        "date": date_similarity(
            _value(lost, "report_date") or _value(lost, "reportDate"),
            _value(found, "report_date") or _value(found, "reportDate"),
        ),
        "description": text_similarity(
            _value(lost, "description"), _value(found, "description")
        ),
    }
    available_weight = sum(
        normalized[key] for key, value in similarities.items() if value is not None
    )
    if available_weight <= 0:
        return {
            "score": 0.0,
            "evidence_coverage": 0.0,
            "breakdown": [
                {
                    "attribute": key,
                    "label": LABELS[key],
                    "weight": round(normalized[key] * 100, 2),
                    "similarity": None,
                    "contribution": None,
                    "available": False,
                    "explanation": "Not enough information to compare this attribute.",
                }
                for key in LABELS
            ],
        }

    score = sum(
        normalized[key] * value
        for key, value in similarities.items()
        if value is not None
    ) / available_weight
    breakdown: list[dict[str, Any]] = []
    for key, label in LABELS.items():
        similarity = similarities[key]
        weight = normalized[key]
        contribution = (weight * similarity / available_weight * 100) if similarity is not None else None
        if similarity is None:
            explanation = "Unavailable; excluded from the weighted average."
        else:
            explanation = (
                f"{weight * 100:.1f}% weight × {similarity * 100:.1f}% similarity; "
                f"{contribution:.1f} score points after renormalization."
            )
        breakdown.append(
            {
                "attribute": key,
                "label": label,
                "weight": round(weight * 100, 2),
                "similarity": similarity,
                "contribution": round(contribution, 2) if contribution is not None else None,
                "available": similarity is not None,
                "explanation": explanation,
            }
        )

    evidence_coverage = available_weight * 100
    lost_description = str(_value(lost, "description") or "")
    found_description = str(_value(found, "description") or "")
    jaccard = jaccard_details(lost_description, found_description)
    return {
        "score": round(min(100.0, max(0.0, score * 100)), 2),
        "evidence_coverage": round(evidence_coverage, 2),
        "breakdown": breakdown,
        "jaccard": jaccard,
        "similarities": similarities,
    }


def grade_for(score: float) -> str:
    if score >= 85:
        return "Strong similarity"
    if score >= 65:
        return "Moderate similarity"
    if score >= 40:
        return "Weak similarity"
    return "Low similarity"

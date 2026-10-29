from __future__ import annotations

from datetime import date
from typing import Any


def _get(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def candidate_validation(
    lost_item: Any,
    found_item: Any,
    date_tolerance_days: int = 1,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _get(lost_item, "id") == _get(found_item, "id"):
        reasons.append("A report cannot match itself.")
    if _get(lost_item, "type") != "lost":
        reasons.append("The selected report must be a lost report.")
    if _get(found_item, "type") != "found":
        reasons.append("A lost report can only match a found report.")
    if _get(lost_item, "status") != "active":
        reasons.append("Resolved lost reports cannot be matched.")
    if _get(found_item, "status") != "active":
        reasons.append("Resolved found reports cannot be matched.")

    lost_date = _get(lost_item, "report_date") or _get(lost_item, "reportDate")
    found_date = _get(found_item, "report_date") or _get(found_item, "reportDate")
    if lost_date and found_date:
        if isinstance(lost_date, str):
            lost_date = date.fromisoformat(lost_date)
        if isinstance(found_date, str):
            found_date = date.fromisoformat(found_date)
        if (found_date - lost_date).days < -max(0, date_tolerance_days):
            reasons.append(
                f"Found date is {(lost_date - found_date).days} days before the lost date "
                f"(tolerance: {date_tolerance_days} days)."
            )
    return not reasons, reasons


def is_valid_candidate(
    lost_item: Any,
    found_item: Any,
    date_tolerance_days: int = 1,
) -> bool:
    """Boolean eligibility predicate used by conditional-rule demonstrations."""
    return candidate_validation(lost_item, found_item, date_tolerance_days)[0]


def assert_approved_claim_is_valid(lost_item: Any, found_item: Any) -> None:
    valid, reasons = candidate_validation(lost_item, found_item, date_tolerance_days=30)
    if not valid:
        raise ValueError("Claim cannot be approved: " + " ".join(reasons))
    if _get(lost_item, "id") == _get(found_item, "id"):
        raise ValueError("A report cannot match itself.")

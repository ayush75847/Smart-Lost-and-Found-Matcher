from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Any


def rank_top_k(candidates: Iterable[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Sort by score descending, then by the stable report ID ascending."""
    def report_id(candidate: dict[str, Any]) -> int:
        item = candidate["item"]
        return int(item["id"] if isinstance(item, dict) else item.id)

    return heapq.nsmallest(
        max(0, k),
        candidates,
        key=lambda candidate: (
            -float(candidate["score"]),
            report_id(candidate),
        ),
    )


def maximum_score(candidates: Iterable[dict[str, Any]]) -> float | None:
    scores = [float(candidate["score"]) for candidate in candidates]
    return max(scores) if scores else None

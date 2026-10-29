from __future__ import annotations

import statistics
import time
from datetime import date
from typing import Any

from algorithms.heuristic import grade_for, score_pair
from algorithms.invariants import candidate_validation
from algorithms.pattern_recognition import text_similarity
from algorithms.ranking import maximum_score, rank_top_k
from schemas import (
    AlgorithmRun,
    CategoryCount,
    CompareResponse,
    DecisionStep,
    InvariantCheck,
    ItemOut,
    MatchRecommendation,
    MatchRequest,
)


def _get(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _item_json(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return {
        "id": item.id,
        "type": item.type,
        "name": item.name,
        "category": item.category,
        "brand": item.brand,
        "color": item.color,
        "description": item.description,
        "location": item.location,
        "report_date": item.report_date,
        "status": item.status,
        "created_at": item.created_at,
        "owner_id": item.owner_id,
    }


def _category_counts(reports: list[Any]) -> list[CategoryCount]:
    counts: dict[str, dict[str, int]] = {}
    for item in reports:
        category = str(_get(item, "category") or "Uncategorized")
        item_type = str(_get(item, "type") or "")
        counts.setdefault(category, {"lost": 0, "found": 0})
        if item_type in {"lost", "found"}:
            counts[category][item_type] += 1
    return [
        CategoryCount(category=category, **totals)
        for category, totals in sorted(counts.items(), key=lambda pair: pair[0].casefold())
    ]


def _date_gap(left: Any, right: Any) -> int | None:
    def parse(value: Any):
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    a = parse(_get(left, "report_date") or _get(left, "reportDate"))
    b = parse(_get(right, "report_date") or _get(right, "reportDate"))
    return abs((b - a).days) if a and b else None


def _recommendation(lost: Any, candidate: Any, score_data: dict[str, Any]) -> dict[str, Any]:
    score = float(score_data["score"])
    matched: list[str] = []
    different: list[str] = []
    for part in score_data["breakdown"]:
        if not part["available"]:
            continue
        if float(part["similarity"]) >= 0.45:
            matched.append(part["label"])
        else:
            different.append(part["label"])
    top_reasons = [
        f"{part['label'].lower()} {float(part['similarity']) * 100:.0f}%"
        for part in sorted(
            (p for p in score_data["breakdown"] if p["available"]),
            key=lambda p: (-float(p["contribution"]), p["attribute"]),
        )[:3]
    ]
    explanation = (
        f"{score:.1f}% heuristic similarity across "
        f"{score_data['evidence_coverage']:.0f}% of the configured attribute weight."
    )
    if top_reasons:
        explanation += " Strongest evidence: " + ", ".join(top_reasons) + "."
    return {
        "item": ItemOut.model_validate(_item_json(candidate)),
        "score": score,
        "evidence_coverage": score_data["evidence_coverage"],
        "grade": grade_for(score),
        "matching_attributes": matched,
        "differing_attributes": different,
        "explanation": explanation,
        "breakdown": score_data["breakdown"],
        "jaccard": score_data["jaccard"],
    }


def run_once(
    lost_item: Any,
    reports: list[Any],
    request: MatchRequest,
    algorithm: str,
    dataset_name: str,
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[str, Any]:
    all_found = [item for item in reports if _get(item, "type") == "found"]
    if _get(lost_item, "type") != "lost" or _get(lost_item, "status") != "active":
        raise ValueError("Choose an active lost report before running matching.")

    active_found = [item for item in all_found if _get(item, "status") == "active"]
    date_tolerance = request.date_tolerance_days
    decisions = [
        DecisionStep(
            step="Select lost report",
            status="passed",
            detail=f"Active lost report #{_get(lost_item, 'id')} selected.",
            count=1,
        ),
        DecisionStep(
            step="Validate report",
            status="passed",
            detail="The selected report has type lost and status active.",
            count=1,
        ),
        DecisionStep(
            step="Load found candidates",
            status="passed",
            detail=f"Read {len(all_found)} found reports; {len(active_found)} are active.",
            count=len(all_found),
        ),
    ]

    evaluated: list[dict[str, Any]] = []
    date_rejected = 0
    invalid_rejected = 0
    category_rejected = 0
    location_date_rejected = 0

    candidates = all_found if algorithm == "brute_force" else active_found
    for candidate in candidates:
        valid, reasons = candidate_validation(lost_item, candidate, date_tolerance)
        if not valid:
            if any("date" in reason.casefold() for reason in reasons):
                date_rejected += 1
            else:
                invalid_rejected += 1
            continue

        if algorithm == "heuristic" and request.strict_category_filter:
            category_similarity = text_similarity(
                str(_get(lost_item, "category") or ""),
                str(_get(candidate, "category") or ""),
            )
            if category_similarity is None or category_similarity < 0.35:
                category_rejected += 1
                continue

        if algorithm == "heuristic" and request.location_date_filter:
            location_similarity = text_similarity(
                str(_get(lost_item, "location") or ""),
                str(_get(candidate, "location") or ""),
            )
            gap = _date_gap(lost_item, candidate)
            if (location_similarity or 0) < 0.15 and (gap is None or gap > 7):
                location_date_rejected += 1
                continue

        score_data = score_pair(lost_item, candidate, request.weights)
        evaluated.append(_recommendation(lost_item, candidate, score_data))

    ranked_all = rank_top_k(evaluated, len(evaluated))
    recommendations = rank_top_k(
        (
            result
            for result in evaluated
            if float(result["score"]) >= request.threshold
        ),
        request.top_k,
    )
    scores = [float(result["score"]) for result in evaluated]
    highest = maximum_score(evaluated)
    found_count = len(all_found)
    reduction = None

    decisions.extend(
        [
            DecisionStep(
                step="Filter categories",
                status=(
                    "passed"
                    if algorithm == "heuristic" and request.strict_category_filter
                    else "skipped"
                ),
                detail=(
                    f"Removed {category_rejected} candidates with incompatible categories."
                    if algorithm == "heuristic" and request.strict_category_filter
                    else "No optional category prefilter was applied."
                ),
                count=category_rejected,
            ),
            DecisionStep(
                step="Check mandatory conditions",
                status="passed",
                detail=(
                    f"Excluded {invalid_rejected} inactive/type-invalid reports and "
                    f"{date_rejected} date-incompatible reports."
                ),
                count=len(candidates) - invalid_rejected - date_rejected,
            ),
            DecisionStep(
                step="Optional location/date filter",
                status=(
                    "passed"
                    if algorithm == "heuristic" and request.location_date_filter
                    else "skipped"
                ),
                detail=(
                    f"Removed {location_date_rejected} candidates outside the location/date window."
                    if algorithm == "heuristic" and request.location_date_filter
                    else "Optional location/date prefilter is off."
                ),
                count=location_date_rejected,
            ),
            DecisionStep(
                step="Calculate weighted similarity",
                status="completed",
                detail=f"Calculated scores for {len(evaluated)} eligible candidates.",
                count=len(evaluated),
            ),
            DecisionStep(
                step="Rank and return Top-K",
                status="completed",
                detail=(
                    f"Returned {len(recommendations)} results at or above "
                    f"the {request.threshold:.0f}% threshold, capped at K={request.top_k}."
                ),
                count=len(recommendations),
            ),
        ]
    )

    report_ids = {int(_get(item, "id")) for item in reports}
    invariant_checks = [
        InvariantCheck(
            invariant="Lost reports match only found reports",
            passed=_get(lost_item, "type") == "lost"
            and all(_get(item, "type") == "found" for item in all_found),
            detail="The selected report is lost and the candidate pool contains only found reports.",
        ),
        InvariantCheck(
            invariant="Resolved reports are excluded",
            passed=all(_get(result["item"], "status") == "active" for result in evaluated),
            detail="Only active candidates reached the scoring step.",
        ),
        InvariantCheck(
            invariant="A report cannot match itself",
            passed=all(int(result["item"].id) != int(_get(lost_item, "id")) for result in evaluated),
            detail="Every evaluated report has a distinct ID from the selected lost report.",
        ),
        InvariantCheck(
            invariant="Similarity scores stay in [0, 100]",
            passed=all(0 <= float(result["score"]) <= 100 for result in evaluated),
            detail="All calculated weighted scores were bounded to the valid interval.",
        ),
        InvariantCheck(
            invariant="Candidate IDs belong to the selected dataset",
            passed=all(int(result["item"].id) in report_ids for result in evaluated),
            detail="Every recommendation comes from the selected report set.",
        ),
    ]
    if algorithm == "heuristic" and request.strict_category_filter:
        invariant_checks.append(
            InvariantCheck(
                invariant="Category prefilter is explicit",
                passed=True,
                detail=(
                    "Strict category filtering is enabled and can exclude mislabeled "
                    "matches; disable it to broaden the search."
                ),
            )
        )

    truth = (ground_truth or {}).get(int(_get(lost_item, "id")))
    precision: float | None = None
    recall: float | None = None
    labeled_count = 0
    if truth is not None:
        labeled_count = len(truth)
        predicted = {int(result["item"].id) for result in recommendations}
        true_positives = len(predicted & truth)
        if predicted:
            precision = round(true_positives / len(predicted), 4)
        if truth:
            recall = round(true_positives / len(truth), 4)

    result = AlgorithmRun(
        algorithm=algorithm,
        total_candidates=found_count,
        evaluated_candidates=len(evaluated),
        similarity_calculations=len(evaluated),
        execution_time_ms=0.0,
        highest_score=round(highest, 2) if highest is not None else None,
        average_score=round(statistics.fmean(scores), 2) if scores else None,
        candidate_reduction_percentage=round(reduction, 2) if reduction is not None else None,
        recommendations=recommendations,
        decision_tree=decisions,
        invariant_checks=invariant_checks,
        category_distribution=_category_counts(reports),
        dataset_name=dataset_name,
        precision=precision,
        recall=recall,
        labeled_pair_count=labeled_count,
    )
    return result.model_dump(by_alias=False)


def _benchmark(
    lost_item: Any,
    reports: list[Any],
    request: MatchRequest,
    algorithm: str,
    dataset_name: str,
    ground_truth: dict[int, set[int]] | None,
) -> dict[str, Any]:
    # Warm-up is discarded. Timings use real perf_counter measurements; output
    # reflects the median of five completed runs over the same immutable data.
    run_once(lost_item, reports, request, algorithm, dataset_name, ground_truth)
    durations: list[float] = []
    result: dict[str, Any] | None = None
    for _ in range(5):
        started = time.perf_counter()
        result = run_once(
            lost_item, reports, request, algorithm, dataset_name, ground_truth
        )
        durations.append((time.perf_counter() - started) * 1000)
    assert result is not None
    result["execution_time_ms"] = round(statistics.median(durations), 4)
    return result


def run_benchmarked(
    lost_item: Any,
    reports: list[Any],
    request: MatchRequest,
    algorithm: str,
    dataset_name: str,
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[str, Any]:
    return _benchmark(
        lost_item, reports, request, algorithm, dataset_name, ground_truth
    )


def compare_benchmarked(
    lost_item: Any,
    reports: list[Any],
    request: MatchRequest,
    dataset_name: str,
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[str, Any]:
    brute = _benchmark(
        lost_item, reports, request, "brute_force", dataset_name, ground_truth
    )
    heuristic = _benchmark(
        lost_item, reports, request, "heuristic", dataset_name, ground_truth
    )
    baseline = brute["similarity_calculations"]
    heuristic_count = heuristic["similarity_calculations"]
    reduction = ((baseline - heuristic_count) / baseline * 100) if baseline else None
    brute["candidate_reduction_percentage"] = 0.0 if baseline else None
    heuristic["candidate_reduction_percentage"] = (
        round(reduction, 2) if reduction is not None else None
    )
    return CompareResponse(
        brute_force=AlgorithmRun.model_validate(brute),
        heuristic=AlgorithmRun.model_validate(heuristic),
        candidate_reduction_percentage=round(reduction, 2) if reduction is not None else None,
        ground_truth_available=ground_truth is not None and bool(ground_truth),
        methodology=(
            "Each algorithm ran once as a warm-up and five measured times on identical "
            "data. Execution time is the median perf_counter duration; similarity is "
            "a heuristic, not a probability of ownership. Filtering can miss a mislabeled match."
        ),
    ).model_dump(by_alias=False)

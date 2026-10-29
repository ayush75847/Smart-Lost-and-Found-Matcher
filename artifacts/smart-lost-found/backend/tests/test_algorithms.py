from __future__ import annotations

from datetime import date

import pytest

from algorithms.ai_rules import analyze_description
from algorithms.datasets import get_dataset
from algorithms.heuristic import normalize_weights, score_pair
from algorithms.invariants import candidate_validation, is_valid_candidate
from algorithms.matching import compare_benchmarked, run_benchmarked, run_once
from algorithms.pattern_recognition import normalize_text, text_similarity, tokenize
from algorithms.ranking import maximum_score, rank_top_k
from algorithms.set_similarity import jaccard_details, jaccard_similarity
from schemas import MatchInput


def make_item(
    item_id=1,
    item_type="lost",
    *,
    name="Wallet",
    category="Accessories",
    brand="Wildcraft",
    color="Black",
    description="Black leather wallet with a zipper.",
    location="College Library",
    report_date=date(2026, 10, 25),
    status="active",
):
    return {
        "id": item_id,
        "type": item_type,
        "name": name,
        "category": category,
        "brand": brand,
        "color": color,
        "description": description,
        "location": location,
        "report_date": report_date,
        "status": status,
        "created_at": "2026-10-29T12:00:00",
        "owner_id": None,
    }


def test_text_normalization_is_case_and_whitespace_insensitive():
    assert normalize_text("  BLACK   Leather WALLET! ") == "black leather wallet"
    assert text_similarity("BLACK wallet", "dark WALLET") == 1.0


def test_synonyms_and_fuzzy_matching_recognize_patterns():
    assert tokenize("navy backpack") == {"blue", "backpack"}
    assert text_similarity("wallet", "walet") > 0.7
    assert text_similarity("wallet", "steel bottle") < 0.3


def test_jaccard_formula_uses_sets_and_returns_intersection_union():
    details = jaccard_details(
        {"black", "leather", "wallet", "zipper"},
        {"black", "leather", "wallet", "small"},
    )
    assert details["intersection"] == ["black", "leather", "wallet"]
    assert details["union"] == ["black", "leather", "small", "wallet", "zipper"]
    assert jaccard_similarity(
        {"black", "leather", "wallet", "zipper"},
        {"black", "leather", "wallet", "small"},
    ) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (set(), set(), None),
        (set(), {"wallet"}, 0.0),
        ({"wallet"}, {"wallet"}, 1.0),
    ],
)
def test_jaccard_handles_empty_sets(left, right, expected):
    assert jaccard_similarity(left, right) == expected


def test_jaccard_removes_stopwords_and_normalizes_text():
    details = jaccard_details("the wallet near the library", "wallet at library")
    assert details["intersection"] == ["library", "wallet"]
    assert details["similarity"] == 1.0


def test_weighted_heuristic_is_100_for_identical_complete_reports():
    lost = make_item()
    found = make_item(2, "found")
    result = score_pair(lost, found)
    assert result["score"] == 100
    assert result["evidence_coverage"] == 100
    assert sum(part["contribution"] for part in result["breakdown"]) == pytest.approx(100)


def test_weight_normalization_scales_percentages():
    weights = normalize_weights(
        {
            "category": 50,
            "name": 40,
            "color": 30,
            "brand": 20,
            "location": 30,
            "date": 10,
            "description": 20,
        }
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["category"] == pytest.approx(0.25)


def test_missing_optional_attributes_are_excluded_and_reduce_coverage():
    lost = make_item(brand="Wildcraft", color="Black")
    found = make_item(2, "found", brand=None, color=None)
    result = score_pair(lost, found)
    assert result["score"] == pytest.approx(100)
    assert result["evidence_coverage"] < 100
    assert next(part for part in result["breakdown"] if part["attribute"] == "brand")[
        "available"
    ] is False


def test_empty_comparison_fields_have_zero_evidence_not_a_perfect_score():
    result = score_pair(
        make_item(name="", category="", brand=None, color=None, description="", location="", report_date=None),
        make_item(2, "found", name="", category="", brand=None, color=None, description="", location="", report_date=None),
    )
    assert result["score"] == 0
    assert result["evidence_coverage"] == 0


def test_candidate_conditions_check_type_status_identity_and_dates():
    lost = make_item()
    valid_found = make_item(2, "found", report_date=date(2026, 10, 26))
    assert is_valid_candidate(lost, valid_found, 1)
    assert candidate_validation(lost, make_item(1, "found"))[0] is False
    assert candidate_validation(lost, make_item(2, "lost"))[0] is False
    assert candidate_validation(lost, make_item(2, "found", status="resolved"))[0] is False


def test_impossible_date_is_rejected_outside_configured_tolerance():
    lost = make_item(report_date=date(2026, 10, 25))
    found = make_item(2, "found", report_date=date(2026, 10, 20))
    valid, reasons = candidate_validation(lost, found, date_tolerance_days=1)
    assert not valid
    assert "Found date is 5 days before the lost date" in reasons[0]
    assert candidate_validation(lost, found, date_tolerance_days=5)[0]


def test_top_k_uses_score_then_report_id_for_ties():
    options = [
        {"score": 89, "item": {"id": 4}},
        {"score": 96, "item": {"id": 3}},
        {"score": 89, "item": {"id": 2}},
    ]
    assert [row["item"]["id"] for row in rank_top_k(options, 3)] == [3, 2, 4]
    assert maximum_score(options) == 96


def test_top_k_supports_pydantic_item_results_and_empty_lists():
    dataset = get_dataset("exact")
    request = MatchInput(
        dataset_id="exact",
        top_k=1,
        threshold=0,
        strict_category_filter=False,
    )
    result = run_once(
        dataset.reports[0], dataset.reports, request, "brute_force", dataset.info.name,
        dataset.ground_truth,
    )
    assert len(result["recommendations"]) == 1
    assert rank_top_k([], 5) == []
    assert maximum_score([]) is None


def test_brute_force_compares_active_candidates_without_optional_prefilters():
    dataset = get_dataset("filtering-tradeoff")
    request = MatchInput(
        dataset_id=dataset.info.id,
        top_k=5,
        threshold=40,
        strict_category_filter=True,
    )
    result = run_once(
        dataset.reports[0], dataset.reports, request, "brute_force", dataset.info.name,
        dataset.ground_truth,
    )
    assert result["evaluated_candidates"] == 2
    assert result["recommendations"][0]["item"]["id"] == 2


def test_heuristic_category_filter_reduces_work_but_can_miss_mislabeled_match():
    dataset = get_dataset("filtering-tradeoff")
    request = MatchInput(
        dataset_id=dataset.info.id,
        top_k=5,
        threshold=40,
        strict_category_filter=True,
    )
    result = run_once(
        dataset.reports[0], dataset.reports, request, "heuristic", dataset.info.name,
        dataset.ground_truth,
    )
    assert result["evaluated_candidates"] == 1
    assert all(item["item"]["id"] != 2 for item in result["recommendations"])
    assert "can exclude mislabeled matches" in result["invariant_checks"][-1]["detail"]


def test_resolved_candidates_and_temporal_inconsistencies_are_not_scored():
    dataset = get_dataset("invalid-conflicting")
    request = MatchInput(
        dataset_id=dataset.info.id,
        top_k=5,
        threshold=0,
        strict_category_filter=False,
    )
    result = run_once(
        dataset.reports[0], dataset.reports, request, "brute_force", dataset.info.name
    )
    assert result["evaluated_candidates"] == 2
    assert all(item["item"]["status"] == "active" for item in result["recommendations"])


def test_empty_candidate_set_returns_no_recommendations_and_safe_metrics():
    dataset = get_dataset("no-matches")
    empty = dataset.reports[:1]
    request = MatchInput(dataset_id="no-matches", top_k=5, threshold=40)
    result = run_once(dataset.reports[0], empty, request, "brute_force", "empty")
    assert result["total_candidates"] == 0
    assert result["recommendations"] == []
    assert result["highest_score"] is None
    assert result["average_score"] is None


def test_compare_metrics_are_measured_and_truth_metrics_are_labelled():
    dataset = get_dataset("exact")
    request = MatchInput(
        dataset_id="exact",
        top_k=5,
        threshold=40,
        strict_category_filter=False,
    )
    response = compare_benchmarked(
        dataset.reports[0], dataset.reports, request, dataset.info.name, dataset.ground_truth
    )
    assert response["brute_force"]["execution_time_ms"] >= 0
    assert response["heuristic"]["execution_time_ms"] >= 0
    assert response["brute_force"]["precision"] == 1
    assert response["brute_force"]["recall"] == 1
    assert "warm-up" in response["methodology"]


@pytest.mark.parametrize(
    "description,expected",
    [
        (
            "Lost my dark-colored leather wallet near the college library.",
            {
                "item_name": "wallet",
                "category": "Accessories",
                "color": "Dark",
                "material": "Leather",
                "location": "college library",
            },
        ),
        ("Something is missing.", {"item_name": None, "category": None}),
    ],
)
def test_rule_based_description_extraction_does_not_invent_attributes(description, expected):
    result = analyze_description(description)
    for field, value in expected.items():
        assert result["attributes"][field] == value
    assert result["source"] == "deterministic_rules"


@pytest.mark.parametrize("count", [100, 500, 1000])
def test_synthetic_datasets_are_reproducible_and_have_ground_truth(count):
    first = get_dataset(f"synthetic-{count}-seed42")
    second = get_dataset(f"synthetic-{count}-seed42")
    assert len(first.reports) == count
    assert first.reports[0] == second.reports[0]
    assert first.ground_truth == {1: {2}}


def test_all_demo_datasets_have_valid_counts_and_isolated_records():
    for info in __import__("algorithms.datasets", fromlist=["dataset_catalog"]).dataset_catalog():
        dataset = get_dataset(info.id)
        assert len(dataset.reports) == info.report_count
        assert info.lost_count + info.found_count == info.report_count


def test_invalid_custom_weights_are_rejected():
    with pytest.raises(ValueError):
        normalize_weights(
            {
                "category": 0,
                "name": 0,
                "color": 0,
                "brand": 0,
                "location": 0,
                "date": 0,
                "description": 0,
            }
        )

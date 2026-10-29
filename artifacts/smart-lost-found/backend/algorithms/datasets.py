from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from schemas import DatasetInfo


@dataclass
class ExperimentDataset:
    info: DatasetInfo
    reports: list[dict[str, Any]]
    ground_truth: dict[int, set[int]]


def _item(
    item_id: int,
    item_type: str,
    *,
    name: str = "Leather wallet",
    category: str = "Accessories",
    brand: str | None = "Wildcraft",
    color: str | None = "Black",
    description: str = "Black leather wallet with a small zipper.",
    location: str = "College Library",
    report_date: date = date(2026, 10, 25),
    status: str = "active",
) -> dict[str, Any]:
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
        "created_at": datetime(2026, 10, 29, 12, 0, 0),
        "owner_id": None,
    }


def _base_dataset(dataset_id: str) -> ExperimentDataset:
    lost = _item(1, "lost")
    exact_found = _item(
        2,
        "found",
        name="Leather wallet",
        description="Found a dark leather wallet with a small zipper.",
        location="Outside the college library",
        report_date=date(2026, 10, 26),
    )
    distractor = _item(
        3,
        "found",
        name="Blue water bottle",
        category="Drinkware",
        brand="HydroPeak",
        color="Blue",
        description="Tall steel bottle with a flip lid.",
        location="Science Building",
        report_date=date(2026, 10, 27),
    )

    if dataset_id == "exact":
        reports = [lost, exact_found, distractor]
        return ExperimentDataset(
            DatasetInfo(
                id="exact",
                name="Exact matches",
                description="One near-exact labeled match plus a distractor.",
                report_count=len(reports),
                lost_count=1,
                found_count=2,
                has_ground_truth=True,
                is_synthetic=False,
            ),
            reports,
            {1: {2}},
        )
    if dataset_id == "partial":
        partial_found = _item(
            2,
            "found",
            name="Small wallet",
            category="Accessories",
            brand=None,
            color="Brown",
            description="Compact leather wallet found by the campus library.",
            location="Library entrance",
            report_date=date(2026, 10, 27),
        )
        reports = [lost, partial_found, distractor]
        return ExperimentDataset(
            DatasetInfo(
                id="partial",
                name="Partial matches",
                description="A plausible candidate shares some, but not all, attributes.",
                report_count=len(reports),
                lost_count=1,
                found_count=2,
                has_ground_truth=True,
                is_synthetic=False,
            ),
            reports,
            {1: {2}},
        )
    if dataset_id == "missing-information":
        missing = _item(
            2,
            "found",
            name="Leather wallet",
            brand=None,
            color=None,
            description="Wallet found near the library.",
            location="College Library",
            report_date=date(2026, 10, 26),
        )
        reports = [lost, missing, distractor]
        return ExperimentDataset(
            DatasetInfo(
                id="missing-information",
                name="Missing information",
                description="Optional brand and color evidence are deliberately absent.",
                report_count=len(reports),
                lost_count=1,
                found_count=2,
                has_ground_truth=True,
                is_synthetic=False,
            ),
            reports,
            {1: {2}},
        )
    if dataset_id == "no-matches":
        found_a = _item(
            2, "found", name="Tent", category="Camping", brand="Trail", color="Green",
            description="Large green tent.", location="Sports Field",
        )
        found_b = _item(
            3, "found", name="Running shoes", category="Footwear", brand="Pace",
            color="White", description="Pair of running shoes.", location="Gym",
        )
        reports = [lost, found_a, found_b]
        return ExperimentDataset(
            DatasetInfo(
                id="no-matches",
                name="No matches",
                description="Found items have unrelated categories and attributes.",
                report_count=len(reports),
                lost_count=1,
                found_count=2,
                has_ground_truth=True,
                is_synthetic=False,
            ),
            reports,
            {1: set()},
        )
    if dataset_id == "invalid-conflicting":
        impossible = _item(
            2, "found", report_date=date(2026, 10, 20), location="College Library"
        )
        resolved = _item(3, "found", status="resolved")
        equal_score = _item(
            4, "found", name="Leather case", category="Accessories", brand=None,
            color=None, description="Leather item.", location="College Library",
        )
        equal_score_tie = _item(
            5, "found", name="Leather case", category="Accessories", brand=None,
            color=None, description="Leather item.", location="College Library",
        )
        reports = [lost, impossible, resolved, equal_score, equal_score_tie]
        return ExperimentDataset(
            DatasetInfo(
                id="invalid-conflicting",
                name="Invalid & conflicting",
                description="Includes impossible dates, resolved reports, and tied scores.",
                report_count=len(reports),
                lost_count=1,
                found_count=4,
                has_ground_truth=False,
                is_synthetic=False,
            ),
            reports,
            {},
        )
    if dataset_id == "filtering-tradeoff":
        mislabeled = _item(
            2,
            "found",
            name="Leather wallet",
            category="Electronics",
            brand="Wildcraft",
            color="Black",
            description="Black leather wallet with a small zipper.",
            location="College Library",
            report_date=date(2026, 10, 26),
        )
        category_matched_distractor = _item(
            3,
            "found",
            name="Blue water bottle",
            category="Accessories",
            brand="HydroPeak",
            color="Blue",
            description="Tall steel bottle with a flip lid.",
            location="Science Building",
            report_date=date(2026, 10, 27),
        )
        reports = [lost, mislabeled, category_matched_distractor]
        return ExperimentDataset(
            DatasetInfo(
                id="filtering-tradeoff",
                name="Filtering trade-off",
                description="A relevant match has the wrong category label by design.",
                report_count=len(reports),
                lost_count=1,
                found_count=2,
                has_ground_truth=True,
                is_synthetic=False,
            ),
            reports,
            {1: {2}},
        )
    raise KeyError(f"Unknown dataset: {dataset_id}")


def _generate_dataset(report_count: int, seed: int) -> ExperimentDataset:
    if report_count not in {100, 500, 1000}:
        raise ValueError("Synthetic datasets support 100, 500, or 1,000 reports.")
    rng = random.Random(seed)
    info_id = f"synthetic-{report_count}-seed{seed}"
    lost = _item(
        1,
        "lost",
        name="Wireless headphones",
        category="Electronics",
        brand="SoundWay",
        color="Navy",
        description="Navy wireless headphones with a silver mark on the case.",
        location="Library study room",
    )
    true_match = _item(
        2,
        "found",
        name="Wireless headphone case",
        category="Electronics",
        brand="SoundWay",
        color="Blue",
        description="Found navy headphones and a case with a silver mark.",
        location="Library study area",
        report_date=date(2026, 10, 26),
    )
    reports = [lost, true_match]
    names = ["Wallet", "Phone", "Backpack", "Water bottle", "Headphones", "Notebook"]
    categories = ["Accessories", "Electronics", "Bags", "Drinkware", "Stationery"]
    colors = ["Black", "Blue", "Red", "Green", "Gray", "Brown"]
    brands = ["Northstar", "SoundWay", "Wildcraft", "Campus", "Unbranded"]
    locations = [
        "College Library", "Science Building", "Cafeteria", "Sports Field",
        "Main Gate", "Computer Lab",
    ]
    date_offsets = list(range(-7, 10))
    lost_target_count = max(1, int(report_count * 0.30))
    for item_id in range(3, report_count + 1):
        is_lost = item_id < 2 + lost_target_count
        item_type = "lost" if is_lost else "found"
        name = rng.choice(names)
        category = rng.choice(categories)
        color = rng.choice(colors)
        brand = rng.choice(brands)
        location = rng.choice(locations)
        offset = rng.choice(date_offsets)
        report_date = date(2026, 10, 25) + timedelta(days=offset)
        reports.append(
            _item(
                item_id,
                item_type,
                name=name,
                category=category,
                brand=brand if rng.random() > 0.15 else None,
                color=color if rng.random() > 0.15 else None,
                description=f"{color} {name.lower()} reported near {location.lower()}.",
                location=location,
                report_date=report_date,
            )
        )

    lost_count = sum(item["type"] == "lost" for item in reports)
    found_count = report_count - lost_count
    return ExperimentDataset(
        DatasetInfo(
            id=info_id,
            name=f"Synthetic {report_count:,} reports",
            description=f"Reproducibly generated with seed {seed}; includes one labeled true match.",
            report_count=report_count,
            lost_count=lost_count,
            found_count=found_count,
            has_ground_truth=True,
            is_synthetic=True,
        ),
        reports,
        {1: {2}},
    )


def dataset_catalog() -> list[DatasetInfo]:
    names = [
        "exact",
        "partial",
        "missing-information",
        "no-matches",
        "invalid-conflicting",
        "filtering-tradeoff",
    ]
    catalog = [_base_dataset(dataset_id).info for dataset_id in names]
    catalog.extend(_generate_dataset(count, 42).info for count in (100, 500, 1000))
    return catalog


def get_dataset(
    dataset_id: str,
    report_count: int | None = None,
    seed: int = 42,
) -> ExperimentDataset:
    if dataset_id in {
        "exact",
        "partial",
        "missing-information",
        "no-matches",
        "invalid-conflicting",
        "filtering-tradeoff",
    }:
        return _base_dataset(dataset_id)
    match = re.fullmatch(r"synthetic-(100|500|1000)-seed(\d+)", dataset_id)
    if match:
        return _generate_dataset(int(match.group(1)), int(match.group(2)))
    if report_count in {100, 500, 1000}:
        return _generate_dataset(report_count, seed)
    raise KeyError(f"Unknown dataset: {dataset_id}")

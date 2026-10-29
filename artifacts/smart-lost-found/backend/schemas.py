from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class APIModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        extra="forbid",
    )


class HealthStatus(APIModel):
    status: str


class Message(APIModel):
    message: str


class UserOut(APIModel):
    id: int
    username: str
    role: Literal["user", "admin"]


class RegistrationInput(APIModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=10, max_length=128)


class LoginInput(APIModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class SessionResponse(APIModel):
    user: UserOut


class ItemInput(APIModel):
    name: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=80)
    brand: str | None = Field(default=None, max_length=120)
    color: str | None = Field(default=None, max_length=80)
    description: str = Field(min_length=1, max_length=2000)
    location: str = Field(min_length=1, max_length=200)
    report_date: date

    @field_validator("name", "category", "description", "location")
    @classmethod
    def trim_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("This field cannot be blank.")
        return cleaned


class LostItemInput(ItemInput):
    identifying_features: str | None = Field(default=None, max_length=1000)


class FoundItemInput(ItemInput):
    private_identifying_features: str | None = Field(
        default=None, max_length=1000
    )


class ItemOut(APIModel):
    id: int
    type: Literal["lost", "found"]
    name: str
    category: str
    brand: str | None
    color: str | None
    description: str
    location: str
    report_date: date
    status: Literal["active", "resolved"]
    created_at: datetime
    owner_id: int | None


class ItemPage(APIModel):
    items: list[ItemOut]
    total: int
    page: int
    page_size: int
    page_count: int
    categories: list[str]


class CategoryCount(APIModel):
    category: str
    lost: int
    found: int


class DashboardStats(APIModel):
    total_lost: int
    total_found: int
    active_reports: int
    potential_matches: int
    resolved_reports: int
    categories: list[CategoryCount]
    recent_reports: list[ItemOut]


DEFAULT_WEIGHTS = {
    "category": 0.25,
    "name": 0.20,
    "color": 0.15,
    "brand": 0.10,
    "location": 0.15,
    "date": 0.05,
    "description": 0.10,
}


class MatchInput(APIModel):
    lost_item_id: int | None = None
    dataset_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=40, ge=0, le=100)
    date_tolerance_days: int = Field(default=1, ge=0, le=30)
    weights: dict[str, float] | None = None
    strict_category_filter: bool = True
    location_date_filter: bool = False

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float] | None):
        if value is None:
            return value
        if any(not key for key in value) or any(weight < 0 for weight in value.values()):
            raise ValueError("Weights must be non-negative.")
        if value and sum(value.values()) <= 0:
            raise ValueError("At least one weight must be greater than zero.")
        return value


MatchRequest = MatchInput


class ExperimentInput(MatchInput):
    dataset_id: str
    report_count: int | None = Field(default=None, ge=100, le=1000)
    seed: int = Field(default=42, ge=0, le=2147483647)


class ScorePart(APIModel):
    attribute: str
    label: str
    weight: float
    similarity: float | None
    contribution: float | None
    available: bool
    explanation: str


class SetComparison(APIModel):
    set_a: list[str]
    set_b: list[str]
    intersection: list[str]
    union: list[str]
    similarity: float | None


class MatchRecommendation(APIModel):
    item: ItemOut
    score: float = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=100)
    grade: str
    matching_attributes: list[str]
    differing_attributes: list[str]
    explanation: str
    breakdown: list[ScorePart]
    jaccard: SetComparison


class DecisionStep(APIModel):
    step: str
    status: Literal["passed", "filtered", "skipped", "completed"]
    detail: str
    count: int | None = None


class InvariantCheck(APIModel):
    invariant: str
    passed: bool
    detail: str


class AlgorithmRun(APIModel):
    algorithm: Literal["brute_force", "heuristic"]
    total_candidates: int
    evaluated_candidates: int
    similarity_calculations: int
    execution_time_ms: float
    highest_score: float | None
    average_score: float | None
    candidate_reduction_percentage: float | None
    recommendations: list[MatchRecommendation]
    decision_tree: list[DecisionStep]
    invariant_checks: list[InvariantCheck]
    category_distribution: list[CategoryCount]
    dataset_name: str
    precision: float | None = None
    recall: float | None = None
    labeled_pair_count: int = 0


class CompareResponse(APIModel):
    brute_force: AlgorithmRun
    heuristic: AlgorithmRun
    candidate_reduction_percentage: float | None
    ground_truth_available: bool
    methodology: str


class AnalyzeInput(APIModel):
    description: str = Field(min_length=1, max_length=2000)


class ExtractedAttributes(APIModel):
    item_name: str | None = None
    category: str | None = None
    color: str | None = None
    material: str | None = None
    brand: str | None = None
    location: str | None = None


class AnalyzeResult(APIModel):
    source: Literal["deterministic_rules"] = "deterministic_rules"
    attributes: ExtractedAttributes
    confidence_notes: list[str]


class ClaimInput(APIModel):
    lost_item_id: int
    found_item_id: int
    ownership_evidence: str = Field(min_length=10, max_length=2000)

    @field_validator("ownership_evidence")
    @classmethod
    def trim_evidence(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Please describe identifying evidence.")
        return cleaned


class ClaimReviewInput(APIModel):
    decision: Literal["approved", "rejected"]
    review_note: str | None = Field(default=None, max_length=1000)


class ClaimOut(APIModel):
    id: int
    lost_item: ItemOut
    found_item: ItemOut
    claimant: UserOut
    ownership_evidence: str
    status: Literal["pending", "approved", "rejected"]
    review_note: str | None
    private_feature_for_review: str | None = None
    created_at: datetime
    reviewed_at: datetime | None


class DatasetInfo(APIModel):
    id: str
    name: str
    description: str
    report_count: int
    lost_count: int
    found_count: int
    has_ground_truth: bool
    is_synthetic: bool


class GenerateDatasetInput(APIModel):
    report_count: Literal[100, 500, 1000]
    seed: int = Field(default=42, ge=0, le=2147483647)

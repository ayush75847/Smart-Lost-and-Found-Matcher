from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from sqlalchemy import func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from algorithms.ai_rules import analyze_description
from algorithms.datasets import dataset_catalog, get_dataset
from algorithms.invariants import assert_approved_claim_is_valid, candidate_validation
from algorithms.matching import compare_benchmarked, run_benchmarked, run_once
from database import Base, SessionLocal, engine, get_db
from models import (
    Claim,
    ClaimVerificationHistory,
    Item,
    LoginSession,
    User,
)
from schemas import (
    AnalyzeInput,
    AnalyzeResult,
    AlgorithmRun,
    ClaimInput,
    ClaimOut,
    ClaimReviewInput,
    CompareResponse,
    DashboardStats,
    DatasetInfo,
    ExperimentInput,
    FoundItemInput,
    GenerateDatasetInput,
    HealthStatus,
    ItemOut,
    ItemPage,
    LoginInput,
    LostItemInput,
    MatchInput,
    Message,
    RegistrationInput,
    SessionResponse,
    UserOut,
)
from security import find_session_user, hash_password, issue_session, revoke_session, verify_password

@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    _prepare_database()
    yield


app = FastAPI(
    title="Smart Lost & Found Matcher API",
    description=(
        "Python-based, explainable lost-and-found matching with SQLite persistence, "
        "claim verification, and isolated algorithm experiments."
    ),
    version="1.0.0",
    lifespan=app_lifespan,
)
API_PREFIX = "/api"
SESSION_COOKIE = "session"
COOKIE_MAX_AGE = 7 * 24 * 60 * 60


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, username=user.username, role=user.role)


def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    secure_cookie = (
        request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
        or request.url.scheme == "https"
    )
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        path=API_PREFIX,
    )


def current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User | None:
    return find_session_user(db, request.cookies.get(SESSION_COOKIE))


def require_user(user: Annotated[User | None, Depends(current_user)]) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


def require_admin(user: Annotated[User, Depends(require_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    return user


def _public_item(item: Item | dict) -> ItemOut:
    if isinstance(item, dict):
        return ItemOut.model_validate(item)
    return ItemOut.model_validate(item)


def _claim_out(claim: Claim, show_private: bool = False) -> ClaimOut:
    return ClaimOut(
        id=claim.id,
        lost_item=_public_item(claim.lost_item),
        found_item=_public_item(claim.found_item),
        claimant=_user_out(claim.claimant),
        ownership_evidence=claim.ownership_evidence,
        status=claim.status,
        review_note=claim.review_note,
        private_feature_for_review=(
            claim.found_item.identifying_features if show_private else None
        ),
        created_at=claim.created_at,
        reviewed_at=claim.reviewed_at,
    )


def _prepare_database() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        if admin is None:
            admin = User(
                username="admin",
                password_hash=hash_password("AdminPass123!"),
                role="admin",
            )
            db.add(admin)
        student = db.scalar(select(User).where(User.username == "student"))
        if student is None:
            student = User(
                username="student",
                password_hash=hash_password("StudentPass123!"),
                role="user",
            )
            db.add(student)
        db.commit()
        db.refresh(student)

        if db.scalar(select(func.count(Item.id))) == 0:
            lost_date = date.today() - timedelta(days=4)
            examples = [
                Item(
                    type="lost",
                    name="Leather wallet",
                    category="Accessories",
                    brand="Wildcraft",
                    color="Black",
                    description="Black leather wallet with a small zipper.",
                    location="College Library",
                    report_date=lost_date,
                    identifying_features="Small initials stamped inside.",
                    status="active",
                    owner_id=student.id,
                ),
                Item(
                    type="found",
                    name="Dark wallet",
                    category="Accessories",
                    brand="Wildcraft",
                    color="Black",
                    description="Found a dark leather wallet with a small zipper.",
                    location="Outside the college library",
                    report_date=lost_date + timedelta(days=1),
                    identifying_features="Small initials stamped inside.",
                    status="active",
                    owner_id=student.id,
                ),
                Item(
                    type="lost",
                    name="Blue backpack",
                    category="Bags",
                    brand="Northstar",
                    color="Blue",
                    description="Blue backpack with a stitched astronomy patch.",
                    location="Science Building",
                    report_date=lost_date - timedelta(days=1),
                    identifying_features="Astronomy patch on the front pocket.",
                    status="active",
                    owner_id=student.id,
                ),
                Item(
                    type="found",
                    name="Water bottle",
                    category="Drinkware",
                    brand="HydroPeak",
                    color="Green",
                    description="Green steel bottle with a flip lid.",
                    location="Sports Field",
                    report_date=lost_date + timedelta(days=2),
                    identifying_features="White lettering on the bottom.",
                    status="active",
                    owner_id=student.id,
                ),
            ]
            db.add_all(examples)
            db.commit()


@app.get(f"{API_PREFIX}/healthz", response_model=HealthStatus, tags=["health"])
def health_check() -> HealthStatus:
    return HealthStatus(status="ok")


@app.post(
    f"{API_PREFIX}/auth/register",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def register(
    body: RegistrationInput,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> SessionResponse:
    username = body.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Username must have at least 3 characters.")
    if db.scalar(select(User).where(func.lower(User.username) == username.casefold())):
        raise HTTPException(status_code=409, detail="That username is already in use.")
    user = User(username=username, password_hash=hash_password(body.password), role="user")
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="That username is already in use.") from exc
    db.refresh(user)
    token = issue_session(db, user)
    _set_session_cookie(response, token, request)
    return SessionResponse(user=_user_out(user))


@app.post(f"{API_PREFIX}/auth/login", response_model=SessionResponse, tags=["auth"])
def login(
    body: LoginInput,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> SessionResponse:
    user = db.scalar(
        select(User).where(func.lower(User.username) == body.username.strip().casefold())
    )
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    token = issue_session(db, user)
    _set_session_cookie(response, token, request)
    return SessionResponse(user=_user_out(user))


@app.get(f"{API_PREFIX}/auth/me", response_model=UserOut | None, tags=["auth"])
def get_session(user: Annotated[User | None, Depends(current_user)]) -> UserOut | None:
    return _user_out(user) if user is not None else None


@app.post(f"{API_PREFIX}/auth/logout", response_model=Message, tags=["auth"])
def logout(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> Message:
    revoke_session(db, request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path=API_PREFIX, httponly=True, samesite="lax")
    return Message(message="Signed out.")


@app.post(
    f"{API_PREFIX}/items/lost",
    response_model=ItemOut,
    status_code=status.HTTP_201_CREATED,
    tags=["items"],
)
def create_lost_item(
    body: LostItemInput,
    user: Annotated[User, Depends(require_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ItemOut:
    item = Item(
        type="lost",
        name=body.name,
        category=body.category,
        brand=body.brand,
        color=body.color,
        description=body.description,
        location=body.location,
        report_date=body.report_date,
        identifying_features=body.identifying_features,
        status="active",
        owner_id=user.id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _public_item(item)


@app.post(
    f"{API_PREFIX}/items/found",
    response_model=ItemOut,
    status_code=status.HTTP_201_CREATED,
    tags=["items"],
)
def create_found_item(
    body: FoundItemInput,
    user: Annotated[User, Depends(require_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ItemOut:
    item = Item(
        type="found",
        name=body.name,
        category=body.category,
        brand=body.brand,
        color=body.color,
        description=body.description,
        location=body.location,
        report_date=body.report_date,
        identifying_features=body.private_identifying_features,
        status="active",
        owner_id=user.id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _public_item(item)


@app.get(f"{API_PREFIX}/items", response_model=ItemPage, tags=["items"])
def list_items(
    db: Annotated[Session, Depends(get_db)],
    search: str | None = None,
    category: str | None = None,
    item_type: str | None = Query(default=None, alias="type", pattern="^(lost|found)$"),
    report_status: str | None = Query(default=None, alias="status", pattern="^(active|resolved)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50, alias="pageSize"),
) -> ItemPage:
    filters = []
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        filters.append(
            or_(
                Item.name.ilike(pattern),
                Item.category.ilike(pattern),
                Item.brand.ilike(pattern),
                Item.color.ilike(pattern),
                Item.description.ilike(pattern),
                Item.location.ilike(pattern),
            )
        )
    if category:
        filters.append(Item.category == category)
    if item_type:
        filters.append(Item.type == item_type)
    if report_status:
        filters.append(Item.status == report_status)

    total = db.scalar(select(func.count(Item.id)).where(*filters)) or 0
    categories = list(
        db.scalars(select(Item.category).distinct().order_by(Item.category)).all()
    )
    rows = db.scalars(
        select(Item)
        .where(*filters)
        .order_by(Item.created_at.desc(), Item.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return ItemPage(
        items=[_public_item(item) for item in rows],
        total=total,
        page=page,
        page_size=page_size,
        page_count=(total + page_size - 1) // page_size,
        categories=categories,
    )


@app.get(f"{API_PREFIX}/items/{{item_id}}", response_model=ItemOut, tags=["items"])
def get_item(item_id: int, db: Annotated[Session, Depends(get_db)]) -> ItemOut:
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return _public_item(item)


def _source_for_match(
    match_request: MatchInput,
    db: Session,
    report_count: int | None = None,
    seed: int = 42,
):
    if match_request.dataset_id:
        try:
            dataset = get_dataset(match_request.dataset_id, report_count, seed)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        lost_id = match_request.lost_item_id
        lost = next(
            (
                item
                for item in dataset.reports
                if item["type"] == "lost" and (lost_id is None or item["id"] == lost_id)
            ),
            None,
        )
        if lost is None:
            raise HTTPException(status_code=404, detail="Lost report not found in this dataset.")
        return lost, dataset.reports, dataset.info.name, dataset.ground_truth
    if match_request.lost_item_id is None:
        raise HTTPException(status_code=422, detail="Select a lost report or a demo dataset.")
    lost = db.get(Item, match_request.lost_item_id)
    if lost is None:
        raise HTTPException(status_code=404, detail="Lost report not found.")
    reports = list(db.scalars(select(Item).order_by(Item.id)).all())
    return lost, reports, "Stored reports", None


def _run_one(
    body: MatchInput,
    db: Session,
    algorithm: str,
    report_count: int | None = None,
    seed: int = 42,
):
    lost, reports, name, truth = _source_for_match(body, db, report_count, seed)
    try:
        return run_benchmarked(lost, reports, body, algorithm, name, truth)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(f"{API_PREFIX}/match/brute-force", response_model=AlgorithmRun, tags=["matching"])
def match_brute_force(
    body: MatchInput, db: Annotated[Session, Depends(get_db)]
) -> AlgorithmRun:
    return AlgorithmRun.model_validate(_run_one(body, db, "brute_force"))


@app.post(f"{API_PREFIX}/match/heuristic", response_model=AlgorithmRun, tags=["matching"])
def match_heuristic(
    body: MatchInput, db: Annotated[Session, Depends(get_db)]
) -> AlgorithmRun:
    return AlgorithmRun.model_validate(_run_one(body, db, "heuristic"))


def _compare(
    body: MatchInput,
    db: Session,
    report_count: int | None = None,
    seed: int = 42,
) -> CompareResponse:
    lost, reports, name, truth = _source_for_match(body, db, report_count, seed)
    try:
        return CompareResponse.model_validate(
            compare_benchmarked(lost, reports, body, name, truth)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    f"{API_PREFIX}/match/compare",
    response_model=CompareResponse,
    tags=["matching"],
)
def compare_matches(
    body: MatchInput, db: Annotated[Session, Depends(get_db)]
) -> CompareResponse:
    return _compare(body, db)


@app.post(f"{API_PREFIX}/ai/analyze", response_model=AnalyzeResult, tags=["matching"])
def analyze(body: AnalyzeInput) -> AnalyzeResult:
    return AnalyzeResult.model_validate(analyze_description(body.description))


@app.get(f"{API_PREFIX}/claims", response_model=list[ClaimOut], tags=["claims"])
def list_claims(
    user: Annotated[User, Depends(require_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[ClaimOut]:
    statement = select(Claim).order_by(Claim.created_at.desc(), Claim.id.desc())
    if user.role != "admin":
        statement = statement.where(Claim.claimant_id == user.id)
    claims = db.scalars(statement).all()
    return [_claim_out(claim, show_private=user.role == "admin") for claim in claims]


@app.post(
    f"{API_PREFIX}/claims",
    response_model=ClaimOut,
    status_code=status.HTTP_201_CREATED,
    tags=["claims"],
)
def create_claim(
    body: ClaimInput,
    user: Annotated[User, Depends(require_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ClaimOut:
    if body.lost_item_id == body.found_item_id:
        raise HTTPException(status_code=422, detail="A report cannot claim itself.")
    lost = db.get(Item, body.lost_item_id)
    found = db.get(Item, body.found_item_id)
    if lost is None or found is None:
        raise HTTPException(status_code=404, detail="Both reports must exist.")
    valid, reasons = candidate_validation(lost, found, date_tolerance_days=30)
    if not valid:
        raise HTTPException(status_code=409, detail=" ".join(reasons))
    pending = db.scalar(
        select(Claim).where(
            Claim.claimant_id == user.id,
            Claim.lost_item_id == lost.id,
            Claim.found_item_id == found.id,
            Claim.status == "pending",
        )
    )
    if pending:
        raise HTTPException(status_code=409, detail="You already have a pending claim for these reports.")
    claim = Claim(
        lost_item_id=lost.id,
        found_item_id=found.id,
        claimant_id=user.id,
        ownership_evidence=body.ownership_evidence,
        status="pending",
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)
    return _claim_out(claim)


@app.patch(
    f"{API_PREFIX}/claims/{{claim_id}}",
    response_model=ClaimOut,
    tags=["claims"],
)
def review_claim(
    claim_id: int,
    body: ClaimReviewInput,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> ClaimOut:
    # SQLite's immediate write lock makes the status check and both report
    # transitions atomic with respect to competing approvals.
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        claim = db.get(Claim, claim_id)
        if claim is None:
            raise HTTPException(status_code=404, detail="Claim not found.")
        if claim.status != "pending":
            raise HTTPException(status_code=409, detail="This claim has already been reviewed.")

        if body.decision == "approved":
            lost = db.get(Item, claim.lost_item_id)
            found = db.get(Item, claim.found_item_id)
            if lost is None or found is None:
                raise HTTPException(status_code=409, detail="Claim reports are no longer available.")
            try:
                assert_approved_claim_is_valid(lost, found)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            conflicting = db.scalar(
                select(Claim.id).where(
                    Claim.status == "approved",
                    or_(
                        Claim.lost_item_id == lost.id,
                        Claim.found_item_id == found.id,
                    ),
                )
            )
            if conflicting is not None:
                raise HTTPException(
                    status_code=409,
                    detail="One of these reports is already part of an approved claim.",
                )
            lost_update = db.execute(
                update(Item)
                .where(Item.id == lost.id, Item.status == "active")
                .values(status="resolved")
            )
            found_update = db.execute(
                update(Item)
                .where(Item.id == found.id, Item.status == "active")
                .values(status="resolved")
            )
            if lost_update.rowcount != 1 or found_update.rowcount != 1:
                raise HTTPException(
                    status_code=409,
                    detail="A report is no longer active and cannot be resolved.",
                )
            lost.status = "resolved"
            found.status = "resolved"

        claim.status = body.decision
        claim.review_note = body.review_note.strip() if body.review_note else None
        claim.reviewed_by = admin.id
        claim.reviewed_at = datetime.utcnow()
        db.add(
            ClaimVerificationHistory(
                claim_id=claim.id,
                admin_id=admin.id,
                action=body.decision,
                notes=claim.review_note,
            )
        )
        db.flush()
        db.commit()
        db.refresh(claim)
        return _claim_out(claim, show_private=True)
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A conflicting claim was approved at the same time.",
        ) from exc
    except Exception:
        db.rollback()
        raise


@app.get(
    f"{API_PREFIX}/dashboard/stats",
    response_model=DashboardStats,
    tags=["dashboard"],
)
def dashboard_stats(db: Annotated[Session, Depends(get_db)]) -> DashboardStats:
    reports = list(db.scalars(select(Item).order_by(Item.id)).all())
    total_lost = sum(item.type == "lost" for item in reports)
    total_found = sum(item.type == "found" for item in reports)
    active_count = sum(item.status == "active" for item in reports)
    resolved_count = sum(item.status == "resolved" for item in reports)
    categories: dict[str, dict[str, int]] = {}
    for item in reports:
        categories.setdefault(item.category, {"lost": 0, "found": 0})
        categories[item.category][item.type] += 1
    category_rows = [
        {"category": category, **counts}
        for category, counts in sorted(categories.items(), key=lambda pair: pair[0].casefold())
    ]

    active_lost = [item for item in reports if item.type == "lost" and item.status == "active"]
    potential_pairs = 0
    preview_request = MatchInput(
        top_k=20,
        threshold=65,
        date_tolerance_days=1,
        strict_category_filter=False,
        location_date_filter=False,
    )
    for lost in active_lost:
        try:
            result = run_once(lost, reports, preview_request, "brute_force", "Stored reports")
            potential_pairs += len(result["recommendations"])
        except ValueError:
            continue
    return DashboardStats(
        total_lost=total_lost,
        total_found=total_found,
        active_reports=active_count,
        potential_matches=potential_pairs,
        resolved_reports=resolved_count,
        categories=category_rows,
        recent_reports=[
            _public_item(item)
            for item in sorted(
                reports, key=lambda item: (item.created_at, item.id), reverse=True
            )[:6]
        ],
    )


@app.get(
    f"{API_PREFIX}/experiments/datasets",
    response_model=list[DatasetInfo],
    tags=["experiments"],
)
def list_datasets():
    return dataset_catalog()


@app.post(
    f"{API_PREFIX}/experiments/generate",
    response_model=DatasetInfo,
    tags=["experiments"],
)
def generate_dataset(body: GenerateDatasetInput):
    try:
        return get_dataset(
            f"synthetic-{body.report_count}-seed{body.seed}",
            body.report_count,
            body.seed,
        ).info
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    f"{API_PREFIX}/experiments/run",
    response_model=CompareResponse,
    tags=["experiments"],
)
def run_experiment(
    body: ExperimentInput,
    db: Annotated[Session, Depends(get_db)],
) -> CompareResponse:
    matching_input = MatchInput(
        lost_item_id=body.lost_item_id,
        dataset_id=body.dataset_id,
        top_k=body.top_k,
        threshold=body.threshold,
        date_tolerance_days=body.date_tolerance_days,
        weights=body.weights,
        strict_category_filter=body.strict_category_filter,
        location_date_filter=body.location_date_filter,
    )
    return _compare(matching_input, db, body.report_count, body.seed)

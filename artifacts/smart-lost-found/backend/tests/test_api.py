from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
from models import Item, User
from security import hash_password


@pytest.fixture
def client(tmp_path):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'api-test.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(test_engine)
    with TestSession() as db:
        db.add_all(
            [
                User(
                    username="admin",
                    password_hash=hash_password("AdminPass123!"),
                    role="admin",
                ),
                User(
                    username="student",
                    password_hash=hash_password("StudentPass123!"),
                    role="user",
                ),
            ]
        )
        db.commit()

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()


def login(client: TestClient, username="student", password="StudentPass123!"):
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response


def create_item(client: TestClient, endpoint: str, *, name: str, **extra):
    data = {
        "name": name,
        "category": "Accessories",
        "brand": "Northstar",
        "color": "Black",
        "description": "A black item with an identifying feature.",
        "location": "College Library",
        "reportDate": "2026-10-25",
        **extra,
    }
    return client.post(endpoint, json=data)


def test_health_docs_and_empty_dashboard_are_available(client):
    assert client.get("/api/healthz").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    stats = client.get("/api/dashboard/stats")
    assert stats.status_code == 200
    assert stats.json()["activeReports"] == 0


def test_registration_validates_password_and_creates_cookie_session(client):
    weak = client.post(
        "/api/auth/register", json={"username": "new-user", "password": "short"}
    )
    assert weak.status_code == 422
    response = client.post(
        "/api/auth/register",
        json={"username": "new-user", "password": "LongEnoughPass123"},
    )
    assert response.status_code == 201
    assert response.json()["user"]["role"] == "user"
    assert client.get("/api/auth/me").json()["username"] == "new-user"


def test_login_is_case_insensitive_and_rejects_bad_password(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "STUDENT", "password": "StudentPass123!"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "student"
    bad = client.post(
        "/api/auth/login",
        json={"username": "student", "password": "wrong-password"},
    )
    assert bad.status_code == 401


def test_logout_revokes_session(client):
    login(client)
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").json() is None


def test_item_creation_browse_filter_pagination_and_private_feature_protection(client):
    login(client)
    lost = create_item(
        client,
        "/api/items/lost",
        name="Leather wallet",
        identifyingFeatures="Initials inside",
    )
    found = create_item(
        client,
        "/api/items/found",
        name="Wallet",
        privateIdentifyingFeatures="Initials inside",
    )
    assert lost.status_code == 201
    assert found.status_code == 201
    assert "privateIdentifyingFeatures" not in found.json()
    found_id = found.json()["id"]
    assert "privateIdentifyingFeatures" not in client.get(
        f"/api/items/{found_id}"
    ).json()
    page = client.get("/api/items", params={"type": "found", "pageSize": 1})
    assert page.status_code == 200
    assert page.json()["total"] == 1
    assert page.json()["pageSize"] == 1
    assert page.json()["items"][0]["type"] == "found"


def test_item_creation_requires_auth_and_validation(client):
    missing_auth = create_item(client, "/api/items/lost", name="Wallet")
    assert missing_auth.status_code == 401
    login(client)
    invalid = client.post(
        "/api/items/lost",
        json={
            "name": "",
            "category": "Accessories",
            "description": "x",
            "location": "Library",
            "reportDate": "not-a-date",
        },
    )
    assert invalid.status_code == 422


def test_matching_and_experiment_endpoints_return_actual_python_metrics(client):
    response = client.post(
        "/api/experiments/run",
        json={
            "datasetId": "filtering-tradeoff",
            "topK": 5,
            "threshold": 40,
            "dateToleranceDays": 1,
            "strictCategoryFilter": True,
            "locationDateFilter": False,
            "seed": 42,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["bruteForce"]["evaluatedCandidates"] == 2
    assert data["heuristic"]["evaluatedCandidates"] == 1
    assert data["candidateReductionPercentage"] == 50
    assert data["bruteForce"]["executionTimeMs"] >= 0


def test_match_endpoint_requires_source_and_handles_unknown_item(client):
    missing = client.post(
        "/api/match/compare",
        json={
            "topK": 5,
            "threshold": 40,
            "dateToleranceDays": 1,
            "strictCategoryFilter": True,
            "locationDateFilter": False,
        },
    )
    assert missing.status_code == 422
    unknown = client.post(
        "/api/match/brute-force",
        json={
            "lostItemId": 98765,
            "topK": 5,
            "threshold": 40,
            "dateToleranceDays": 1,
            "strictCategoryFilter": True,
            "locationDateFilter": False,
        },
    )
    assert unknown.status_code == 404


def test_rule_based_ai_endpoint_returns_structured_attributes(client):
    response = client.post(
        "/api/ai/analyze",
        json={"description": "Lost dark leather wallet near the college library."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "deterministic_rules"
    assert body["attributes"]["itemName"] == "wallet"
    assert body["attributes"]["location"] == "college library"


def test_claims_require_authentication_and_admin_role(client):
    assert client.get("/api/claims").status_code == 401
    login(client)
    not_admin = client.patch(
        "/api/claims/1", json={"decision": "approved"}
    )
    assert not_admin.status_code == 403


def test_complete_claim_approval_resolves_reports_and_keeps_private_feature_admin_only(client):
    login(client)
    lost = create_item(
        client, "/api/items/lost", name="Leather wallet", identifyingFeatures="Stamped initials"
    ).json()
    found = create_item(
        client,
        "/api/items/found",
        name="Dark wallet",
        privateIdentifyingFeatures="Stamped initials",
    ).json()
    assert "privateIdentifyingFeatures" not in found
    assert "privateIdentifyingFeatures" not in client.get(
        f"/api/items/{found['id']}"
    ).json()
    public_items = client.get("/api/items", params={"type": "found"}).json()["items"]
    assert all("privateIdentifyingFeatures" not in item for item in public_items)
    claim = client.post(
        "/api/claims",
        json={
            "lostItemId": lost["id"],
            "foundItemId": found["id"],
            "ownershipEvidence": "My initials are stamped inside the wallet.",
        },
    )
    assert claim.status_code == 201, claim.text
    claim_id = claim.json()["id"]
    assert claim.json()["privateFeatureForReview"] is None
    own_claim = client.get("/api/claims").json()[0]
    assert own_claim["privateFeatureForReview"] is None

    client.post("/api/auth/logout")
    login(client, "admin", "AdminPass123!")
    review = client.patch(
        f"/api/claims/{claim_id}",
        json={"decision": "approved", "reviewNote": "Evidence checked."},
    )
    assert review.status_code == 200, review.text
    assert review.json()["status"] == "approved"
    assert review.json()["privateFeatureForReview"] == "Stamped initials"
    assert client.get(f"/api/items/{lost['id']}").json()["status"] == "resolved"
    assert client.get(f"/api/items/{found['id']}").json()["status"] == "resolved"


def test_duplicate_approvals_cannot_resolve_reports_twice(client):
    login(client)
    lost = create_item(client, "/api/items/lost", name="Blue wallet").json()
    found = create_item(
        client, "/api/items/found", name="Blue wallet", privateIdentifyingFeatures="Blue mark"
    ).json()
    payload = {
        "lostItemId": lost["id"],
        "foundItemId": found["id"],
        "ownershipEvidence": "I know the identifying blue mark inside.",
    }
    first = client.post("/api/claims", json=payload).json()
    client.post("/api/auth/logout")
    client.post(
        "/api/auth/register",
        json={"username": "second-user", "password": "SecondUserPass123"},
    )
    second = client.post("/api/claims", json=payload).json()
    client.post("/api/auth/logout")
    login(client, "admin", "AdminPass123!")
    approved = client.patch(
        f"/api/claims/{first['id']}", json={"decision": "approved"}
    )
    conflict = client.patch(
        f"/api/claims/{second['id']}", json={"decision": "approved"}
    )
    assert approved.status_code == 200
    assert conflict.status_code == 409
    assert client.get(f"/api/items/{lost['id']}").json()["status"] == "resolved"


def test_rejected_claim_does_not_resolve_reports(client):
    login(client)
    lost = create_item(client, "/api/items/lost", name="Notebook").json()
    found = create_item(client, "/api/items/found", name="Notebook").json()
    claim = client.post(
        "/api/claims",
        json={
            "lostItemId": lost["id"],
            "foundItemId": found["id"],
            "ownershipEvidence": "I can describe the marked inside cover.",
        },
    ).json()
    client.post("/api/auth/logout")
    login(client, "admin", "AdminPass123!")
    rejected = client.patch(
        f"/api/claims/{claim['id']}",
        json={"decision": "rejected", "reviewNote": "Details did not match."},
    )
    assert rejected.status_code == 200
    assert client.get(f"/api/items/{lost['id']}").json()["status"] == "active"
    assert client.get(f"/api/items/{found['id']}").json()["status"] == "active"


def test_experiment_catalog_and_generation_do_not_modify_user_reports(client):
    login(client)
    original = create_item(client, "/api/items/lost", name="Original report").json()
    count_before = client.get("/api/items").json()["total"]
    catalog = client.get("/api/experiments/datasets")
    generated = client.post(
        "/api/experiments/generate", json={"reportCount": 100, "seed": 7}
    )
    count_after = client.get("/api/items").json()["total"]
    assert catalog.status_code == 200
    assert {dataset["id"] for dataset in catalog.json()} >= {
        "exact",
        "partial",
        "missing-information",
        "no-matches",
        "filtering-tradeoff",
        "synthetic-100-seed42",
    }
    assert generated.status_code == 200
    assert generated.json()["id"] == "synthetic-100-seed7"
    assert count_before == count_after
    assert client.get(f"/api/items/{original['id']}").status_code == 200


def test_sqlite_data_survives_engine_recreation(tmp_path):
    url = f"sqlite:///{tmp_path / 'persistent.db'}"
    first_engine = create_engine(
        url, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(first_engine)
    FirstSession = sessionmaker(bind=first_engine)
    with FirstSession() as db:
        db.add(
            Item(
                type="lost",
                name="Persistent item",
                category="Other",
                description="A report saved to SQLite.",
                location="Campus",
                report_date=date(2026, 10, 29),
                status="active",
            )
        )
        db.commit()
    first_engine.dispose()

    second_engine = create_engine(
        url, connect_args={"check_same_thread": False}
    )
    SecondSession = sessionmaker(bind=second_engine)
    with SecondSession() as db:
        saved = db.scalar(select(Item).where(Item.name == "Persistent item"))
        assert saved is not None
    second_engine.dispose()

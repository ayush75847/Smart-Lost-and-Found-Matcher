# Smart Lost & Found Matcher

A campus lost-and-found platform and MCA Problem Solving demonstration. The Python backend stores reports in SQLite, ranks plausible lost/found pairs, explains its calculations, and supports a role-protected ownership-claim workflow. The React interface includes an algorithm lab that compares exhaustive and filtered searches using measured backend results.

## Problem statement

Lost property reports often use different words for the same item. This project uses text normalization, simple synonym recognition, set similarity, weighted scoring, and explicit eligibility rules to prioritize candidate matches. Scores are explainable similarities, not probabilities or proof of ownership. A human administrator reviews claims before reports are resolved.

## Features

- Create lost and found reports; browse, search, filter, and paginate reports.
- Compare matching candidates with weighted scores, evidence coverage, Jaccard sets, matching and differing attributes, and plain-language explanations.
- Run brute-force, heuristic, and side-by-side comparisons on stored reports or isolated demo datasets.
- Generate deterministic synthetic datasets with 100, 500, or 1,000 reports.
- View real candidate counts, `perf_counter` timing medians, score distributions, Top-K ranking, decision paths, invariants, and labeled-dataset precision/recall.
- Analyze descriptions using a deterministic Python rules engine without an API key.
- Submit ownership claims; administrators approve or reject them. Approval resolves both reports in one SQLite transaction and prevents conflicting approvals.
- Hide private identifying features from public report responses and regular users.
- Demonstrate all eight syllabus topics plus Polya's four-step method.

## Technology

- React, Vite, Tailwind CSS, Recharts, Lucide React, and Wouter routing.
- Python 3.11+, FastAPI, Pydantic, SQLAlchemy, and SQLite.
- Pytest and FastAPI TestClient.
- Standard-library algorithms: `re`, `difflib`, `heapq`, `statistics`, `time`, `datetime`, and Python sets.

## Demo accounts

These accounts are seeded for classroom demonstrations:

| Role | Username | Password |
| --- | --- | --- |
| User | `student` | `StudentPass123!` |
| Admin | `admin` | `AdminPass123!` |

Passwords are stored as salted PBKDF2-SHA256 hashes. The accounts are intentionally predictable demo credentials: replace them and disable demo seeding before exposing a public production instance.

## Run in Replit

The project has two managed services:

- Web app: `pnpm --filter @workspace/smart-lost-found run dev`
- FastAPI service: `pnpm --filter @workspace/api-server run dev`

Start both workflows and use the app preview. The shared `/api` route forwards API calls to FastAPI. FastAPI's interactive documentation is available at `/docs`; its OpenAPI JSON is at `/openapi.json`.

The Python packages are listed in `artifacts/smart-lost-found/backend/requirements.txt`. SQLite data is stored at `artifacts/smart-lost-found/backend/data/lost_found.db`; set `LOST_FOUND_DATABASE_URL` to a different SQLite URL to use another file. Demo reports are inserted only when the report table is empty, and demo datasets never overwrite stored reports.

## Tests

From the backend directory:

```bash
cd artifacts/smart-lost-found/backend
../../../.pythonlibs/bin/python -m pytest -q
```

The suite covers algorithm formulas, eligibility rules, filtering trade-offs, scoring bounds, generated datasets, authentication, persistence, report APIs, private-field protection, and the claim approval flow.

## API overview

The canonical API contract is `lib/api-spec/openapi.yaml`; FastAPI serves the same `/api` endpoints.

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout`
- `GET /api/dashboard/stats`
- `GET /api/items`, `GET /api/items/{id}`, `POST /api/items/lost`, `POST /api/items/found`
- `POST /api/match/brute-force`, `POST /api/match/heuristic`, `POST /api/match/compare`
- `POST /api/ai/analyze`
- `GET /api/claims`, `POST /api/claims`, `PATCH /api/claims/{id}`
- `GET /api/experiments/datasets`, `POST /api/experiments/generate`, `POST /api/experiments/run`

## Screenshots

Add screenshots of the dashboard and algorithm lab here.

## Limitations and future scope

- Similarity is lexical and rule-based; it does not establish item ownership or use embeddings.
- The default attribute weights and date windows are teaching defaults and should be tuned with representative labeled reports.
- Synthetic data is reproducible but does not represent measured real-world performance or matching accuracy.
- Precision and recall are shown only for datasets with ground-truth labels, and are scoped to the selected lost report and current threshold/Top-K.
- SQLite is suitable for a small academic project and local deployment; a multi-instance service needs a shared transactional database.
- The deterministic description analyzer does not call an LLM; it leaves unsupported attributes blank.
- The app has no email verification, password reset, or account lockout workflow.

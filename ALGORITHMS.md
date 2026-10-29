# Algorithms and Problem-Solving Concepts

This document describes the implementations in `artifacts/smart-lost-found/backend/algorithms/`.

## 1. Pattern recognition

`normalize_text` lowercases text, trims and collapses whitespace, and removes most punctuation. `tokenize` removes common report words and maps a small set of synonyms to shared tokens (`dark`/`black`, `navy`/`blue`, `cellphone`/`phone`, `rucksack`/`backpack`). `text_similarity` combines token Jaccard similarity with `difflib.SequenceMatcher` for sufficiently close strings.

Example: “black leather wallet near library” and “dark wallet outside the library” share normalized tokens `{black, leather, wallet, library}` and receive strong lexical similarity.

The technique is deliberately transparent and limited: it does not understand context, misspellings beyond simple fuzzy similarity, or broad semantic relationships.

## 2. Heuristic reasoning

For an available attribute set `A`, weights are normalized to sum to one. Missing optional values are excluded from the denominator:

```text
score = 100 × Σ(wᵢ × similarityᵢ) / Σ(wᵢ for available attributes)
evidence coverage = 100 × Σ(wᵢ for available attributes) / Σ(all configured wᵢ)
```

Default weights:

| Attribute | Weight |
| --- | ---: |
| Category | 25% |
| Item name | 20% |
| Color | 15% |
| Brand | 10% |
| Location | 15% |
| Date | 5% |
| Description | 10% |

Date similarity is `max(0, 1 - |lostDate - foundDate| / 30 days)`. A found date earlier than the lost date by more than the configured tolerance is rejected as an eligibility condition, rather than treated as a low score.

Example: if all available attributes have similarity `1.0`, but brand and color are missing, the score remains 100 while evidence coverage falls below 100%. This avoids turning unknown fields into negative evidence.

Interpretation bands (not probabilities): 85–100 strong, 65–84 moderate, 40–64 weak, below 40 low similarity.

## 3. Set theory: Jaccard similarity

Descriptions become normalized Python token sets `A` and `B`:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

For `{black, leather, wallet, zipper}` and `{black, leather, wallet, small}`, the intersection has three tokens and the union has five, so `J = 3/5 = 0.60`. When both sets are empty the result is unavailable (`None`), not 100%.

## 4. Brute force versus heuristic search

Both methods use the same weighted score:

- **Brute force:** scans the entire found-report set, applies mandatory eligibility rules, scores every remaining candidate, then ranks results.
- **Heuristic:** uses the same mandatory rules, optionally removes inactive reports and mismatched categories, and can apply an optional location/date proximity filter before scoring.

For `F` found reports, `H` candidates that reach scoring, and `C` the cost of comparing one pair:

```text
Brute force: O(F × C + F log F)
Heuristic:   O(F + H × C + H log K), where H ≤ F and K is requested Top-K
```

Top-K uses a bounded `heapq.nsmallest` selection with deterministic report-ID tie-breaking. The API measures a warm-up and five measured runs using `time.perf_counter`; displayed duration is the median of the measured runs. The filtering trade-off dataset includes a known relevant candidate with an incorrect category label so the heuristic's recall risk is visible.

Candidate reduction is calculated from the actual similarity-call counts:

```text
reduction (%) = ((brute-force calculations - heuristic calculations)
                 / brute-force calculations) × 100
```

If the baseline count is zero, reduction is unavailable. Runtime depends on the host and dataset; the app reports measurements rather than fixed claims.

## 5. Conditional statements

`candidate_validation` checks:

1. The selected report is lost.
2. The candidate is found.
3. Both reports are active.
4. The IDs are different.
5. The found date is not earlier than the lost date beyond the configured tolerance.

Each rejected report is skipped before scoring, and the API includes actual filter counts and reasons in its decision path.

## 6. Extreme principle and Top-K

`maximum_score` selects the best score with `max`. `rank_top_k` uses `heapq.nsmallest` on `(-score, report_id)`, which returns descending scores and resolves ties by ascending report ID. The lab displays the top candidates and highlights their score breakdowns.

## 7. Invariants

Matching checks that results are from the selected report set, no report matches itself, only lost-to-found pairs are evaluated, resolved reports are excluded, and each score is within `[0, 100]`.

Claim approval additionally requires an administrator, valid active reports, no existing approved claim involving either report, and an atomic transition of both report statuses to `resolved`. SQLite partial unique indexes provide a second guard against conflicting approvals. A verification-history row records the decision.

## 8. Prompt-based problem solving

`analyze_description` implements the required no-key fallback using regular expressions, keyword dictionaries, synonym mappings, and normalized text. It can extract an item name, category, color, material, brand, and nearby location. Unknown values are returned as `null`; the analyzer does not invent attributes.

The result is labeled `deterministic_rules`. No private verification answer is passed to an external service. No LLM is required for the app to work.

## Polya's four steps

1. **Understand:** identify the lost/found reports and the matching objective.
2. **Plan:** define normalized attributes, sets, weights, constraints, and candidate-search approaches.
3. **Execute:** run the Python implementation and return a ranked, explained result.
4. **Review:** compare actual search work and timing, inspect invariants, and consider false positives, missed labels, and missing evidence.

## Experimental methodology

Demo datasets are generated in isolation and never replace user reports. The catalog includes exact, partial, missing-information, no-match, invalid/conflicting, filtering trade-off, and reproducible 100/500/1,000-report cases. Synthetic datasets are generated from a deterministic seed. Precision and recall are calculated only when a dataset has a known positive pair for the selected lost report; undefined precision or recall is `N/A`.

## Complexity notes

For `T` text characters and `W` extracted words, normalization/tokenization are linear in input size, set operations are expected linear in set size, and fuzzy sequence matching can be quadratic in string length in the worst case. For `F` found reports and `K` requested results, brute-force ranking is `O(F × C + F log K)` and the prefiltered search is `O(F + H × C + H log K)`.

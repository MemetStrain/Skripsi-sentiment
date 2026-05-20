# Flowchart Restructure Report (Option C: Hybrid)

**Date:** 2026-05-15
**Branch:** `docs/flowchart-restructure`
**Scope:** Additive documentation only. No production code modified.

## Goal

Restructure flowchart documentation to follow Option C (hybrid):
keep one overall orchestration flowchart, plus add detail flowcharts for the
three processes with non-trivial control logic (P3, P4, P5). P1 and P2 stay
covered by the orchestration view because they are mostly linear.

Mirrors the structure already used for structure charts
(`diagrams/structure_charts/` per-process directory) so the documentation set
becomes visually and structurally consistent.

## Files Created

| File | Lines | Purpose |
|------|------:|---------|
| `diagrams/flowcharts/index.html` | 171 | Index + DFD→flowchart mapping (yes/no detail table) |
| `diagrams/flowcharts/orchestration.html` | 242 | Scheduler pipeline (P1→P2→P3→P4) with skip-if-current guards |
| `diagrams/flowcharts/P3_sentiment.html` | 221 | Title-only FinBERT-Tone + 3-way aggregate branching |
| `diagrams/flowcharts/P4_hmm_decoding.html` | 226 | Frozen-params HMM + forward filter + skip-existing-dates |
| `diagrams/flowcharts/P5_dashboard.html` | 295 | 4-route URL dispatch + forecasts_api loop (horizon × anchor + lru_cache) |
| `FLOWCHART_RESTRUCTURE_REPORT.md` | this | report |
| **Total new lines** | **1155** | (HTML/Tailwind/Mermaid) |

## Files Modified (Navigation Updates)

| File | Change |
|------|--------|
| `diagrams/data_flow_diagram.html` | nav pill: `Flowchart` → `Flowcharts` (`flowcharts/index.html`) |
| `diagrams/structure_chart.html` (legacy) | same |
| `diagrams/pseudocode.html` | same |
| `diagrams/structure_charts/index.html` | same (`../flowcharts/index.html`) |
| `diagrams/structure_charts/P1_price_acquisition.html` | added `Flowcharts (Index)` pill |
| `diagrams/structure_charts/P2_news_acquisition.html` | same |
| `diagrams/structure_charts/P3_sentiment.html` | same |
| `diagrams/structure_charts/P4_hmm_decoding.html` | same |
| `diagrams/structure_charts/P5_dashboard.html` | same |
| `diagrams/flowchart.html` (legacy, preserved) | self-pill renamed to `Flowchart (Legacy, single file)`, added `Flowcharts (per Process) →` pill, added amber legacy notice banner inside `<main>` |

## Phase Results

### Phase 1 — Pre-Execution Verification

- Branch: Initially `main` (would normally abort per spec), but user explicitly authorized continuation.
- Working tree: `diagrams/structure_charts/P5_dashboard.html` had uncommitted changes; resolved by switching to existing `docs/flowchart-restructure` branch which had those changes pre-committed (`d3f4333`).
- All required existing files confirmed present.

### Phase 2 — Discovery Summary

Entry points and decision points extracted from production code (read-only):

| File | LOC | Entry-point(s) | Key decision branches |
|------|----:|----------------|------------------------|
| `scheduler/sentiment_runner.py` | 185 | `run_sentiment_on_articles`, `compute_sentiment_aggregates` | empty articles, empty title (degenerate Neutral), batch exception |
| `scheduler/hmm_updater.py` | 250 | `update_hmm_states(db, write_existing=False)` | params=None (error+return), <50 prices, <30 feat rows, date in existing_dates AND not write_existing |
| `website/web/views.py` | (post-migration) | `dashboard, news, about, forecasts_api` | Firebase init ValueError, param int parse + clamp, `forecast_meta/Daily` missing → 503 (metrics N/A), `forecasts` collection empty → 503, Exception → 500 |
| `prediction/inference.py` | (relocated from `website/web/predictor.py` during the precompute-forecasts migration) | `compute_forecast_trails, build_inference_frame, load_model, load_winners` | `winners.json` missing (FileNotFoundError), model artifact missing, no winner for h, NaN in X (skip), feature_cols mismatch (ValueError) — same error paths, now invoked from `scheduler/precompute_forecasts.py` |

### Phase 3–8 — File Creation

All 5 new HTML files created with the same template style as
`structure_charts/P*.html` (teal header, nav pills, Tailwind + Mermaid CDN,
print CSS, classDef colors).

### Phase 9 — Navigation Updates

10 files updated. All nav pills resolve correctly (Phase 10 cross-ref check).

### Phase 10 — Sanity Test Results

| Test | Result | Notes |
|------|:------:|-------|
| 1. File existence (5 new files) | **PASS** | All 5 files present in `diagrams/flowcharts/` |
| 2. HTML validity basics | **PASS** | All files: `<!DOCTYPE html>`, matching `<html>/<head>/<body>`, Tailwind CDN present, Mermaid CDN on 4 diagram pages (intentionally omitted on `index.html` — mirrors `structure_charts/index.html` which also has no Mermaid) |
| 3. Mermaid syntax | **PASS** | All diagram files: `flowchart TD`, 4–5 `classDef` definitions, 27–47 `:::` references |
| 4. Cross-reference integrity | **PASS** | All `href` targets resolve to existing files; in-page anchors `#p1` and `#p2` confirmed present in `orchestration.html` |
| 5. Browser preview | SKIPPED | Optional per spec |

## Discrepancies Encountered (Code vs Plan)

Per task constraint #8, where the spec described a flow that differed from
actual code, I followed the code. Items noted:

1. **P3 — aggregate orchestration is 3-way, not 2-way.**
   Spec described: *"empty → full rebuild; else → incremental."* Actual code
   in `scheduler/main.py::run_daily_update` has three branches:
   - `news_changed AND new_articles` → incremental for affected dates
   - `(else) AND _is_aggregates_empty(db)` → full rebuild from CSV
   - `else` → skip (no-op, already up to date)
   The P3 flowchart implements all three branches.

2. **P4 — `_forward_filter` is called ONCE on the full X matrix, not per-row.**
   Spec described a per-row loop calling `_forward_filter(model, X[:t+1])`.
   Actual code: `states = _forward_filter(model, X)` runs once over the
   complete observation matrix; per-row iteration is a separate loop that
   only builds `to_write[]` from the precomputed `states` vector. The
   flowchart shows the actual single-call structure.

3. **P4 — additional guards not mentioned in spec.**
   Spec mentioned the params-missing error. Code has two additional
   early-returns:
   - `len(price_rows) < 50` → log.warning, return
   - `len(feat_df) < 30` → log.warning, return
   Both included in the P4 flowchart as error nodes.

4. **Legacy `diagrams/flowchart.html` had stale sentiment label** — node S2R
   was labeled `(FinBERT-Tone, sentence-level)`. The current implementation
   is title-only. Corrected to `(FinBERT-Tone, title-only)` in
   `orchestration.html`. Legacy `flowchart.html` left as-is (preserved as
   reference, with banner pointing to the new index).

## Things to Manually Verify

1. Open `diagrams/flowcharts/index.html` in a browser. Confirm:
   - Header renders correctly (teal banner).
   - Mapping table shows 5 rows with correct yes/no badges.
   - All 4 quick-link cards navigate to their target pages.
   - Nav pills work both directions (to DFD, SC index, legacy flowchart, pseudocode).

2. Open each detail flowchart (`orchestration`, `P3`, `P4`, `P5`). Confirm:
   - Mermaid renders without parser errors.
   - Cross-reference link to corresponding `structure_charts/P*.html` works.
   - "active" nav-pill styling appears on the current page.

3. Open `diagrams/flowchart.html` (legacy). Confirm:
   - The amber legacy notice banner appears above the Pendahuluan section.
   - The new `Flowcharts (per Process) →` pill in nav works.

4. Open `diagrams/data_flow_diagram.html` / `pseudocode.html` /
   `structure_chart.html`. Confirm the `Flowcharts` pill in each works.

5. Visual consistency check: place the new
   `flowcharts/orchestration.html` next to the legacy
   `flowchart.html` (or `structure_charts/P3_sentiment.html` next to
   `flowcharts/P3_sentiment.html`). Color scheme, header style, table
   borders should match.

## Commit Plan (per spec, 4 atomic commits)

1. `docs(flowchart): add orchestration + index files`
   - `diagrams/flowcharts/index.html` (new)
   - `diagrams/flowcharts/orchestration.html` (new)

2. `docs(flowchart): add P3 sentiment detail flowchart`
   - `diagrams/flowcharts/P3_sentiment.html` (new)

3. `docs(flowchart): add P4 HMM + P5 dashboard detail flowcharts`
   - `diagrams/flowcharts/P4_hmm_decoding.html` (new)
   - `diagrams/flowcharts/P5_dashboard.html` (new)

4. `docs(flowchart): update navigation across sibling diagram files`
   - `diagrams/data_flow_diagram.html`
   - `diagrams/structure_chart.html`
   - `diagrams/pseudocode.html`
   - `diagrams/flowchart.html` (legacy banner + new pill)
   - `diagrams/structure_charts/index.html`
   - `diagrams/structure_charts/P1_price_acquisition.html`
   - `diagrams/structure_charts/P2_news_acquisition.html`
   - `diagrams/structure_charts/P3_sentiment.html`
   - `diagrams/structure_charts/P4_hmm_decoding.html`
   - `diagrams/structure_charts/P5_dashboard.html`
   - `FLOWCHART_RESTRUCTURE_REPORT.md` (this report, included in final commit)

## Out-of-scope Files (NOT Modified)

Verified by `git status --short`. No files under these paths touched:
- `scheduler/`
- `website/` (except previously committed structure-chart cosmetic change on `d3f4333`)
- `prediction/`
- `markov/`
- `news/`
- `cpo/`

No npm dependencies or external libraries added. Same Tailwind + Mermaid
CDN pattern as existing diagram files.

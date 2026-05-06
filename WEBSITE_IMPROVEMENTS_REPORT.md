# Website Improvements Report — 2026-05-06

## Summary
- Branch: `cleanup/remove-auth`
- Total commits: 3 (one per improvement, atomic)
- Files modified: `website/web/views.py`, `website/web/templates/dashboard.html`
- No `news.html` changes were needed — pagination clamp is purely backend.

### Commits
| SHA      | Message                                                                                                |
|----------|--------------------------------------------------------------------------------------------------------|
| 2d7a08f  | fix(news): clamp page_number to valid range to prevent empty results on filter change                   |
| 5cb87a4  | feat(dashboard): add per-horizon expandable cards with Base vs CSA comparison                           |
| 3057eb2  | feat(dashboard): add 90-day sentiment trend chart from sentiment_aggregates                             |

## Improvement 1: News Pagination Clamp
- File: `website/web/views.py`
- Lines changed: +1
- Behavior: `page_number` is now clamped to `[1, total_pages]` after `total_pages` is computed. Visiting `?page=999&sentiment=Positive` no longer renders an empty grid; it now resolves to the last valid page.

## Improvement 2: Per-Horizon Cards
- Files: `website/web/views.py`, `website/web/templates/dashboard.html`
- Backend lines added/replaced: ~40 (the previous h=1-only metrics block was replaced wholesale; backward-compatible h=1 summary preserved for the existing 4-card header row).
- Frontend lines added: ~15 HTML + ~95 JS.
- New Firestore queries: 14 individual `predictions` doc `.get()` calls (1 model × 2 variants × 7 horizons). No composite index required.
- New context key: `horizon_data` (JSON-encoded list of 7 horizon entries).
- Empty-context fallback (`_empty_dashboard_ctx`) updated to include `horizon_data`.

## Improvement 3: Sentiment Trend Chart
- Files: `website/web/views.py`, `website/web/templates/dashboard.html`
- Backend lines added: ~22.
- Frontend lines added: ~8 HTML + ~46 JS.
- New Firestore queries: 1 (`sentiment_aggregates` filtered by `date >= three_months_ago`, ordered by date; `frequency == 'Daily'` filter applied in Python).
- New context key: `sentiment_data` (JSON-encoded list of `{date, positive_prob, negative_prob, neutral_prob, sentiment_score}`).
- Empty-context fallback updated to include `sentiment_data`.
- `three_months_ago` was already defined at the top of `dashboard()` (as `.isoformat()` string) — reused directly.

## Sanity Tests
| Test                                                          | Result |
|---------------------------------------------------------------|--------|
| `python manage.py check`                                      | PASS — System check identified no issues |
| `python manage.py runserver 8000` (server boot)               | PASS — server started, no tracebacks |
| `curl -i http://localhost:8000/`                              | 200 |
| `curl -i "http://localhost:8000/news/?page=999&sentiment=Positive"` | 200 (clamped to page 193 of 193 — last valid page, NOT empty) |
| `curl -i http://localhost:8000/news/`                         | 200 |
| `curl -i http://localhost:8000/about/`                        | 200 |
| `python -c "from web import views"`                           | PASS — imports OK |
| `get_template('dashboard.html')` parse                        | PASS |
| Dashboard HTML contains `Per-Horizon Model Performance`, `horizon-cards`, `Sentiment Trend`, `sentimentChart` markers | PASS |

## Issues Encountered
- The system `python` resolves to a Microsoft Store stub on this Windows machine. Used the project's venv interpreter at `website/venv/Scripts/python.exe` for all sanity tests. No code changes needed.

## Things to Verify Manually
1. Visit `http://localhost:8000/`:
   - Existing 4 metric cards (top row) — should still show MAPE, R², Directional Acc., Best Model. The h=1 summary now derives from the new horizon-data fetch (XGBoost BASE/CSA, whichever has lower MAPE at h=1).
   - **NEW:** "Per-Horizon Model Performance" card with 7 H1–H7 horizon buttons, each showing best variant badge (BASE/CSA, teal for CSA, slate for BASE) and best MAPE %.
   - Click a horizon card (e.g. H3) → an expandable detail panel renders below with a Base vs CSA comparison table (MAPE / RMSE / R² / Dir. Accuracy / Predicted Price / Predicted Date). Better values are highlighted in green.
   - Click the same card again (or the "Collapse ×" button) → panel collapses.
   - **NEW:** "Sentiment Trend (90d)" chart with three lines (Positive / Negative / Neutral), y-axis 0–1.
2. The existing sidebar prediction panel ("Run Prediction") and price chart should be **unchanged** — Improvement 2's per-horizon cards are an independent main-content section.
3. `/news/?page=999&sentiment=Positive` → should land on the last valid Positive page (clamping verified server-side; HTTP 200, non-empty grid).
4. Spot-check the sentiment chart's tooltip formatting (4 decimal places per series).

## Compliance Checklist
- [x] Branch + clean tree verified before edits
- [x] Three atomic commits, one per improvement
- [x] Only `views.py`, `dashboard.html` modified — no out-of-scope files touched
- [x] No new dependencies (Chart.js + chartjs-plugin-annotation already loaded in `base.html`)
- [x] No Firestore schema changes — read-only on existing `predictions`, `sentiment_aggregates` collections
- [x] All Phase 4 sanity tests PASS

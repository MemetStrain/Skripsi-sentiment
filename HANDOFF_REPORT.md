# Handoff Report — CPO Dashboard Website Improvements

**Date:** 2026-05-06
**Repo:** `d:\Skripsi1`
**Final branch state:** `main` pushed to `origin/main`. Working tree clean.

---

## What was done

Three independent improvements were applied per the `CLAUDE_CODE_PROMPT_PHASE_A2_FINAL.md` plan, then the feature branch (`cleanup/remove-auth`) was fast-forward merged into local `main`, and a documentation report was committed on top.

### Files modified
- `website/web/views.py` — backend (Django views)
- `website/web/templates/dashboard.html` — frontend (HTML + Chart.js JS)
- `WEBSITE_IMPROVEMENTS_REPORT.md` — new report file at repo root

### Files NOT touched (out-of-scope, as required)
`scheduler/**`, `prediction/**`, `markov/**`, `news/**` scrapers, `cpo/**`, `news.html`, settings, Dockerfile, `requirements.txt`, archives.

---

## Commits (4 new, on `main`)

| SHA | Message |
|-----|---------|
| `2d7a08f` | fix(news): clamp page_number to valid range to prevent empty results on filter change |
| `5cb87a4` | feat(dashboard): add per-horizon expandable cards with Base vs CSA comparison |
| `3057eb2` | feat(dashboard): add 90-day sentiment trend chart from sentiment_aggregates |
| `706bf73` | docs: add website improvements report (pagination clamp, per-horizon cards, sentiment chart) |

(Plus 10 older commits on `cleanup/remove-auth` were fast-forwarded in the same merge — auth removal, diagram doc rewrite, multi-horizon plotting feature. These existed on the feature branch before this session.)

---

## Improvement 1 — News Pagination Clamp
**File:** `website/web/views.py` (`news()` function, +1 line)

Added one line after `total_pages` is computed:
```python
page_number = max(1, min(page_number, total_pages))
```
Visiting `/news/?page=999&sentiment=Positive` now resolves to the last valid page (verified: clamps to "Page 193 of 193") instead of returning an empty grid. No template change needed — `pagination` dict already consumes the clamped value.

---

## Improvement 2 — Per-Horizon Expandable Cards

### Backend (`website/web/views.py`)
- Replaced the old "h=1 only" metrics block (lines 101–124) with a loop fetching all 14 prediction docs (`xgboost_{base|csa}_Daily_h{1..7}`) via individual `.get()` calls — no composite index needed.
- Builds a `horizon_data` list of 7 entries, each shaped:
  ```python
  {'horizon': h, 'best_variant': 'base'|'csa'|None,
   'best_mape': float|None,
   'base': {mape, rmse, r2, da, predicted_price, predicted_date} | None,
   'csa':  {...} | None}
  ```
- Backward-compatible `metrics` dict (drives existing 4-card header row) is still derived — now from h=1's best variant.
- Added `'horizon_data': json.dumps(horizon_data)` to render context.
- Updated `_empty_dashboard_ctx()` to include `'horizon_data': json.dumps([])`.

### Frontend (`website/web/templates/dashboard.html`)
- Inserted new card "Per-Horizon Model Performance" **after** the 4-metric grid, **before** the Main Content closing div.
- Appended ~95 lines of JS in `extra_js`:
  - `renderHorizonCards()` — 7 buttons (H1–H7) with badge (BASE/CSA, teal for CSA, slate for BASE) + best MAPE.
  - `renderHorizonDetail(h)` — toggleable Base vs CSA comparison table; better values bolded green (lower-is-better for MAPE/RMSE, higher-is-better for R²/DA).
  - `toggleHorizonDetail(h)` — collapse/expand handler.

---

## Improvement 3 — 90-Day Sentiment Trend Chart

### Backend (`website/web/views.py`)
- Inserted between the horizon block and the `return render(...)` call:
  ```python
  db.collection('sentiment_aggregates')
    .where('date', '>=', three_months_ago)
    .order_by('date')
    .stream()
  ```
  Filters `frequency == 'Daily'` in Python.
- Each entry: `{date, positive_prob, negative_prob, neutral_prob, sentiment_score}`.
- Reuses the already-defined `three_months_ago` (an isoformat string at line 27).
- Added `'sentiment_data': json.dumps(sentiment_list)` to render context.
- Updated `_empty_dashboard_ctx()` to include `'sentiment_data': json.dumps([])`.

### Frontend (`website/web/templates/dashboard.html`)
- New card "Sentiment Trend (90d)" inserted after the per-horizon cards block.
- 200px-tall canvas `<canvas id="sentimentChart">`.
- Chart.js line chart with three datasets (Positive=green, Negative=red, Neutral=slate), y-axis fixed to [0, 1], 4-decimal tooltip formatting.
- Includes `annotation: { annotations: {} }` because `chartjs-plugin-annotation` is registered globally in `base.html`.

---

## Sanity Tests — all PASS

| Test | Result |
|------|--------|
| `python manage.py check` | No issues |
| `runserver 8000` boot | Clean, no tracebacks |
| `GET /` | 200 (renders new sections — `Per-Horizon Model Performance`, `horizon-cards`, `Sentiment Trend`, `sentimentChart` markers all present) |
| `GET /news/?page=999&sentiment=Positive` | 200 — clamped to "Page 193 of 193" (NOT empty) ✅ verifies the fix |
| `GET /news/` | 200 |
| `GET /about/` | 200 |
| `from web import views` | imports OK |
| `get_template('dashboard.html')` parse | OK |

**Environment note:** System `python` resolves to a Microsoft Store stub on this Windows machine; tests ran via `website/venv/Scripts/python.exe`. No code change needed.

---

## Final git state

```
main (HEAD)        — pushed to origin/main
cleanup/remove-auth — same SHA as main minus the report commit
working tree       — clean
```

**Last 5 commits on `main`:**
```
706bf73 docs: add website improvements report (pagination clamp, per-horizon cards, sentiment chart)
3057eb2 feat(dashboard): add 90-day sentiment trend chart from sentiment_aggregates
5cb87a4 feat(dashboard): add per-horizon expandable cards with Base vs CSA comparison
2d7a08f fix(news): clamp page_number to valid range to prevent empty results on filter change
69fc530 docs(diagrams): rewrite use-case, activity, sequence for current website
```

---

## Things still requiring manual verification (UI)

1. Visit `http://localhost:8000/`:
   - 4 metric cards (top row) — unchanged layout; values now derived from h=1 best variant.
   - **NEW:** "Per-Horizon Model Performance" — 7 horizon cards H1–H7 with BASE/CSA badge + best MAPE.
   - Click H3 → expandable detail table renders; better values highlighted green.
   - Click H3 again or "Collapse ×" → panel hides.
   - **NEW:** "Sentiment Trend (90d)" line chart with Positive/Negative/Neutral.
2. Existing sidebar prediction panel ("Run Prediction") and price chart should be **unchanged** — Improvement 2 is an independent main-content section.
3. `/news/?page=999&sentiment=Positive` → lands on the last valid page (HTTP 200, non-empty).

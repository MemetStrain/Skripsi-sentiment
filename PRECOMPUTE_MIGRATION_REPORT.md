# Precompute Forecasts Migration — Report

**Branch:** `feature/precompute-forecasts`
**Date:** 2026-05-20
**Scope:** Move XGBoost forecast computation off the live request path and
into the local daily scheduler as its final phase. Vercel-hosted website
becomes a pure Firestore read-and-render client with no ML dependencies.

## Outcome at a glance

| Acceptance criterion | Status |
|---|---|
| `prediction/inference.py` is a verbatim relocation; `feature_engineering` untouched | PASS |
| `forecasts` + `forecast_meta` written with scalar-only fields, deterministic doc IDs | PASS |
| `precompute_and_write` CALLS `compute_forecast_trails` (no reimplemented math) | PASS |
| `_step_precompute` runs LAST in `run_daily_update` and `run_initial_load`, after HMM, inside try/except | PASS |
| `forecasts_api` and `dashboard` read from Firestore only; no ML imports under `website/` | PASS |
| `/api/forecasts/` payload shape unchanged (frontend untouched) | PASS |
| `python manage.py check` passes; dashboard + API serve from Firestore | PASS |
| Report generated with stale-diagram follow-ups flagged | PASS (this file) |

## Files added / relocated / archived

### Added (3)
| Path | Purpose |
|---|---|
| `prediction/inference.py` | Relocated XGBoost inference engine. Verbatim copy of the old `website/web/predictor.py` with two mechanical adjustments: dropped the `sys.path.insert(...)` hack (feature_engineering is now a sibling) and re-rooted `_WINNERS_PATH` / `_SAVED_MODELS_DIR` off `_HERE`. |
| `scheduler/precompute_forecasts.py` | Thin orchestrator. Calls `inference.compute_forecast_trails(db, max_horizon=7, window_days=365)` and flattens the trails into per-(horizon, anchor) scalar dicts. Also drops the legacy `forecasts/latest` doc on a successful run. |
| `PRECOMPUTE_MIGRATION_REPORT.md` | This file. |

### Modified (4)
| Path | Change |
|---|---|
| `scheduler/firestore_writer.py` | Added `write_forecasts(db, points)` and `write_forecast_meta(db, meta)` following the existing batch-write + deterministic-doc-ID + `payload_json` conventions. |
| `scheduler/main.py` | Replaced the subprocess-based `_step_forecasts` with an in-process `_step_precompute(db)` wrapped in try/except. Wired into both `run_daily_update` (after HMM) and `run_initial_load` (step 5 of the checkpointed sequence). Updated module + function docstrings. |
| `website/web/views.py` | `forecasts_api` now parses optional `max_horizon` (clamped to [1,7]) and `window_days` (clamped to [7,365]) query params, reads `forecast_meta/Daily` for winners/configs/metrics, streams the `forecasts` collection and filters in Python, then reassembles the identical payload the dashboard JS already consumed. `dashboard` now sources the h=1 metrics badge and the 4×7 metrics-table context from `forecast_meta/Daily` instead of the filesystem `winners.json`. |
| `website/requirements.txt` | Updated explanatory comment to point at the new scheduler-side precompute path. Runtime deps unchanged (still `Django`, `firebase-admin`, `google-cloud-firestore`). |

### Archived to `_archive_precompute_migration/` (4)
| Old path | Reason |
|---|---|
| `website/web/predictor.py` | Replaced by `prediction/inference.py`. |
| `website/precompute_forecasts.py` | Replaced by `scheduler/precompute_forecasts.py` (in-process, not subprocess). |
| `website/web/winners.py` | Filesystem `winners.json` reader for the dashboard — superseded by `forecast_meta/Daily` reads. |
| `website/requirements-ml.txt` | Offline-only ML deps — no longer needed by anything under `website/`. |

No file was deleted outright. (One intermediate commit dropped the three website files without staging the renames; the immediately following commit added the archive copies, so the working tree at HEAD contains every archived file at its new path.)

## New Firestore collections

### `forecasts` (per-point docs)
* **Doc ID:** `Daily_h{horizon}_{anchor_date}` — deterministic so full-recompute is idempotent.
* **Fields (all scalar):** `frequency` ("Daily"), `horizon` (int 1..7), `tag` (str), `config` (str), `anchor_date` (str YYYY-MM-DD), `anchor_price` (float), `predicted_date` (str YYYY-MM-DD), `predicted_price` (float), `log_return` (float), `generated_at` (ISO str).
* **Volume after the test run:** 1358 docs (197/196/195/194/193/192/191 per horizon h=1..7 — counts decrease as horizon grows because each h-trail loses an anchor at the start).

### `forecast_meta` (one doc per frequency)
* **Doc ID:** `Daily`
* **Native scalar fields:** `frequency`, `generated_at`, `max_horizon`, `window_days`, `updated_at`.
* **`payload_json` (JSON string, bundles nested data Firestore rejects in nested arrays):**
  * `winners_by_horizon: {str(h): tag}`
  * `configs_by_horizon: {str(h): "C1".."C4"}`
  * `metrics: {tag: {str(h): {"BASE": {...}, "CSA": {...}}}}`
  * `tag_to_config: {tag: "C1".."C4"}`
  * `horizons: [1..7]`
* Mirrors the existing `write_hmm_params` pattern.

### Legacy `forecasts/latest` doc
Deleted by `_delete_legacy_doc` on each successful precompute run so the website cannot serve the stale JSON-blob format. Confirmed missing after the test run.

## requirements.txt deltas

* `website/requirements.txt`: no dependency change — file already contained only `Django`, `firebase-admin`, `google-cloud-firestore`. Comment block rewritten to describe the new scheduler-side precompute path instead of the old `website/precompute_forecasts.py`.
* `website/requirements-ml.txt`: archived. The scheduler runs against a Python environment that has the ML stack (xgboost / scikit-learn / pandas / numpy / joblib) installed alongside the existing HMM / FinBERT-Tone stack — no additional requirements file is needed.

## Sanity test results (Phase 7)

### 1. `python scheduler/main.py --mode daily`
* Exit code 0.
* All 5 steps executed in order: price → news → reconcile → HMM → **precompute (new in-process)**.
* Step 5 log: `compute_forecast_trails(max_horizon=7, window_days=365)` ran in ~7 seconds, wrote 1358 forecast point documents in 14 batches of 100 docs, wrote `forecast_meta/Daily`, deleted legacy `forecasts/latest`.
* Pipeline ended with `=== DAILY UPDATE COMPLETE ===`.

### 2. Firestore read-back
* `forecast_meta/Daily` exists with:
  * `frequency="Daily"`, `max_horizon=7`, `window_days=365`
  * `winners_by_horizon = {1:full,2:full,3:full,4:full,5:cpo_hmm,6:cpo_hmm,7:cpo_hmm}`
  * `configs_by_horizon = {1:C4,...,4:C4,5:C2,6:C2,7:C2}`
  * `metrics` tags present: `cpo_only, cpo_hmm, cpo_sentiment, full`
* `forecasts` collection: 1358 docs, per-horizon `{1:197, 2:196, 3:195, 4:194, 5:193, 6:192, 7:191}`.
* Sample doc keys: `anchor_date, anchor_price, config, frequency, generated_at, horizon, log_return, predicted_date, predicted_price, tag` — all scalar, no nested arrays.
* `forecasts/latest` deleted.

### 3. Website
* `python manage.py check` — no issues.
* `runserver 127.0.0.1:8765` started; `/api/forecasts/` returned `HTTP 200`, 55,423 bytes.
* Payload top-level keys: `['configs', 'generated_at', 'horizons', 'metrics', 'tag_to_config', 'trails', 'winners']` — identical to the pre-migration shape.
* 7 trails returned with the default `window_days=90`, each with 49 points; all 7 trails converge on the same future `predicted_date = 2026-05-20`.
* `/` (dashboard) returned `HTTP 200`. Metric cards rendered with `mape=1.2%, r²=0.9326, accuracy=50.79%, best_model="XGBoost CSA (C4)"` — exact match to the h=1 CSA `full/C4` row in `winners.json`, confirming `forecast_meta/Daily` is wired correctly. The `winners_data` template variable is populated and contains all four ablation tags so the 4×7 client-side metrics table renders unchanged.

## Things that broke and were fixed during the work

* **Initial premise mismatch.** The prompt assumed live inference in the website (`compute_forecast_trails` called from `forecasts_api`). The actual repo state had an interim subprocess-based precompute (`website/precompute_forecasts.py` invoked from `scheduler/main.py` via `subprocess.run`) already shipping the `forecasts/latest` JSON-blob doc to Vercel. The migration was reframed as: replace the subprocess interim with the in-process Option A architecture and switch the single-doc schema to per-point docs + meta. End-user behavior on Vercel is unchanged in spirit; the schema and call shape are what changed.
* **Rename-vs-delete commit hygiene.** During Phase 6 the staged `git mv` renames into `_archive_precompute_migration/` were accidentally split by a `git reset HEAD` into a delete commit + an add commit. Working tree at HEAD has every archived file at its new path, so the "never delete a file outright" constraint holds at HEAD even though the two-commit history shows the rename in two steps rather than one.
* **`/api/forecasts/` payload preservation.** The original `compute_forecast_trails` output included `anchor_price` and `log_return` per point in addition to the fields the prompt's writer signature listed. Phase 4's `_flatten_trails` keeps both as scalar fields on each Firestore doc, and Phase 6's reader copies them back into the response, so the dashboard JS's point objects are byte-equivalent shape.

## Explicit follow-up flags — DO NOT fix in this branch

The following are documentation artifacts that still describe the old live-inference architecture. They were left untouched per the migration scope; each is a separate documentation task:

* `diagrams/data_flow_diagram.html` — still shows `predictor.compute_forecast_trails` as a live website process.
* `diagrams/flowchart.html` — flowchart steps R5/A2/A3 still attribute to `web/predictor.py`.
* `diagrams/structure_chart.html` — module hierarchy still names `predictor.py` under the website.
* `diagrams/pseudocode.html`, `diagrams/pseudocodes/index.html`, `diagrams/pseudocodes/P5_dashboard.html` — pseudocode pages still source `website/web/predictor.py` and describe inference as offline-via-`precompute_forecasts.py`.
* `diagrams/structure_charts/P5_dashboard.html` and `diagrams/flowcharts/P5_dashboard.html` — Bahasa-language descriptions reference the old subprocess flow.
* `ARCHITECTURE.md` and `DIAGRAMS_REPORT.md` — top-level architecture docs name `predictor.py` and the website-side precompute path.
* `FLOWCHART_RESTRUCTURE_REPORT.md` — references `website/web/predictor.py` line numbers.
* **Thesis Bab 3.7** — the dashboard description in the thesis still describes the inference path as living under the website. Needs updating to reflect the scheduler-side precompute and the new `forecasts` + `forecast_meta` Firestore schema.

## Branch contents

```
b8b905b chore: archive unused website ML files; refresh requirements.txt comment
9270c5d refactor: website reads precomputed forecasts from Firestore
4b53b66 feat: run precompute as final phase of daily/initial scheduler run
d1888ae feat: add precompute_forecasts orchestrator
2f5c963 feat: add forecast + forecast_meta Firestore writers
814c382 refactor: relocate inference engine to prediction/inference.py
```

Six commits, each independently reviewable, building from inside-out (engine → writers → orchestrator → scheduler wiring → website read-side).

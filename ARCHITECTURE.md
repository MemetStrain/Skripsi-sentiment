# ARCHITECTURE — CPO Price Prediction (post-2026-04-26)

This document captures the **current** architecture after the
thesis-scope-reduction sweep recorded in [CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md).
For history of what was removed, see that inventory.

## Scope

- **Single model:** XGBoost. Random Forest / ARIMAX / SARIMAX dropped.
- **Hyperparameter optimisation:** Crow Search Algorithm (CSA) only.
  Bayesian optimisation dropped.
- **Forecast horizons:** t+1 … t+7 (Daily only — Weekly / Monthly variants dropped).
- **Ablation study (offline):** four feature configurations for the thesis
  comparison —
    - **C1** lagged price only
    - **C2** lagged price + HMM market state
    - **C3** lagged price + news sentiment
    - **C4** lagged price + HMM + sentiment (full)
- **Naive baseline (control experiment for H4):** random walk, historical
  mean, seasonal naive — compared against the best parametric model with
  Diebold-Mariano.
- **Sentiment model:** FinBERT (English).
- **Regime detection:** Gaussian HMM with 3 states (Bullish / Neutral /
  Bearish), `N_STATES_RANGE = range(2, 5)` for BIC selection.
- **Database:** Google Cloud Firestore for all data.
- **Deployment:** Vercel for the public-facing read-only Django frontend
  (no authentication — three pages: Dashboard, News, About). The ML
  scheduler is a **local Python script** run ad-hoc by the maintainer;
  it is no longer hosted on Cloud Run.
- **Prediction inference:** runs as the **final phase of the local
  scheduler** (`scheduler/precompute_forecasts.py` calls
  `prediction/inference.py::compute_forecast_trails` in-process using
  the offline-trained CSA model artefacts under
  `prediction/saved_models/`). Results are written to Firestore as a
  per-(horizon, anchor) `forecasts` collection plus a
  `forecast_meta/Daily` summary doc. The website does **not** import any
  ML libraries — it only reads those documents at request time.

## Pipeline overview

```
Local CSVs (source of truth)   ──┐
  cpo/Data_CPO_Daily.csv         │
  news/mpob_news_with_sentiment_tone.csv
                                 │
                  scheduler/ (local, run ad-hoc)
                                 │
   1. price_fetcher.py    →  append cpo CSV
   2. news_extractor.py   →  scrape + preprocess + FinBERT-Tone
                          →  append 3 news CSVs
   3. reconcile.py        →  diff CSVs vs Firestore, write missing/stale
                              prices, news_articles, sentiment_aggregates
   4. hmm_updater.py      →  hmm_states  (3-state Gaussian HMM)
   5. precompute_forecasts.py → prediction/inference.py
                              →  per-(horizon, anchor) docs in `forecasts`
                              +  winners + metrics in `forecast_meta/Daily`
                                 │
                                 ▼
              website/ (Django app on Vercel — read-only)
                  ├── Dashboard reads daily_prices + hmm_states
                  │   + sentiment_aggregates + forecast_meta/Daily.
                  └── /api/forecasts/ reads forecast_meta/Daily + streams
                      the `forecasts` collection, filters in Python, and
                      reassembles the same payload the JS already consumed.
                      No ML imports under website/.
```

## Module layout

```
cpo/
  fetch_cpo_data.py              Daily Investing.com fetcher
  preprocess_cpo_variables.py    Builds cpo/output/cpo_variables_Daily.csv
  Data_CPO_Daily.csv             Source CSV used for offline runs

news/
  scrap_fast.py                  MPOB scraper
  news_preprocessing.py          Cleans the raw scrape
  finbert_sentiment_analysis_flexible.py  Active FinBERT scorer
  finbert_sentiment_analysis.py           Older variant (kept for reference)
  finbert_tone_sentiment_analysis.py      Tone variant (kept for reference)
  check_cuda.py                  Debug utility
  mpob_news_*.csv                Source artifacts

markov/
  cpo_hmm_states.py              Offline HMM trainer (Daily-only)

prediction/
  horizon_forecast_C1_price_only.py     C1 ablation (offline training)
  horizon_forecast_C2_price_hmm.py      C2 ablation
  horizon_forecast_C3_price_sentiment.py C3 ablation
  horizon_forecast_C4_full.py           C4 ablation
  crow_search_optimizer.py              CSA implementation
  naive_baseline.py                     Naive predictors (H4 control)
  baselines/                            DM comparison + integration runner
  feature_engineering.py                Single source of truth for features
                                         (shared by training + inference)
  inference.py                          XGBoost forecast engine used by the
                                         scheduler's precompute phase
  utils/forecast_utils.py               Shared utilities (XGBoost-only)
  saved_models/                         Cached `xgboost_{base,csa}` artifacts
  winners.json                          Auto-picked winning config per horizon
  output_horizons*/                     Per-ablation prediction outputs
  output_validation/                    Ablation validation summary

scheduler/                       Local ad-hoc data pipeline
  main.py                        Entry: --mode initial | daily
                                 (5 phases: price → news → reconcile →
                                  HMM → forecast precompute)
  hmm_updater.py                 Daily HMM state refit
  sentiment_runner.py            FinBERT-Tone on new articles only
  reconcile.py                   Diff full CSVs vs Firestore, mirror gaps
  precompute_forecasts.py        Final phase — calls prediction/inference
                                 in-process, writes forecasts/* +
                                 forecast_meta/Daily
  news_extractor.py              Scrape + preprocess bridge to news/
  price_fetcher.py               Investing.com fetcher + trading-day helper
  firestore_writer.py            Mirror writes (CSV is source of truth)
  local_csv_writer.py            CSV append-with-dedup helpers
  initial_load_progress.json     Checkpoint file (currently {})

website/                         Vercel-hosted public read-only Django app
  config/                        Django settings
  web/views.py, templates/       Dashboard / News / About
  web/tasks.py                   Stubs for future scheduler triggers

revision/CPO_COUNCIL_VERDICT_…   Most recent thesis-council verdict.
_archive_before_cleanup/         Everything removed by the 2026-04-26 sweep.
```

## Firestore collections (web-facing)

| Collection            | Doc ID                                  | Notes |
|-----------------------|-----------------------------------------|-------|
| `users`               | UID                                     | Legacy — historical login data, no longer read or written. Site is now public-facing read-only. |
| `daily_prices`        | `YYYY-MM-DD`                            | OHLCV. |
| `hmm_states`          | `{frequency}_{YYYY-MM-DD}`              | Currently `Daily_…` only. |
| `hmm_models`          | `{frequency}` (e.g. `Daily`)            | Frozen HMM parameters. Nested arrays bundled in `payload_json`. |
| `news_articles`       | md5(url)                                | Article metadata + FinBERT-Tone sentiment. |
| `sentiment_aggregates`| `{frequency}_{YYYY-MM-DD}`              | Daily aggregate. |
| `forecasts`           | `{frequency}_h{horizon}_{anchor_date}`  | One scalar-only doc per (horizon, anchor). Written by `scheduler/precompute_forecasts.py` (~1.4k docs/run for a 365-day × 7-horizon window). Deterministic doc IDs → idempotent full-recompute. |
| `forecast_meta`       | `{frequency}` (e.g. `Daily`)            | One summary doc per frequency. Native scalars + `payload_json` bundling `winners_by_horizon`, `configs_by_horizon`, `metrics`, `tag_to_config`, `horizons`. Read by both the dashboard view (h=1 metrics badge + 4×7 metrics table) and `forecasts_api`. |

The legacy single-doc `forecasts/latest` JSON-blob store and the
filesystem `web/winners.py` reader were removed during the precompute
migration. The `predictions` and `HorizonModelParameters` collections
remain dropped — inference is now precomputed by the scheduler using
offline-trained model artefacts. CSA hyperparameters are baked into the
saved model objects.

## How to run

### Offline ablation training (researcher workflow)

Training is staged so CSA optimisation is only spent on the configs the
dashboard will actually serve (one per horizon):

```bash
# Phase A — base-only training across all 4 ablations × 7 horizons.
python prediction/horizon_forecast_C1_price_only.py     --interval daily --no-csa
python prediction/horizon_forecast_C2_price_hmm.py      --interval daily --no-csa
python prediction/horizon_forecast_C3_price_sentiment.py --interval daily --no-csa
python prediction/horizon_forecast_C4_full.py            --interval daily --no-csa

# Phase B — pick the lowest-base-MAPE config per horizon → winners.json.
python prediction/compute_winners.py

# Phase C — CSA-optimise only the winning (tag, horizon) pairs (7 runs).
python prediction/train_winners_csa.py

# Phase D — refresh the metrics matrix to include the new CSA cells.
python prediction/compute_winners.py

# Naive baseline + Diebold-Mariano (H4 control experiment)
python prediction/baselines/run_naive_integration.py
```

Outputs land in `prediction/output_horizons/{tag}/Daily/horizon_*/`,
and saved model artefacts in
`prediction/saved_models/{tag}/Daily/h{horizon}/xgboost_{base,csa}/`.

### Daily scheduler (local, ad-hoc)

```bash
# from project root — credentials picked up from website/firebase-credentials.json
python scheduler/main.py --mode daily
```

`--mode initial` does the historical bootstrap from local CSVs into
Firestore; `--reset-progress` clears the checkpoint file before re-running.

### Web app (local development)

```bash
cd website
python manage.py runserver
```

Public-facing read-only system; no authentication required. All data
reads go directly through Firestore in `web/views.py` — no Django ORM,
no SQLite.

## What changed in 2026-04-26 sweep

See [CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md) and
[CLEANUP_REPORT.md](CLEANUP_REPORT.md) (forthcoming).

## What changed in 2026-05-05 auth-removal sweep

The custom Firestore-backed login flow (`auth_backend.py`, login.html,
register.html, login/logout/register URLs and views) was removed. The
site is now a public-facing read-only dashboard with three pages —
Dashboard, News, About. See
[AUTH_REMOVAL_INVENTORY.md](AUTH_REMOVAL_INVENTORY.md) and
[AUTH_REMOVAL_REPORT.md](AUTH_REMOVAL_REPORT.md). All archived files
live under `_archive_auth_removal/`.

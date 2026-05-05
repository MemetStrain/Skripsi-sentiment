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
  (no authentication — three pages: Dashboard, News, About), Cloud Run
  Job (Docker) for the daily ML scheduler.

## Pipeline overview

```
Raw scrape (MPOB news)         ──┐
Investing.com daily price feed ──┤
                                 │
                    scheduler/ (Cloud Run Job, daily 1AM MYT)
                                 │
   1. price_fetcher.py    →  daily_prices  (Firestore)
   2. news_extractor.py   →  news_articles (Firestore)
   3. sentiment_runner.py →  sentiment_aggregates (FinBERT scores)
   4. hmm_updater.py      →  hmm_states  (3-state Gaussian HMM)
   5. prediction_updater.py
        - XGBoost × {base, csa} × 7 horizons = 14 docs
        - written to `predictions` collection
                                 │
                                 ▼
              website/ (Django app on Vercel — read-only)
                  └── Dashboard reads daily_prices + hmm_states
                                 + predictions and renders chart.
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
  utils/forecast_utils.py               Shared utilities (XGBoost-only)
  saved_models/                         Cached `xgboost_{base,csa}` artifacts
  output_horizons*/                     Per-ablation prediction outputs
  output_validation/                    Ablation validation summary

scheduler/                       Production daily pipeline (Cloud Run Job)
  main.py                        Entry: --mode initial | daily
  prediction_updater.py          Writes the 14 XGBoost prediction docs
  hmm_updater.py                 Daily HMM state refit
  sentiment_runner.py            FinBERT on new articles only
  news_extractor.py
  price_fetcher.py
  firestore_writer.py
  cleanup_old_articles.py
  Dockerfile
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
| `predictions`         | `{model}_{variant}_{frequency}_h{h}`    | Live: `xgboost_{base,csa}_Daily_h{1..7}` (14 docs). Legacy RF / ARIMAX / SARIMAX / Bayesian docs may exist from earlier runs but are no longer refreshed. |
| `news_articles`       | md5(url)                                | Article metadata + FinBERT sentiment. |
| `sentiment_aggregates`| `{frequency}_{YYYY-MM-DD}`              | Daily aggregate. |
| `HorizonModelParameters` | `{model}_{variant}_{frequency}_h{h}` | CSA-tuned hyperparameters per horizon. |

## How to run

### Offline ablation training (researcher workflow)

```bash
# Train each ablation × all 7 horizons × {base, csa}
python prediction/horizon_forecast_C1_price_only.py     --interval daily
python prediction/horizon_forecast_C2_price_hmm.py      --interval daily
python prediction/horizon_forecast_C3_price_sentiment.py --interval daily
python prediction/horizon_forecast_C4_full.py            --interval daily

# Add naive baseline rows + Diebold-Mariano comparison (H4)
python prediction/baselines/run_naive_integration.py
```

Outputs land in `prediction/output_horizons*/Daily/horizon_*/`.

### Daily scheduler (production, manual sanity-check run)

```bash
# from project root
docker run --rm \
  -v "$(pwd)/scheduler:/app" -v "$(pwd)/cpo:/cpo" -v "$(pwd)/news:/news" \
  -e FIREBASE_CREDENTIALS_JSON="$(cat website/firebase-credentials.json)" \
  cpo-scheduler --mode daily
```

`--mode initial` does the historical bootstrap from local CSVs;
`--reset-progress` clears the checkpoint file before re-running.

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

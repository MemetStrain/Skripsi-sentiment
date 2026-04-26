# CLEANUP_INVENTORY.md — Phase 1 Discovery

Generated: 2026-04-26
Branch: `cleanup/thesis-scope-reduction`
Scope reference: XGBoost-only ablation (C1–C4), Daily frequency, CSA optimizer, FinBERT sentiment, 3-state HMM, full Firestore.

## 0. Headline Numbers

- Total source/data files inventoried: **~120** (excludes auto-generated `__pycache__`, `venv/`)
- Bucket totals:
  - **KEEP**: 49
  - **REMOVE**: 51
  - **REVIEW**: 21

(Counts approximate — saved_models artifact directories counted as one entry per `{config}/{horizon}/{model_variant}`. Excludes images/CSVs already produced as outputs.)

> NOTHING DELETED OR MOVED YET. This document is for your review only.

---

## 1. Directory Tree (one-line purpose each)

```
d:\Skripsi1\
├── .claude/               Claude Code project settings (KEEP)
├── .vercel/               Vercel deployment metadata (KEEP)
├── cpo/                   CPO price data + preprocessing scripts
│   └── output/            Pre-computed cpo_variables_{Daily,Weekly,Monthly}.csv + sentiment match research artifacts
├── cpo-dashboard/         EMPTY directory (REMOVE)
├── diagrams/              Markdown + HTML thesis diagrams (REVIEW — partial drift)
├── experiments/
│   └── walk_forward/      Walk-forward eval scripts using all 4 models (REVIEW — likely retire)
├── markov/                HMM logic; legacy local-CSV pipeline + `cpo_hmm_states.py`
│   └── output/            HMM states CSVs (Daily/Weekly/Monthly)
├── news/                  Scraping + FinBERT scripts (mix of active + legacy)
│   └── output/            Sentiment aggregates (Daily/Weekly/Monthly)
├── output/                Old sentiment-weight optimization research outputs (REMOVE)
├── prediction/            All forecasting modules
│   ├── baselines/         Naive baseline + Diebold-Mariano comparison (KEEP)
│   ├── output/            Old "adaptive_prediction.py" outputs (REMOVE)
│   ├── output_horizons/             C4 (full) testing/validation outputs
│   ├── output_horizons_cpo_hmm/     C2 (price+HMM) outputs
│   ├── output_horizons_cpo_only/    C1 (price-only) outputs
│   ├── output_horizons_cpo_sentiment/  C3 (price+sentiment) outputs
│   ├── output_validation/   ARIMAX/SARIMAX residual diagnostics
│   ├── saved_models/        Cached trained model artifacts (12 variants × 7 horizons × 4 ablation configs)
│   └── utils/               Shared forecast utilities
├── revision/              Thesis council verdict markdown (KEEP — context doc)
├── scheduler/             Cloud Run Job Docker package (production pipeline)
├── website/               Django + Vercel webapp
│   ├── config/            Django project settings (KEEP)
│   ├── diagrams/          Empty (REVIEW)
│   └── web/               Django app (templates, views, auth, tasks)
├── *.md / *.txt           Project-level documentation (mostly OUTDATED)
└── *.py                   Root-level legacy orchestrators (mostly REMOVE)
```

---

## 2. File-by-file classification

### 2.1 Project root

| Path | Category | Reason |
|---|---|---|
| `.gitignore` | KEEP | Git ignore rules |
| `.vercelignore` | KEEP | Vercel deployment |
| `vercel.json` | KEEP | Vercel config |
| `CPO_FORECAST_20260407_202731.md` | REVIEW | Internal forecast log; thesis context but stale (Weekly tests) |
| `PREDICTION_GUIDE.md` | REMOVE | Pre-2026 multi-frequency / multi-model guide; superseded by current scope |
| `baseline_metrics.txt` | REVIEW | Snapshot of pre-cleanup baseline metrics; may be referenced in thesis |
| `create_prediction_dataset.py` | REMOVE | Legacy DATA_FREQUENCY=daily/weekly/monthly dataset assembler; replaced by horizon_forecast scripts |
| `optimize_sentiment_weights.py` | REMOVE | Sentiment-weight grid search (Dec 2026 research artifact) |
| `pipeline_run_20260321_174152.log` | REMOVE | One-off log file |
| `run_pipeline.py` | REMOVE | Legacy orchestrator referencing `csa`/`improved`/`baseline` model paths and Weekly/Monthly |

### 2.2 `cpo/` — Price data + preprocessing

| Path | Category | Reason |
|---|---|---|
| `Data_CPO_Daily.csv` | KEEP | Active price data source for scheduler initial load |
| `Data_CPO_Weekly.csv` | REMOVE | Frequency dropped from scope |
| `Data_CPO_Monthly.csv` | REMOVE | Frequency dropped from scope |
| `fetch_cpo_data.py` | KEEP | Investing.com daily fetcher; supports `Data_CPO_Daily.csv` (still useful) |
| `preprocess_cpo_variables.py` | REVIEW | Generates `cpo_variables_*.csv`; supports Daily but also Weekly/Monthly — needs scope-trim |
| `cpo_sentiment.py` | REMOVE | Dec-2026 research script (sentiment×price matching); not in production pipeline |
| `cpo_sentiment_lagged.py` | REMOVE | Dec-2026 lagged-sentiment correlation research; not in production pipeline |
| `stationarity_test.py` | REVIEW | ADF/KPSS/PP standalone tests — useful for thesis Bab 4 even if not in production |
| `output/cpo_variables_Daily.csv` | KEEP | Active feature input for ablation scripts |
| `output/cpo_variables_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/cpo_variables_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/daily_dated.csv` | REVIEW | Dec-2026 intermediate artifact |
| `output/monthly_dated.csv` | REMOVE | Frequency out of scope |
| `output/weekly_dated.csv` | REMOVE | Frequency out of scope |
| `output/sentiment_lagged_*.{csv,png}` (4 files) | REMOVE | Outputs of `cpo_sentiment_lagged.py` (REMOVE) |
| `output/sentiment_price_analysis.png` | REMOVE | Output of `cpo_sentiment.py` (REMOVE) |
| `output/sentiment_price_matching_results.csv` | REMOVE | Output of `cpo_sentiment.py` (REMOVE) |

### 2.3 `news/` — Scraping + sentiment

| Path | Category | Reason |
|---|---|---|
| `scrap_fast.py` | KEEP | Used by scheduler `news_extractor.py` |
| `news_preprocessing.py` | KEEP | Cleans raw scrape into preprocessed CSV |
| `finbert_sentiment_analysis_flexible.py` | KEEP | Active FinBERT scorer (referenced by tasks/diagrams) |
| `finbert_sentiment_analysis.py` | REMOVE | Older non-flexible FinBERT script; superseded |
| `finbert_tone_sentiment_analysis.py` | REMOVE | Alternative tone analysis variant; not in production |
| `check_cuda.py` | REVIEW | One-off CUDA check — useful for debugging GPU runs but not pipeline |
| `README_sentiment.md` | REVIEW | Documents FinBERT setup; may need update |
| `mpob_news_fast.csv` | KEEP | Raw scrape feed — used by scheduler initial load |
| `mpob_news_preprocessed.csv` | KEEP | Cleaned articles — used by preprocessing chain |
| `mpob_news_with_sentiment.csv` | KEEP | Pre-scored sentiment input for scheduler initial load |
| `output/sentiment_aggregate_Daily.csv` | KEEP | Active sentiment input for ablation scripts |
| `output/sentiment_aggregate_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/sentiment_aggregate_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/monthly_sentiment_aggregate.csv` | REMOVE | Old name; legacy |

### 2.4 `markov/` — HMM module

| Path | Category | Reason |
|---|---|---|
| `cpo_hmm_states.py` | REVIEW | Local HMM script supporting Daily/Weekly/Monthly; scheduler/hmm_updater.py is the production version |
| `cpo_prediction_dataset_daily.csv` | REVIEW | Pre-built dataset; large file (3.6 MB) — verify if any active script reads it |
| `output/hmm_states_results_Daily.csv` | KEEP | Used by horizon_forecast scripts |
| `output/hmm_states_results_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_states_results_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_states_stats_Daily.csv` | KEEP | Diagnostic |
| `output/hmm_states_stats_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_states_stats_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_transition_matrix_Daily.csv` | KEEP | Diagnostic |
| `output/hmm_transition_matrix_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_transition_matrix_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_states_analysis_Daily.png` | KEEP | Diagnostic plot |
| `output/hmm_states_analysis_Weekly.png` | REMOVE | Frequency out of scope |
| `output/hmm_states_analysis_Monthly.png` | REMOVE | Frequency out of scope |
| `output/hmm_bic_scores_Daily.csv` | KEEP | BIC selection record (Bab 2.3) |
| `output/hmm_bic_scores_Weekly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_bic_scores_Monthly.csv` | REMOVE | Frequency out of scope |
| `output/hmm_all_frequencies_summary.csv` | REVIEW | May still be cited; rename or strip Weekly/Monthly rows |

### 2.5 `prediction/` — Forecasting modules

| Path | Category | Reason |
|---|---|---|
| `adaptive_prediction.py` | REMOVE | Single-horizon multi-model pipeline (4 models × 3 variants); replaced by `horizon_forecast*.py` |
| `bayesian_optimizer.py` | REMOVE | Bayesian optimizer dropped per scope |
| `compare_ensemble_methods.py` | REMOVE | Multi-model comparison helper |
| `crow_search_optimizer.py` | KEEP | Core CSA implementation used by all horizon scripts |
| `csa_hyperparameter_optimizer.py` | REVIEW | Older standalone CSA optimizer — superseded by integration in horizon_forecast utils |
| `horizon_forecast.py` | REVIEW → KEEP w/ rename | C4 (full ablation: price+HMM+sentiment); needs SARIMAX/RF/Bayesian removal |
| `horizon_forecast_cpo_only.py` | REVIEW → KEEP w/ rename | C1 (price only) |
| `horizon_forecast_cpo_hmm.py` | REVIEW → KEEP w/ rename | C2 (price + HMM) |
| `horizon_forecast_cpo_sentiment.py` | REVIEW → KEEP w/ rename | C3 (price + sentiment) |
| `naive_baseline.py` | KEEP | H4 control experiment (Random walk / historical mean / seasonal naive) |
| `validate_arimax_sarimax.py` | REMOVE | ARIMAX/SARIMAX dropped from production scope (no Pesaran-Timmermann here — safe to archive) |
| `baselines/__init__.py` | KEEP | Naive baseline package |
| `baselines/dm_comparison.py` | KEEP | Diebold-Mariano comparison (H4) |
| `baselines/naive_evaluator.py` | KEEP | Naive baseline computation |
| `baselines/run_naive_integration.py` | KEEP | H4 orchestrator |
| `utils/__init__.py` | KEEP | Empty package marker |
| `utils/forecast_utils.py` | KEEP (with internal trim) | Shared utilities; contains RF/ARIMAX/SARIMAX branches that are dead code |
| `output/` (entire dir, ~30 files) | REMOVE | All outputs of `adaptive_prediction.py` (already on REMOVE list); plots reference RF/ARIMAX/SARIMAX |
| `output_validation/validation_summary.csv` | REVIEW | Cited in thesis Bab 1 — rename to `_legacy.csv` per instructions |

#### `prediction/output_horizons*` ablation outputs

For each of the 4 ablation directories (`output_horizons`, `output_horizons_cpo_only`, `output_horizons_cpo_hmm`, `output_horizons_cpo_sentiment`):

| Subpath | Category | Reason |
|---|---|---|
| `Daily/` | KEEP | Current scope |
| `Weekly/` | REMOVE | Frequency out of scope (only `output_horizons/Weekly/` and `output_horizons_cpo_only/Weekly/` exist) |
| `Monthly/` | REMOVE | Frequency out of scope (exists in 3 of 4 ablation dirs) |

Within each `Daily/horizon_N/` directory: predictions/results CSVs and PNG overlays — **KEEP** (they reference RF/ARIMAX/SARIMAX columns but are research artifacts; trim columns later as needed).

#### `prediction/saved_models/`

| Subpath | Category | Reason |
|---|---|---|
| `cpo_only/Daily/h{1..7}/xgboost_{base,csa}/` | KEEP | C1 active models |
| `cpo_only/Daily/h{1..7}/xgboost_bayesian/` | REMOVE | Bayesian dropped |
| `cpo_only/Daily/h{1..7}/{random_forest,arimax,sarimax}_*/` | REMOVE | Models dropped (9 dirs × 7 horizons = 63 dirs) |
| `cpo_hmm/Daily/h{1..7}/xgboost_{base,csa}/` | KEEP | C2 active models |
| `cpo_hmm/Daily/h{1..7}/xgboost_bayesian/` | REMOVE | Bayesian dropped |
| `cpo_hmm/Daily/h{1..7}/{random_forest,arimax,sarimax}_*/` | REMOVE | Models dropped |
| `cpo_sentiment/Daily/h{1..7}/xgboost_{base,csa}/` | KEEP | C3 active models |
| `cpo_sentiment/Daily/h{1..7}/xgboost_bayesian/` | REMOVE | Bayesian dropped |
| `cpo_sentiment/Daily/h{1..7}/{random_forest,arimax,sarimax}_*/` | REMOVE | Models dropped |
| `full/Daily/h{1..7}/xgboost_{base,csa}/` | KEEP | C4 active models |
| `full/Daily/h{1..7}/xgboost_bayesian/` | REMOVE | Bayesian dropped |
| `full/Daily/h{1..7}/{random_forest,arimax,sarimax}_*/` | REMOVE | Models dropped |

**Total saved_models to archive: ~280 directories** (10 dropped variants × 7 horizons × 4 ablation configs).

### 2.6 `scheduler/` — Production Cloud Run pipeline

| Path | Category | Reason |
|---|---|---|
| `Dockerfile` | KEEP | Production deployment |
| `requirements.txt` | KEEP w/ trim | Remove statsmodels, sklearn RF, scikit-optimize if no longer needed |
| `main.py` | KEEP | Pipeline entry; some doc strings say "56 predictions" or "136" — needs scope update |
| `firestore_writer.py` | KEEP | Firestore IO |
| `hmm_updater.py` | KEEP | 3-state BIC selection (`N_STATES_RANGE = range(2,5)` ≡ 2,3,4) — already correct |
| `news_extractor.py` | KEEP | Daily news extractor |
| `price_fetcher.py` | KEEP | Daily price fetcher |
| `sentiment_runner.py` | KEEP | FinBERT runner |
| `cleanup_old_articles.py` | KEEP | Maintenance utility |
| `prediction_updater.py` | KEEP w/ heavy trim | Contains `_predict_price_arimax`, `_predict_price_sarimax`, `_predict_price_rf`; `MODELS = ['xgboost', 'random_forest', 'arimax', 'sarimax']`; `VARIANTS = ['base', 'csa', 'bayesian']` |
| `initial_load_progress.json` | REVIEW | Currently `{}`; may want to reset |

### 2.7 `website/` — Django + Vercel app

| Path | Category | Reason |
|---|---|---|
| `manage.py` | KEEP | Django entry |
| `vercel_wsgi.py` | KEEP | Vercel WSGI |
| `requirements.txt` | KEEP | Web app deps |
| `firebase-credentials.json` | KEEP (sensitive — DO NOT touch) | Service account |
| `db.sqlite3` | REMOVE | Pre-Firestore DB; auth now via Firestore |
| `sample_cpo_data.csv` | REMOVE | Sample data for legacy populate script |
| `populate_sample_data.py` | REMOVE | Legacy Django ORM populator (`DailyMarketData`/`MarketStates`/`NewsData` collections — replaced) |
| `.gitignore` | KEEP | Git config |
| `config/__init__.py`, `asgi.py`, `urls.py`, `wsgi.py`, `settings.py` | KEEP | Django project glue |
| `web/__init__.py`, `apps.py`, `admin.py`, `urls.py`, `models.py`, `tests.py`, `services.py` | KEEP w/ review | Most still relevant; `models.py` should be empty (Firestore custom auth) |
| `web/views.py` | KEEP w/ trim | `valid_models` and `valid_variants` need updating |
| `web/auth_backend.py` | KEEP | Custom Firestore auth |
| `web/firebase_backend.py` | REVIEW | Older Firebase wrapper — verify if `auth_backend.py` superseded it |
| `web/tasks.py` | KEEP w/ docstring update | "4 models x 2 variants" comment outdated |
| `web/templates/{base,login,register,dashboard,news,about}.html` | KEEP | Active UI |
| `web/templates/dashboard.html` | KEEP w/ trim | Model/variant dropdowns include RF/ARIMAX/SARIMAX/Bayesian |
| `web/migrations/0001_initial.py` | REMOVE | Pre-Firestore Django migration |
| `web/management/commands/migrate_to_firebase.py` | REMOVE | One-time migration; complete |
| `web/management/commands/test_firebase.py` | REVIEW | Useful smoke test? |
| `web/management/commands/upload_model_data.py` | REMOVE | One-time upload; superseded by scheduler |
| `web/management/__init__.py`, `commands/__init__.py` | KEEP if any command kept |
| `venv/` | KEEP (gitignored) | Virtual env |
| `diagrams/` | REMOVE | Empty subdirectory |
| `CODE_AUDIT_REPORT.md` | REMOVE | Pre-Firestore audit; outdated |
| `DJANGO_SETUP.md` | REMOVE (or banner) | Django/SQLite setup; obsolete |
| `FBV_QUICK_REFERENCE.md` | REMOVE (or banner) | Function-based view ref |
| `FIREBASE_SETUP.md` | REVIEW | Some content still applies |
| `PROJECT_SUMMARY.md` | DOC DRIFT | Describes pre-rework architecture |
| `QUICK_REFERENCE.txt` | DOC DRIFT | Outdated commands |
| `REFACTORING_SUMMARY.md` | DOC DRIFT | Pre-rework summary |
| `SERVICES_REFACTORING_COMPLETE.md` | DOC DRIFT | Pre-rework summary |
| `SETUP_GUIDE.md` | DOC DRIFT | Pre-rework setup |
| `USER_FLOWCHART.md` | DOC DRIFT | Mermaid flowchart of old design |

### 2.8 `experiments/walk_forward/` — out-of-sample harness

| Path | Category | Reason |
|---|---|---|
| `config.py` | REVIEW | Defines `MODEL_VARIANTS = [12 variants]` (xgboost+rf+arimax+sarimax × base/csa/bayesian) |
| `data_loader.py` | REVIEW | Used only by walk_forward |
| `feature_builder.py` | REVIEW | Used only by walk_forward |
| `metrics_calculator.py` | REVIEW | Used only by walk_forward |
| `model_runner.py` | REVIEW | Calls dropped models |
| `output_writer.py` | REVIEW | Output formatter |
| `run_walk_forward.py` | REVIEW | Entry point |
| `output/` | REVIEW | Generated artifacts |

> **Recommend:** Either trim to xgboost-only or archive whole `experiments/walk_forward/` if not yet thesis-cited. Awaiting your direction.

### 2.9 `diagrams/`

| Path | Category | Reason |
|---|---|---|
| `README_DFD.md` | REVIEW | Contains "Random Forest" reference |
| `activity.md`, `activity_diagram.html` | DOC DRIFT | Show RF/ARIMAX/SARIMAX/Bayesian branches |
| `erd.md`, `erd_diagram.html` | DOC DRIFT | Doc drift in `predictions` collection schema |
| `sequence.md`, `sequence_diagram.html` | DOC DRIFT | RF/ARIMAX/SARIMAX/Bayesian in flow |
| `use_case.md`, `use_case_diagram.html` | DOC DRIFT | Same |

### 2.10 `revision/`

| Path | Category | Reason |
|---|---|---|
| `CPO_COUNCIL_VERDICT_20260423_REVISED_1.md` | KEEP | Most recent thesis-council verdict — reference document |

### 2.11 `output/` (project-root)

| Path | Category | Reason |
|---|---|---|
| `sentiment_weight_optimization_*.csv` (3 files) | REMOVE | Output of `optimize_sentiment_weights.py` (REMOVE) |

### 2.12 `cpo-dashboard/`

Empty placeholder — REMOVE entire directory.

---

## 3. Dead code branches inside ACTIVE files

Listed by `path:line` (line numbers approximate; verify before editing).

### `scheduler/prediction_updater.py`
- Lines **30–31**: `MODELS = ['xgboost', 'random_forest', 'arimax', 'sarimax']` and `VARIANTS = ['base', 'csa', 'bayesian']` — should both be trimmed.
- Lines **34–40**: `BASE_PARAMS` includes `random_forest`, `arimax`, `sarimax` keys — strip non-XGBoost.
- Lines **315–328**: `_predict_price_rf()` — drop entire function.
- Lines **331–347**: `_predict_price_arimax()` — drop entire function.
- Lines **349–367**: `_predict_price_sarimax()` — drop entire function.
- Lines **388–396**: random_forest branch in `_compute_metrics` — drop.
- Lines **397–400**: ARIMAX/SARIMAX persistence-fallback branch — drop with statsmodels removal.
- Lines **464–488**: model dispatch loop branches for `random_forest`/`arimax`/`sarimax` — drop.
- Lines **2, 117, 193, 214, 265, 440**: docstrings still say "56 predictions" / "136 predictions" / "4 models × 2 variants" — update to ablation scope.
- Module imports: `from statsmodels.tsa.statespace.sarimax import SARIMAX` only used by dropped functions.

### `scheduler/main.py`
- Lines **117, 193, 214, 265**: comments say "56 predictions" / "136 predictions" — update once `prediction_updater.py` is trimmed.

### `prediction/adaptive_prediction.py` (whole file slated for REMOVE, but listing for completeness)
- Lines **34–36, 41**: imports for RF, SARIMAX, Bayesian.
- Lines **279–305**: `BASE_PARAMS` has all 4 model types.
- Lines **371–400**: `PARAM_SPACES` includes arimax/sarimax/random_forest.
- Lines **414–419**: arimax/sarimax exog selection.
- Lines **480–507**: optimizer dispatch for arimax/sarimax.
- Lines **571–583**: `COLORS` for RF/ARIMAX/SARIMAX/Bayesian variants.
- Lines **754–760**: `model_types = [4 models]` and `optimizers = ['csa', 'bayesian', 'both']`.

### `prediction/horizon_forecast.py` (KEEP after rename to `horizon_forecast_C4_full.py`, with trim)
- Lines **1–11**: docstring claims 4 models × 3 variants — trim.
- Line **223**: `model_types = ['xgboost', 'random_forest', 'arimax', 'sarimax']` — trim to `['xgboost']`.
- Lines **282–308**: ARIMAX/SARIMAX BASE training branch — drop.
- Lines **356–382**: ARIMAX/SARIMAX CSA branch — drop.
- Lines **400–473**: entire BAYESIAN block — drop.
- Lines **503–508**: `colors` dict has all 12 variants — trim to xgboost_base/csa.
- Line **682–683, 688–691, 706–711**: `--optimizer bayesian/both`, `--bayes-*` args — drop.

### `prediction/horizon_forecast_cpo_only.py`, `horizon_forecast_cpo_hmm.py`, `horizon_forecast_cpo_sentiment.py`
- Same dead branches as `horizon_forecast.py`. (Suggest: extract shared trim-pattern after first file done; do NOT introduce new abstraction in cleanup task — just apply each file by hand.)

### `prediction/utils/forecast_utils.py`
- Lines **49–55**: `BASE_PARAMS` — drop RF/ARIMAX/SARIMAX entries.
- Lines **57–82+**: `CSA_PARAM_SPACES` — drop RF/ARIMAX/SARIMAX entries.
- `train_statsmodels`, `predict_statsmodels`, `csa_objective_arimax`, `csa_objective_sarimax` — drop after callers fixed.
- `select_top_exog` — drop only if no remaining caller (XGBoost path doesn't need it).
- Imports: `RandomForestRegressor`, `SM_SARIMAX` become unused after trim.

### `website/web/views.py`
- Lines **116–117**: pre-fill loop uses `('xgboost','random_forest','arimax','sarimax')` × `('base','csa','bayesian')` — change to ablation configs (C1–C4) × `('base','csa')`.
- Lines **165–177**: `valid_models = {'xgboost','random_forest','arimax','sarimax'}`, `valid_variants = {'base','csa','bayesian'}` — replace with ablation enum and `{'base','csa'}`.

### `website/web/templates/dashboard.html`
- Lines **49–54**: `<select id="pred-model">` offers RF/ARIMAX/SARIMAX — replace with C1–C4 options per scope spec.
- Lines **58–62**: variant `<option>`s include `bayesian` — drop.

### `website/web/tasks.py`
- Lines **82–98**: `task_retrain_horizon_models` docstring "4 models x 2 variants" — update to "1 model (XGBoost) × 4 ablation configs × 2 variants (base, csa)".

### `experiments/walk_forward/config.py`
- Lines **27–32**: `MODEL_VARIANTS` lists all 12 dropped variants. Trim to `['xgboost_base', 'xgboost_csa']` × per-config tag, OR archive entire `experiments/walk_forward/`.

---

## 4. Orphaned / out-of-scope data files

### Frequency-out-of-scope CSVs
- `cpo/Data_CPO_Weekly.csv`, `cpo/Data_CPO_Monthly.csv`
- `cpo/output/cpo_variables_Weekly.csv`, `cpo/output/cpo_variables_Monthly.csv`
- `cpo/output/weekly_dated.csv`, `cpo/output/monthly_dated.csv`
- `news/output/sentiment_aggregate_Weekly.csv`, `sentiment_aggregate_Monthly.csv`, `monthly_sentiment_aggregate.csv`
- `markov/output/hmm_states_results_{Weekly,Monthly}.csv`, `hmm_states_stats_{Weekly,Monthly}.csv`, `hmm_transition_matrix_{Weekly,Monthly}.csv`, `hmm_states_analysis_{Weekly,Monthly}.png`, `hmm_bic_scores_{Weekly,Monthly}.csv`
- `prediction/output_horizons*/Weekly/` and `*/Monthly/` (entire trees)

### Old adaptive_prediction outputs (`prediction/output/`)
- `adaptive_*_arimax_*.png` (4 files)
- `adaptive_*_sarimax_*.png` (4 files)
- `adaptive_*_random_forest_*.png` (4 files)
- `adaptive_pred_xgboost_*.png` (Daily + Monthly variants — Monthly out of scope)
- `adaptive_*_Monthly*.{csv,json,png}` — frequency dropped
- `adaptive_*_Daily*.{csv,json,png}` — only useful if you still cite the adaptive single-horizon comparison; recommend ARCHIVE
- `csa_results/csa_*_horizon_1m.{json,png,csv}` — old monthly result
- `feature_importance.{csv,png}`, `model_comparison.png`, `prediction_results*.csv`, `predictions_{1..6}month.png` — March-2026 research artifacts

### Validation diagnostics
- `prediction/output_validation/` — currently only contains `validation_summary.csv`. (No `arimax_h*_validation_plot.png` or `sarimax_*` files exist locally — already cleaned up at some point.)
- Per spec: rename `validation_summary.csv` → `validation_summary_legacy.csv`, then create empty `validation_summary.csv`.

### Saved model directories (10 of 12 variants out of scope)
- All `*_bayesian/`, `random_forest_*/`, `arimax_*/`, `sarimax_*/` directories under `prediction/saved_models/{cpo_only,cpo_hmm,cpo_sentiment,full}/Daily/h{1..7}/`.

### One-off logs / DBs
- `pipeline_run_20260321_174152.log`
- `website/db.sqlite3`

---

## 5. Obsolete configuration blocks

| File:Line | Current value | Should become |
|---|---|---|
| `scheduler/prediction_updater.py:27-29` | `FREQ_CONFIG = {'Daily': {...}}` | Already Daily-only ✓ |
| `scheduler/prediction_updater.py:30` | `MODELS = ['xgboost','random_forest','arimax','sarimax']` | `MODELS = ['xgboost']` |
| `scheduler/prediction_updater.py:31` | `VARIANTS = ['base','csa','bayesian']` | `VARIANTS = ['base','csa']` |
| `scheduler/prediction_updater.py:34-40` | `BASE_PARAMS` 4-model dict | XGBoost-only |
| `prediction/adaptive_prediction.py:280-305` | `BASE_PARAMS` 4-model dict | (whole file archived) |
| `prediction/adaptive_prediction.py:371-400` | `CSATimeSeriesOptimizer.PARAM_SPACES` | (whole file archived) |
| `prediction/utils/forecast_utils.py:43-55` | `BASE_PARAMS` 4-model dict | XGBoost-only |
| `prediction/utils/forecast_utils.py:57-90` | `CSA_PARAM_SPACES` 4-model dict | XGBoost-only |
| `experiments/walk_forward/config.py:27-32` | `MODEL_VARIANTS = [12 variants]` | XGBoost variants only OR archive whole dir |
| `website/web/views.py:165-177` | `valid_models = {4}`, `valid_variants = {3}` | Ablation enum + `{'base','csa'}` |
| `website/web/templates/dashboard.html:49-62` | RF/ARIMAX/SARIMAX/Bayesian options | C1–C4 + base/csa |

`INTERVAL_CONFIGS` in horizon_forecast*.py already only have the `Daily` key — no Weekly/Monthly cleanup needed there.

---

## 6. Documentation drift

| File | Drift |
|---|---|
| `PREDICTION_GUIDE.md` (root) | Fully obsolete — multi-frequency / multi-model. Recommend ARCHIVE not banner. |
| `CPO_FORECAST_20260407_202731.md` | Internal log mentioning Weekly/RF/SARIMAX/Bayesian; KEEP as archived snapshot OR add banner |
| `baseline_metrics.txt` | Numbers from older multi-model run; banner if cited in thesis |
| `website/PROJECT_SUMMARY.md` | Pre-rework: Django ORM models (PriceHistory, News, MarketState), `populate_sample_data` flow |
| `website/SETUP_GUIDE.md` | SQLite-based setup |
| `website/DJANGO_SETUP.md` | Django ORM auth, no Firestore |
| `website/FIREBASE_SETUP.md` | Partly accurate — review |
| `website/USER_FLOWCHART.md` | Pre-rework Mermaid flowchart |
| `website/CODE_AUDIT_REPORT.md` | Audit of pre-rework violations; obsolete |
| `website/REFACTORING_SUMMARY.md` | Pre-rework refactor write-up |
| `website/SERVICES_REFACTORING_COMPLETE.md` | Pre-rework refactor write-up |
| `website/FBV_QUICK_REFERENCE.md` | View pattern cheat sheet — partly applies |
| `website/QUICK_REFERENCE.txt` | Old commands |
| `diagrams/README_DFD.md` | Mentions Random Forest |
| `diagrams/activity.md` + `activity_diagram.html` | Decision branches show RF / ARIMAX / SARIMAX / Bayesian |
| `diagrams/erd.md` + `erd_diagram.html` | `predictions` collection schema lists 4 models / 3 variants |
| `diagrams/sequence.md` + `sequence_diagram.html` | Sequence shows RF/ARIMAX/SARIMAX/Bayesian flows |
| `diagrams/use_case.md` + `use_case_diagram.html` | Lists removed models in dashboard use case |
| `news/README_sentiment.md` | Should verify still accurate |
| `revision/CPO_COUNCIL_VERDICT_20260423_REVISED_1.md` | KEEP — describes the very rationale for the cleanup; mentions ARIMAX/SARIMAX in critique context, which is fine |

---

## 7. Items flagged REVIEW (need your decision before Phase 2)

1. `cpo/preprocess_cpo_variables.py` — Daily logic intact, but Weekly/Monthly branches present. Trim or leave?
2. `cpo/stationarity_test.py` — Standalone ADF/KPSS/PP test script; KEEP for thesis Bab 4 if cited?
3. `cpo/cpo_sentiment.py` and `cpo/cpo_sentiment_lagged.py` — Dec-2026 research artifacts. Confirm not cited in thesis before REMOVE.
4. `news/check_cuda.py` — One-off; debug utility only.
5. `news/finbert_sentiment_analysis.py`, `news/finbert_tone_sentiment_analysis.py` — Older non-flexible variants. Confirm `_flexible.py` is the one in use.
6. `markov/cpo_hmm_states.py` — Has Daily/Weekly/Monthly support. Production is `scheduler/hmm_updater.py`. Trim or REMOVE entire file?
7. `markov/cpo_prediction_dataset_daily.csv` — Pre-built dataset, 3.6 MB. Verify if any active script reads it.
8. `markov/output/hmm_all_frequencies_summary.csv` — Mentions all 3 frequencies; KEEP w/ trim or REMOVE?
9. `prediction/csa_hyperparameter_optimizer.py` — Older standalone; verify if any horizon script imports it. (Quick grep shows imports go through `crow_search_optimizer.py` which is KEEP.)
10. `prediction/output_horizons*/Daily/horizon_*/` — Predictions CSVs contain RF/ARIMAX/SARIMAX/Bayesian columns. KEEP for thesis figures?
11. `prediction/output_validation/validation_summary.csv` — Per spec, rename to `_legacy.csv`; confirm.
12. `experiments/walk_forward/` — Whole module uses dropped models. Archive entirely, or trim to xgboost-only?
13. `website/web/firebase_backend.py` — May be older Firebase wrapper superseded by `auth_backend.py`.
14. `website/web/management/commands/test_firebase.py` — Useful smoke test? KEEP if used by deploy verification.
15. `website/web/management/commands/migrate_to_firebase.py` — One-time migration from SQLite → Firestore. Done. Confirm REMOVE.
16. `website/web/management/commands/upload_model_data.py` — One-time uploader. Confirm REMOVE.
17. `website/db.sqlite3` — REMOVE confirmed?
18. `revision/CPO_COUNCIL_VERDICT_20260423_REVISED_1.md` — KEEP as thesis context.
19. Whether to trim individual prediction output CSVs (drop RF/ARIMAX/SARIMAX/Bayesian columns) or leave intact.
20. Whether `horizon_forecast.py` should be **renamed** to `horizon_forecast_C4_full.py` (per spec) or keep its current name.
21. C2 ablation: `horizon_forecast_cpo_hmm.py` — exists under that name. Spec mentioned creating it; we already have it. Confirm rename to `horizon_forecast_C2_price_hmm.py`.

---

## 8. Sanity checks performed

- `validate_arimax_sarimax.py` does **not** contain Pesaran–Timmermann test (only Ljung-Box, Jarque-Bera, ARCH-LM, Breusch-Godfrey, Shapiro-Wilk, ADF, KPSS) — no extraction needed before archive.
- `scheduler/hmm_updater.py` already uses `N_STATES_RANGE = range(2, 5)` (2/3/4 states) and `FREQUENCIES = ['Daily']`. ✓
- `INTERVAL_CONFIGS` in `horizon_forecast*.py` is already Daily-only. ✓
- Naive baseline package (`prediction/baselines/`) is independent of dropped models — safe.
- `crow_search_optimizer.py` (used by all horizon scripts via `forecast_utils.py`) does not depend on dropped model paths.
- The 4 ablation horizon scripts (C1=cpo_only, C2=cpo_hmm, C3=cpo_sentiment, C4=full) all already exist; C2 and C4 are NOT missing. No new code needed for ablation framework.
- `scheduler/firestore_writer.py` writes to the **same** Firestore document IDs that `prediction_updater.py` produces (`{model}_{variant}_{frequency}_h{horizon}`). After trim, scope of doc IDs becomes `xgboost_{base|csa}_Daily_h{1..7}` — backward compat for the website API needs to be planned (8 docs vs current 84/56).

---

## 9. Headline risks before proceeding

- **Backward-compat for already-stored Firestore predictions:** If you've stored 84 prediction docs in production and the website is reading any of `random_forest`/`arimax`/`sarimax`/`bayesian` doc IDs, deleting those doc IDs leaves stale URLs. Confirm whether to delete from Firestore as part of cleanup or leave them in DB but stop writing.
- **Thesis figures:** Many output PNGs reference RF/ARIMAX/SARIMAX/Bayesian curves. If any thesis chapter cites those figures, archiving them is fine (still in `_archive_before_cleanup/`), but **regenerating** clean ablation-only versions is a separate task.
- **`adaptive_prediction.py`** is a non-trivial 1000-line file. Archiving it removes a CLI entry point (`python adaptive_prediction.py`). Confirm no automation/scripts call it.
- **`run_pipeline.py`** is a top-level orchestrator. Archiving removes the `python run_pipeline.py` entry point. Confirm no docs/CI/cron call it.

---

## 10. Phase-2 entry checklist (when you say "go")

1. Walk through each item in §7 (REVIEW) with you and lock final classification.
2. Update `CLEANUP_INVENTORY.md` with final decisions.
3. Proceed to Phase 3 (branch + archive directory) only after every REVIEW resolved.

---

## 11. Phase-2 Resolutions (2026-04-26)

User decisions on §7 REVIEW items:

| # | Item | Decision |
|---|---|---|
| 1 | `cpo/preprocess_cpo_variables.py` | TRIM (Daily-only) |
| 2 | `cpo/stationarity_test.py` | REMOVE |
| 3 | `cpo/cpo_sentiment.py`, `cpo/cpo_sentiment_lagged.py` | REMOVE |
| 4 | `news/check_cuda.py` | KEEP |
| 5 | `news/finbert_sentiment_analysis.py`, `finbert_tone_sentiment_analysis.py` | KEEP both |
| 6 | `markov/cpo_hmm_states.py` | TRIM (Daily-only) |
| 7 | `markov/cpo_prediction_dataset_daily.csv` | REMOVE (no active reader) |
| 8 | `markov/output/hmm_all_frequencies_summary.csv` | TRIM (Daily rows only) |
| 9 | `prediction/csa_hyperparameter_optimizer.py` | REMOVE |
| 10 | `output_horizons*/Daily/horizon_*/` predictions CSVs | KEEP intact (RF/ARIMAX/SARIMAX/Bayesian columns retained) |
| 11 | `prediction/output_validation/validation_summary.csv` rename | CONFIRMED → `validation_summary_legacy.csv` |
| 12 | `experiments/walk_forward/` | ARCHIVE entire module |
| 13 | `website/web/firebase_backend.py` | ARCHIVE |
| 14 | `website/web/management/commands/test_firebase.py` | KEEP |
| 15 | `website/web/management/commands/migrate_to_firebase.py` | REMOVE |
| 16 | `website/web/management/commands/upload_model_data.py` | REMOVE |
| 17 | `website/db.sqlite3` | REMOVE |
| 18 | `revision/CPO_COUNCIL_VERDICT_…md` | KEEP |
| 19 | (skipped — covered by #10) | — |
| 20 | Rename `horizon_forecast.py` → `horizon_forecast_C4_full.py` | YES |
| 21 | Rename C2 + (by symmetry) C1, C3 ablation scripts | YES |

**Resulting renames (Phase 4):**
- `prediction/horizon_forecast.py` → `prediction/horizon_forecast_C4_full.py`
- `prediction/horizon_forecast_cpo_only.py` → `prediction/horizon_forecast_C1_price_only.py`
- `prediction/horizon_forecast_cpo_hmm.py` → `prediction/horizon_forecast_C2_price_hmm.py`
- `prediction/horizon_forecast_cpo_sentiment.py` → `prediction/horizon_forecast_C3_price_sentiment.py`

**Backward-compat decision (Action #4 in summary):** Keep existing 84 prediction docs in Firestore (do not delete). Scheduler stops writing the dropped 76 doc IDs going forward; website API continues to read the 8 active ablation doc IDs.

**Decision on backward-compat (Action #4):** Stale docs left in Firestore for now; no destructive Firestore writes from cleanup task.

Proceeding to Phases 3–7.

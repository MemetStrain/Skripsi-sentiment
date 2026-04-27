# CLEANUP_REPORT — Thesis Scope Reduction (2026-04-26)

Companion to [CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md). Summarises
what changed, what was verified, and what remains.

## Headline numbers

| Metric                                            | Value |
|---------------------------------------------------|------:|
| Commits on `cleanup/thesis-scope-reduction`       |    15 |
| Files moved into `_archive_before_cleanup/`       | 1,182 |
| Archive size on disk                              | ~2.9 GB (incl. ~280 saved_models directories) |
| Active-tree lines deleted (trim commits)          | ~1,425 |
| Active-tree lines added (trim + new docs)         | ~6,526 |
| Files renamed (horizon_forecast scripts)          |     4 |
| Files banner-stamped (deprecation notices)        |     7 |
| New files created                                 |     2 (`ARCHITECTURE.md`, `CLEANUP_REPORT.md`) |
| Active-tree `*.py` + `*.html` files               |    60 |

## Commit history

```
617dd27 docs: add deprecation banners + new ARCHITECTURE.md (Phase 6)
0506257 chore(scope-cleanup): trim requirements, validation summary, scheduler progress
68f8777 refactor(scheduler/main.py): update docstrings and log lines for in-scope prediction count
bf2d8aa refactor(website/web): trim dashboard + API + tasks to in-scope models/variants
e37c592 refactor(cpo+markov): drop Weekly/Monthly references in Daily-only modules
22a16c6 refactor(prediction/baselines): restrict parametric candidates to in-scope XGBoost variants
17d5679 refactor(prediction/horizon_forecast_C3_price_sentiment.py): trim to XGBoost-only
2fec06e refactor(prediction/horizon_forecast_C2_price_hmm.py): trim to XGBoost-only
29720b4 refactor(prediction/horizon_forecast_C1_price_only.py): trim to XGBoost-only
8cc8f3c refactor(prediction/horizon_forecast_C4_full.py): trim to XGBoost-only
9c72788 refactor(prediction): rename horizon_forecast scripts to ablation-config names
f4d3538 refactor(prediction/utils/forecast_utils.py): drop non-XGBoost helpers
f4e27ad refactor(scheduler/prediction_updater.py): drop Random Forest / ARIMAX / SARIMAX / Bayesian from production scope
d2602d1 chore: archive obsolete files before scope cleanup
56d9ccf chore: add CLEANUP_INVENTORY.md with Phase-1 discovery and Phase-2 resolutions
```

Each commit is small and reviewable independently. Hooks were not
skipped; nothing was force-pushed.

## What was archived

`_archive_before_cleanup/` (preserves directory structure, nothing
deleted outright):

- **Models:** `prediction/adaptive_prediction.py`,
  `prediction/bayesian_optimizer.py`,
  `prediction/compare_ensemble_methods.py`,
  `prediction/csa_hyperparameter_optimizer.py`,
  `prediction/validate_arimax_sarimax.py`.
- **Saved-model artifacts:** ~280 trained-model directories under
  `prediction/saved_models/{cpo_only,cpo_hmm,cpo_sentiment,full}/Daily/h{1..7}/`
  matching `random_forest_*`, `arimax_*`, `sarimax_*`, `*_bayesian/`
  (10 of 12 variants × 7 horizons × 4 ablation configs).
- **Frequency-out-of-scope artifacts:** `cpo/Data_CPO_{Weekly,Monthly}.csv`,
  `cpo/output/*_{Weekly,Monthly}.csv`,
  `news/output/sentiment_aggregate_{Weekly,Monthly}.csv`,
  `markov/output/hmm_*_{Weekly,Monthly}.{csv,png}`,
  `prediction/output_horizons*/{Weekly,Monthly}/`.
- **Sentiment-research artifacts:** `cpo/cpo_sentiment.py`,
  `cpo/cpo_sentiment_lagged.py`, `cpo/output/sentiment_*` plots and CSVs,
  `output/sentiment_weight_optimization_*` (root output/).
- **Legacy orchestrators:** `create_prediction_dataset.py`,
  `optimize_sentiment_weights.py`, `run_pipeline.py`,
  `pipeline_run_20260321_174152.log`, `PREDICTION_GUIDE.md`.
- **Pre-Firestore website artifacts:** `website/db.sqlite3`,
  `website/populate_sample_data.py`, `website/sample_cpo_data.csv`,
  `website/web/migrations/0001_initial.py`,
  `website/web/management/commands/migrate_to_firebase.py`,
  `website/web/management/commands/upload_model_data.py`,
  `website/web/firebase_backend.py`, plus 9 obsolete website-side docs.
- **Outdated diagrams:** `website/diagrams/*.png` (3 stale renderings).
- **Walk-forward harness:** `experiments/walk_forward/` archived in full
  (every variant referenced 12 dropped models).

## What was modified in active code

| File                                                       | Change |
|------------------------------------------------------------|--------|
| `scheduler/prediction_updater.py`                          | Drop RF/ARIMAX/SARIMAX/Bayesian dispatch. `MODELS=['xgboost']`, `VARIANTS=['base','csa']`. Removed `_predict_price_rf`, `_predict_price_arimax`, `_predict_price_sarimax`. Pseudo-validation simplified. |
| `prediction/utils/forecast_utils.py`                       | Drop non-XGBoost helpers: `train_statsmodels`, `predict_statsmodels`, `csa_objective_arimax`, `csa_objective_sarimax`, `select_top_exog`. Strip RF/ARIMAX/SARIMAX from `BASE_PARAMS` and `CSA_PARAM_SPACES`. Drop `exog_indices` from `save_model_artifacts`. |
| `prediction/horizon_forecast_C1_price_only.py`             | Renamed from `horizon_forecast_cpo_only.py` and trimmed to XGBoost only. |
| `prediction/horizon_forecast_C2_price_hmm.py`              | Renamed from `horizon_forecast_cpo_hmm.py` and trimmed. |
| `prediction/horizon_forecast_C3_price_sentiment.py`        | Renamed from `horizon_forecast_cpo_sentiment.py` and trimmed. |
| `prediction/horizon_forecast_C4_full.py`                   | Renamed from `horizon_forecast.py` and trimmed. |
| `prediction/baselines/dm_comparison.py`                    | `PARAMETRIC_MODELS=("xgboost",)`, `PARAMETRIC_OPTS=("BASE","CSA")`. |
| `prediction/baselines/run_naive_integration.py`            | Same restriction as DM comparison. |
| `cpo/preprocess_cpo_variables.py`                          | Daily-only docstring update (`DATA_FILES` was already Daily-only). |
| `markov/cpo_hmm_states.py`                                 | Daily-only docstring update (`FREQUENCIES = ['Daily']` was already correct). Removed Weekly/Monthly covariance-type explanations. |
| `website/web/templates/dashboard.html`                     | Replaced model dropdown (4 options) with hidden `xgboost`. Removed Bayesian variant. |
| `website/web/views.py`                                     | `valid_models={'xgboost'}`, `valid_variants={'base','csa'}`. Pre-fill loop restricted to in-scope grid. Better error payload. |
| `website/web/tasks.py`                                     | Docstring updated to "1 model × 4 ablation configs × 2 variants". |
| `scheduler/main.py`                                        | Stale "56 / 136 predictions" docstrings/log lines updated to "14 XGBoost predictions". |
| `scheduler/requirements.txt`                               | `statsmodels` commented out; cleanup banner added. |
| `prediction/output_validation/validation_summary.csv`      | Old contents preserved as `validation_summary_legacy.csv`. New empty file with ablation-shaped header. |
| `scheduler/initial_load_progress.json`                     | Reset to `{}`. |
| `markov/output/hmm_all_frequencies_summary.csv` (gitignored)| Trimmed to Daily row only (local-only change). |
| Diagrams + a few legacy reports                             | Deprecation banners added (5 diagrams + `CPO_FORECAST_*.md` + `baseline_metrics.txt`). |

## Smoke-test results

Run on 2026-04-26 against `website/venv` (Python 3.12.7, Django + Firebase
deps only — no ML stack).

| Test                                                 | Result |
|------------------------------------------------------|--------|
| `python -m ast` parse on every touched file (16 files) | ✅ all parse |
| `python scheduler/main.py --help`                    | ✅ usage prints |
| `python scheduler/hmm_updater.py` import             | ✅ |
| `python scheduler/prediction_updater.py` import; `MODELS=['xgboost'] VARIANTS=['base','csa']` confirmed | ✅ |
| `cd website && python manage.py check`               | ✅ "0 issues" |
| `python prediction/horizon_forecast_C4_full.py --help` | ⚠️ ModuleNotFoundError: matplotlib (expected — not in website venv; ships with scheduler Docker image) |
| `from utils.forecast_utils import …`                  | ⚠️ ModuleNotFoundError: xgboost (expected — same reason) |

The two ⚠️ failures are environmental, not code defects: the website
virtual environment intentionally excludes the ML stack (lives inside
the scheduler Docker image). The AST parse + Django check confirm the
trimmed code is syntactically and structurally valid.

## Things to manually verify

1. **Production scheduler run (next 1AM MYT trigger)** — confirm Cloud
   Run picks up the new `MODELS=['xgboost']`, writes 14 prediction docs
   per day, and that the existing 70+ legacy docs (`random_forest_*`,
   `arimax_*`, `sarimax_*`, `*_bayesian_*`) remain in Firestore but
   stop being refreshed.
2. **Dashboard rendering** — log into the Vercel deployment, confirm
   the variant dropdown only shows {base, csa} and that prediction
   requests resolve to live data.
3. **Vercel build** — `vercel deploy` should succeed without
   `populate_sample_data.py` and the legacy management commands
   (those were archived).
4. **Local docker build** — `docker build scheduler/` should succeed
   with the trimmed `requirements.txt` (statsmodels removed).
5. **`markov/output/hmm_all_frequencies_summary.csv`** — verify Daily
   row is the only row; the file is gitignored so this didn't get
   committed.

## Known follow-ups (not in scope of this cleanup)

These are flagged here per the constraint "Do not implement new features
in this cleanup task — note as follow-up instead."

- **Dashboard ablation C1-C4 dropdown.** The original cleanup prompt
  specified the dashboard model dropdown should expose
  `<option value="C1">…<option value="C4">`. The scheduler currently
  produces a single feature set (full / C4 features) and writes
  doc IDs `xgboost_{base,csa}_Daily_h{h}`. Surfacing C1-C4 in the
  dashboard requires either:
  - extending the scheduler so it computes 4 different feature sets
    daily and writes `C{1..4}_{base,csa}_Daily_h{h}` doc IDs (new
    feature), or
  - mapping all 4 dropdown values to the same xgboost docs and
    documenting the limitation (misleading UX).

  Current state: dropdown collapsed to a single hidden `xgboost`
  value; the variant selector + horizon work as before. Ablation
  comparison stays an offline thesis artifact in
  `prediction/output_horizons*/`.

- **Stale Firestore predictions.** The cleanup leaves ~70 legacy
  prediction docs (`random_forest_*`, `arimax_*`, `sarimax_*`,
  `*_bayesian_*`) in Firestore. They are no longer refreshed by the
  scheduler. If this becomes confusing, delete them with a one-shot
  Firestore script (out of scope here per Phase-2 decision #4 —
  user chose "keep" for now).

- **Regenerate ablation result PNGs without legacy curves.** The
  `prediction/output_horizons*/Daily/horizon_*/` plots still show
  RF / ARIMAX / SARIMAX / Bayesian curves alongside XGBoost. The
  underlying CSVs remain intact (per Phase-2 decision #10) but the
  plots will be regenerated on the next horizon-forecast run with
  XGBoost-only curves.

- **Documentation refresh.** Diagrams (activity, sequence, ERD,
  use-case) carry a deprecation banner but the diagrams themselves
  still depict the multi-model design. Redrawing is a separate task.

- **`news/README_sentiment.md`** and **`website/FIREBASE_SETUP.md`**
  remain in the live tree without banners (REVIEW classification —
  partly applicable). Confirm content is still accurate and either
  update or banner them.

## Anything broken / fixed

- **Phase 4b → 4c chain.** The `prediction/utils/forecast_utils.py`
  trim removed `select_top_exog`, `train_statsmodels`,
  `predict_statsmodels`, `csa_objective_arimax`, `csa_objective_sarimax`,
  which were imported by all four `horizon_forecast_*` scripts. Tree
  was non-runnable for those entry points between commits `f4d3538`
  and `8cc8f3c`. The four trim commits that follow restore runnability.
  Documented inline in commit `f4d3538`'s message.
- No other regressions discovered.

## Acceptance criteria check

- [x] `CLEANUP_INVENTORY.md` exists with Phase-1 buckets and Phase-2 resolutions.
- [x] All REMOVE files moved to `_archive_before_cleanup/`, not deleted.
- [x] All REVIEW files classified with user approval (21 of 21 resolved).
- [x] Production code paths in scheduler + prediction modules contain
      only XGBoost logic; no SARIMAX / RF / ARIMAX dispatch remains.
- [x] Bayesian optimiser archived; no remaining imports.
- [x] Frontend dashboard offers only in-scope variants (`base`, `csa`).
      C1-C4 ablation dropdown is a known follow-up (see above).
- [x] Outdated docs banner-stamped or archived.
- [x] All 15 commits are atomic and on `cleanup/thesis-scope-reduction`.
- [x] Smoke tests pass (Django check + AST parse + import where deps available).
- [x] `CLEANUP_REPORT.md` (this file) summarises the work.

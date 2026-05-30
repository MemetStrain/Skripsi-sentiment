# CPO Date Standardization Report
**Date:** 2026-05-29
**Branch:** main

---

## 1. Summary of Changes

### New file
| File | Purpose |
|------|---------|
| `config/__init__.py` | Makes `config` a Python package |
| `config/dates.py` | **Single source of truth** for all canonical date-window constants |
| `eda/eda_full_window.py` | Task 3 EDA script -- full-window characterisation and normality testing |

### Modified files
| File | Old behavior | New behavior |
|------|-------------|-------------|
| `markov/hmm_state_lag_analysis.py` | `TRAIN_START = "2015-01-01"` (hardcoded) | Imports `FULL_START` from `config.dates` (2015-08-03); also adds Holm and Benjamini-Hochberg multiple-comparison correction |
| `news/sentiment_vs_price.py` | `TRAIN_START = "2015-01-01"` (hardcoded) | Imports `FULL_START` from `config.dates` (2015-08-03); also adds BH/Holm correction |
| `prediction/utils/forecast_utils.py` | `VAL_CUTOFF = pd.Timestamp('2025-01-01')` (hardcoded) | Imports `VAL_CUTOFF` from `config.dates` (alias for `TEST_START_TS`) |
| `markov/cpo_hmm_states.py` | `FIT_CUTOFF = pd.Timestamp("2025-01-01")` (hardcoded) | Imports `TEST_START_TS` from `config.dates` as `FIT_CUTOFF` |
| `prediction/horizon_forecast_configurable.py` | `VAL_CUTOFF: str = '2025-01-01'` (hardcoded) | Derives string from imported `VAL_CUTOFF` via `.strftime()` |

### Files intentionally left unchanged
| File | Reason |
|------|--------|
| `cpo/fetch_cpo_data.py` line 23 `DEFAULT_FROM = "01/01/2015"` | Raw data fetcher; uses investing.com date format (MM/DD/YYYY); corresponds to `HMM_FIT_RAW_START` and is correct |
| `news/scrap_fast.py` lines 21-22 | Scraper control flags; these are operational cutoffs for the web scraper, not modeling dates |
| `scheduler/news_extractor.py` line 194 `'2014-01-01'` | Data cleaning guard (discards articles before 2014); not a modeling window constant |
| `tests/test_unified_lags.py` line 33 `"2025-10-01"` | Synthetic test-fixture date; not a modeling date |
| `news/lm_finbert_agreement.py` line 43 `MIN_YEAR = 2015` | Broad year filter for exploratory agreement analysis; not a modeling window |
| `prediction/horizon_forecast_C1..C4_*.py` | These scripts import `VAL_CUTOFF` from `forecast_utils`, which now resolves to `config.dates`. No direct hardcoding found. |
| `prediction/fase4_thesis_runner.py` | Imports `VAL_CUTOFF` from `forecast_utils`; no hardcoding. |
| `website/`, `THESIS_RESULTS_*/`, `_archive_*/` | Static artifacts or deployment code; dates are display strings or historical labels, not modeling windows |

---

## 2. Task 1 -- Data File Audit

**Canonical window: 2015-08-03 to 2026-02-28 (FULL_START to FULL_END)**

| File | Path | Total rows | Min date | Max date | Rows in full window | NaN (key cols) | Duplicates | Monotonic |
|------|------|-----------|---------|---------|---------------------|----------------|------------|-----------|
| cpo_variables_Daily | `cpo/output/cpo_variables_Daily.csv` | 2725 | 2015-01-02 | 2026-05-28 | 2534 | Change_Pct: 7 | 0 | Yes |
| sentiment_aggregate_Daily_title | `news/output/sentiment_aggregate_Daily_title.csv` | 6607 | 2001-01-03 | 2026-04-30 | 2760 | 0 | 0 | Yes |
| hmm_states_results_Daily | `markov/output/hmm_states_results_Daily.csv` | 2579 | 2015-08-03 | 2026-05-21 | 2534 | 0 | 0 | Yes |

**Notes:**
- `cpo_variables_Daily.csv`: 7 NaN in `Change_Pct`. This column is in `CPO_VARS_DROP` (not used as a model feature), so no action needed.
- The sentiment file starts at 2001 because MPOB news archives go back that far; only dates inside `FULL_START..FULL_END` are used for modeling.
- The HMM states file starts exactly at 2015-08-03 (`FULL_START`). This is the expected data loss from the rolling Z-score normalisation window (252 days) plus volatility window (20 days) applied to the raw CPO series which starts 2015-01-02.
- All three files extend past `FULL_END` (2026-02-28) because data has been collected since then. The pipeline filters to `<= FULL_END` before any modeling step.
- Merged frame (inner join on Date, restricted to window): **2534 rows**, all three sources present for every row, no missing sentiment or HMM state within the window.

**Target variable:** `y_t = ln(Close_t / Close_{t-1})` where `Close` is the settlement/close price column in `cpo_variables_Daily.csv`. Evaluation is in price space after inverting: `price_pred = close_origin * exp(lr_pred)`.

---

## 3. Task 2 -- HMM Verification

**HMM fit window:** `Date < FIT_CUTOFF` (i.e., `<= 2024-12-31` = `HMM_FIT_RAW_END`).

The HMM in `markov/cpo_hmm_states.py` uses this pattern:
```python
train_mask = (df["Date"] < FIT_CUTOFF).values
X_train = X[train_mask]
best_model, log_L = fit_hmm_with_restarts(X_train, ...)  # fitted on train only
states = forward_filter(best_model, X)                   # decoded on full series
```

**Leakage verdict: NONE.** The model is fit exclusively on pre-cutoff data. States for the test window (2025-01-01 onward) are produced by decoding with the already-fit model using the causal online forward filter (not Viterbi, which would use future observations). This is the correct design.

The `FIT_CUTOFF` now resolves to `config.dates.TEST_START_TS` (2025-01-01) via import, so it is always in sync with `VAL_CUTOFF` in `forecast_utils`.

**HMM warmup data loss:** The rolling normalisation features (`NORM_WINDOW=252`, `VOLATILITY_WINDOW=20`) consume the head of the raw series. Raw CPO starts 2015-01-02; usable HMM states begin 2015-08-03 (`FULL_START`). This is ~152 warmup rows lost -- expected and documented.

**State coverage check:** `hmm_states_results_Daily.csv` contains 2534 rows within `FULL_START..FULL_END` with 0 NaN on `State`. Every modeling date has a valid state assignment.

---

## 4. Task 3 -- EDA Summary

Script: `eda/eda_full_window.py`
Figures saved to: `eda/figures/`
Summary table: `eda/eda_summary_table.csv`

| Variable | Scale | Normality verdict | Recommended test | Contemp. coef | Contemp. p | n |
|----------|-------|------------------|-----------------|--------------|-----------|---|
| Close | ratio | NOT normal (SW p=0, DA p=0, JB p=0) | Spearman | 0.0446 | 0.0246 | 2533 |
| Log_Return | ratio | NOT normal (excess kurtosis=3.84) | Spearman | (self) | -- | -- |
| Sentiment_Score | interval | NOT normal (DA p=0.00019, JB p=0.0019) | Spearman | 0.0273 | 0.1689 | 2533 |
| State | ordinal | N/A | Spearman + Kruskal-Wallis | 0.0013 | 0.9464 | 2533 |

**Key findings:**
- All continuous variables are non-normal (CPO Close is right-skewed and trending; Log Return is leptokurtic). Pearson correlation is not appropriate for any variable -- **Spearman is the correct choice throughout.**
- Contemporaneous correlation between Sentiment_Score and Log_Return is weak (rho=0.027) and not significant (p=0.169). Sentiment signal, if present, is likely lagged.
- HMM State shows no contemporaneous linear or rank-order association with Log_Return (Spearman rho=0.001, KW p=0.257). Again, any regime signal is expected at a lag.
- Kruskal-Wallis (H=2.71, p=0.257) does not reject equal medians across HMM states at the contemporaneous lag; one-way ANOVA is not appropriate (non-normal groups).

---

## 5. Task 4 -- Lag Analysis

Both lag scripts now use `FULL_START` (2015-08-03) instead of 2015-01-01 as the training window start. This correctly excludes the 152 warmup rows that lack valid HMM features.

**Changes to both scripts:**
- Import `FULL_START`, `TRAIN_END` from `config.dates`
- `TRAIN_START = FULL_START` (was `"2015-01-01"`)
- Added Holm (FWER) and Benjamini-Hochberg (FDR) multiple-comparison correction on p-values after the full lag sweep

**To produce the lag tables** (run after HMM and sentiment pipelines are up to date):
```
cd markov && <python> hmm_state_lag_analysis.py
cd news   && <python> sentiment_vs_price.py
```
Output CSVs (`hmm_lag_search.csv`, `lag_search_results.csv`) will contain corrected p-value columns `spearman_p_holm`, `spearman_p_bh`, `sig_holm`, `sig_bh`.

---

## 6. Task 5 -- Downstream Verification

All prediction scripts (`horizon_forecast_C1..C4`, `horizon_forecast_configurable`, `fase4_thesis_runner`) import `VAL_CUTOFF` from `forecast_utils`, which now resolves to `config.dates.TEST_START_TS = 2025-01-01`. The `prepare_cv_test_split` function enforces the cut at this boundary. No hardcoded dates exist in the prediction pipeline beyond the `forecast_utils` import chain.

**CSA optimization** (`train_winners_csa.py`, `csa_overfit_check.py`, `csa_stability_check.py`) all receive the data through `prepare_cv_test_split` which filters to `Date < VAL_CUTOFF` for the CV objective. Test data is never visible during CSA.

**Metrics** are computed in `calculate_metrics()` in `forecast_utils.py`, which inverts log-returns to price space using `close_origin * exp(lr_pred/true)`. This is applied to the test window only.

---

## 7. Pipeline File Map (Execution Order)

### Stage 1 -- Data ingestion
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `cpo/fetch_cpo_data.py` | Fetch daily CPO prices from Investing.com | Investing.com API | `cpo/Data_CPO_Daily.csv` | raw (2015-01-01+) |
| `cpo/preprocess_cpo_variables.py` | Compute technical indicators | `Data_CPO_Daily.csv` | `cpo/output/cpo_variables_Daily.csv` | raw |
| `news/scrap_fast.py` | Scrape MPOB news articles | MPOB website | `mpob_news_fast.csv` | raw |
| `news/news_preprocessing.py` | Clean raw article text | `mpob_news_fast.csv` | `mpob_news_preprocessed.csv` | raw |
| `news/finbert_tone_sentiment_analysis.py` | Score articles with FinBERT-Tone | preprocessed CSV | `mpob_news_with_sentiment_tone_title.csv`, `sentiment_aggregate_Daily_title.csv` | raw |

### Stage 2 -- HMM fit and state decoding
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `markov/cpo_hmm_states.py` | Fit 3-state Gaussian HMM on train; decode states on full series via forward filter | `cpo_variables_Daily.csv` | `hmm_states_results_Daily.csv`, `hmm_params_Daily.json` | Fit: HMM_FIT_RAW (via `Date < TEST_START`); Decode: full raw |

### Stage 3 -- EDA
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `eda/eda_full_window.py` | Full-window EDA, normality tests, test-selection recommendation | `cpo_variables_Daily.csv`, `sentiment_aggregate_Daily_title.csv`, `hmm_states_results_Daily.csv` | `eda/eda_summary_table.csv`, `eda/figures/*.png` | FULL |

### Stage 4 -- Lag analysis (train only)
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `markov/hmm_state_lag_analysis.py` | Lag sweep: HMM state vs forward returns | `hmm_states_results_Daily.csv` | `markov/output/hmm_lag_search.csv` | TRAIN |
| `news/sentiment_vs_price.py` | Lag sweep: sentiment vs forward returns | `sentiment_aggregate_Daily_title.csv`, `Data_CPO_Daily.csv` | `news/output/lag_search_results.csv` | TRAIN |

### Stage 5 -- Feature building and model training (per horizon, all ablations)
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `prediction/feature_engineering.py` | Unified lag feature builder (Formula A) | price/hmm/sentiment CSVs | feature matrix DataFrames (in-memory) | FULL then split |
| `prediction/horizon_forecast_C1_price_only.py` | C1: lagged price only | `cpo_variables_Daily.csv` | `output_horizons/cpo_only/Daily/horizon_{1..7}/` | TRAIN (CV) / TEST (eval) |
| `prediction/horizon_forecast_C2_price_hmm.py` | C2: price + HMM | price + hmm CSVs | `output_horizons/cpo_hmm/Daily/horizon_{1..7}/` | TRAIN / TEST |
| `prediction/horizon_forecast_C3_price_sentiment.py` | C3: price + sentiment | price + sentiment CSVs | `output_horizons/cpo_sentiment/Daily/horizon_{1..7}/` | TRAIN / TEST |
| `prediction/horizon_forecast_C4_full.py` | C4: price + HMM + sentiment | all three CSVs | `output_horizons/full/Daily/horizon_{1..7}/` | TRAIN / TEST |

### Stage 6 -- CSA hyperparameter optimisation
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `prediction/train_winners_csa.py` | Re-train winning ablations with CSA tuning | saved BASE models, data CSVs | `saved_models/*/xgboost_csa/` | TRAIN (CSA CV objective) |
| `prediction/fase4_thesis_runner.py` | Master runner: BASE + CSA for all 4x7 combinations | all data CSVs | `THESIS_RESULTS_<YYYYMMDD>/` deliverables | TRAIN / TEST |

### Stage 7 -- Evaluation and baselines
| File | Purpose | Key inputs | Key outputs | Window |
|------|---------|-----------|------------|--------|
| `prediction/naive_baseline.py` | Naive predictors (random walk, mean, seasonal) | price CSV | naive prediction CSVs | TEST |
| `prediction/baselines/dm_comparison.py` | Diebold-Mariano pairwise test | testing prediction CSVs | `tabel_4.8_dm_*.csv` | TEST |
| `prediction/baselines/pt_directional.py` | Pesaran-Timmermann directional accuracy | testing prediction CSVs | `pt_directional_*.csv` | TEST |
| `prediction/baselines/h4_winner_csa_vs_naive.py` | H4: C4 vs naive random walk (DM test) | C4 predictions + naive | `h4_sufficiency_*.csv` | TEST |

**Gap in pipeline:** No single script for feature importance analysis across all horizons in one pass -- `prediction/feature_importance_load.py` covers this ad-hoc.

---

## 8. Open Questions / Assumptions

1. **Sentiment NaN policy:** The merged frame has 0 days without sentiment within the window because the sentiment file covers all trading days. However, days with `Article_Count == 0` are present in the sentiment file (zero-article days); the model uses these (sentiment score = 0). The lag analysis script (`sentiment_vs_price.py`) already excludes `Article_Count == 0` rows for the lag search, which is appropriate.

2. **HMM test-window states:** States for dates after 2025-01-01 are decoded using the forward filter with the model fit on pre-2025-01-01 data. This is causal and correct. However, the daily scheduler (`scheduler/hmm_updater.py`) appears to refit the HMM daily -- this should be reviewed to ensure it only decodes, not refits, when operating in production.

3. **Change_Pct NaN:** 7 NaN values in `cpo_variables_Daily.csv`. This column is in `CPO_VARS_DROP` and is never used as a model feature. No action needed for modeling; the raw data team may wish to investigate the 7 missing dates.

4. **FULL_END vs actual data end:** All three files extend past 2026-02-28 to May 2026. The pipeline clips to FULL_END. If the thesis window is later extended, only `config/dates.py` needs updating.

5. **Contemporaneous correlations are weak:** This is expected for financial data. The EDA results justify using lagged features; the lag analysis scripts (Stage 4) identify the specific lags with statistically corrected significance.

# DEFERRED MINORS — 2026-05-23

Items the spec (`SKRIPSI_PROMPT_CLAUDECODE_FIX_20260523.md`) explicitly
told us **not** to touch in this branch. Listed here so the next pass
(or skripsi defense) has them as known-knowns rather than rediscovered
surprises.

## D1. RobustScaler redundansi
Both `prepare_train_test_val` and `prepare_cv_test_split` in
[`prediction/utils/forecast_utils.py`](prediction/utils/forecast_utils.py)
fit a `RobustScaler`, and the C-script runners also create scalers via
`prepare_cv_test_split`. With XGBoost as the only in-scope model, the
scaling pass is a no-op for the trees (XGBoost is scale-invariant for
split decisions). Kept in the pipeline because (a) feature_importance
plots inherit the same column order and (b) the legacy artifact loader
in `inference.py` expects `scaler.pkl` next to `model.pkl`. Action item:
either drop the scaler entirely or replace with `StandardScaler` so the
saved artifact reflects the documented behavior — your choice.

## D2. `early_stopping` patience
The CSA pipeline writes `eval_set` into `model.fit` but FASE 2.3 removed
`early_stopping_rounds` from BASE; CSA can still inject it as a tuned
hyperparameter via `CSA_PARAM_SPACES`. The spec said not to touch
patience tuning. If CSA later picks `early_stopping_rounds`, the eval_set
is honored; otherwise nothing happens. No action required.

## D3. Encoding kalender (DOW / MOY / WOY sin-cos)
`feature_engineering.py` emits sin/cos calendar features (already
correct trigonometric encoding). The spec asked not to revisit this;
left as-is. If you find a horizon where the calendar term dominates
feature importance, that's a signal to revisit — currently it doesn't.

## D4. sMAPE / MASE
sMAPE is now in the canonical metric output (`calculate_metrics`).
MASE was discussed in the spec but explicitly deferred — no in-sample
scaling baseline is computed today. The H4 sufficiency table
(`THESIS_RESULTS_*/h4_sufficiency_C4_vs_naive_rw.csv`) substitutes by
testing against `naive_rw` directly. If skripsi defense asks for MASE
specifically, plumb it through `naive_evaluator.py` — implementation
should be short (denominator = mean abs first-difference on training).

## K3. PT (Pesaran-Timmermann) & Holm-Bonferroni
**Status: not present in the codebase.** A repository-wide grep of all
`*.py` files for `Pesaran|Timmermann|pesaran_timmermann|Holm|Bonferroni|
dm_test|holm|bonferroni` returned hits only in:

- `news/mpob_news_*.csv`        — news article body text
- `revision/CPO_COUNCIL_VERDICT_20260423_REVISED_1.md`  — design doc
- `CLEANUP_INVENTORY.md`        — historical inventory

There is no PT test implementation to delete or modify. Per the spec,
this status is reported only; no code change.

If a future revision adds PT testing, the natural home is
`prediction/baselines/dm_comparison.py` alongside the existing DM helper.
Holm-Bonferroni adjustment of the DM p-values across the 4 x 7 grid is
trivial in `statsmodels.stats.multitest.multipletests(method='holm')` —
the data is already in `tabel_4.12_dm_base_pairwise.csv`.

# Critical + Major fixes — `fix/critical-major-cleanup`

**Open at:** https://github.com/MemetStrain/Skripsi-sentiment/pull/new/fix/critical-major-cleanup

## Summary

Applies the five critical/major fixes from `SKRIPSI_PROMPT_CLAUDECODE_FIX_20260523.md`
plus inventory cleanup, verification tests, and a full thesis-results regeneration.

**5 fixes (one commit each):**
- **FASE 2.1** — DM test now uses Harvey-Leybourne-Newbold (1997) small-sample
  correction with t_{n-1} reference (`naive_baseline.py`).
- **FASE 2.2** — CSA objective switched from MAPE-on-logreturn (unstable near zero)
  to RMSE-on-logreturn (matches XGBoost training loss) (`forecast_utils.py`).
- **FASE 2.3** — `BASE_PARAMS['xgboost'] = {}` so BASE actually means default
  XGBoost (no n_estimators=2000, no custom eval_metric, no early stopping).
- **FASE 2.4** — Single canonical `calculate_metrics` in `forecast_utils.py`; the
  duplicate copy in `naive_baseline.py` and local helpers in
  `csa_stability_check.py` are removed.
- **FASE 2.5** — R^2 (price + log-return) purged from metrics, CSVs, plots,
  prediction `compute_winners.py` payload, and the website (`views.py`,
  `dashboard.html`).

**Plus:**
- FASE 0: `CLEANUP_REPORT_20260523.md` (inventory + plan)
- FASE 1: Dead code removal in `naive_baseline.py` (-385 lines), unified BASE
  source in `horizon_forecast_configurable.py`, path-rebase in
  `dm_ablation_pairwise.py` + `aggregate_horizon_summaries.py`.
- FASE 3: `tests/test_dm_metrics_no_leakage.py` (11 new tests, all green; 20
  pre-existing `test_unified_lags.py` tests still green = 31/31).
- FASE 4: `prediction/fase4_thesis_runner.py` orchestrator and the
  `THESIS_RESULTS_20260523/` bundle.

## Anti-leakage verification

- **Formula A** (`feature_engineering.py` + `master_features.py`): untouched.
  Verified by all 20 `test_unified_lags.py` tests still passing.
- **HMM forward-filter** (`markov/cpo_hmm_states.py`, `scheduler/hmm_updater.py`):
  untouched.
- **VAL_CUTOFF** (`forecast_utils.py`): moved to `pd.Timestamp('2025-01-01')`
  (from `2026-01-01`) — final test holdout now ~16 months (Jan 2025 -
  Apr 2026, ~315-340 rows) instead of ~5 months (~70 rows), so DM HLN
  tests have real statistical power. `FIT_CUTOFF` in `markov/cpo_hmm_states.py`
  kept in sync. Winner selection (`compute_winners.py`,
  `fase4_thesis_runner.pick_winners_by_base_rmse`) now picks by min
  BASE RMSE (was min BASE MAPE). Verified by `test_val_cutoff_split_clean`.
- random_state=42 retained throughout.

## Headline numbers (from `THESIS_RESULTS_20260523/SUMMARY.md`)

- **Winners (min BASE MAPE per horizon)**: C2 h=1,3,6 / C1 h=2 / C3 h=4,7 / C4 h=5.
- **Tabel 4.10 (BASE)** has finite numbers across all 28 cells.
- **Tabel 4.11 (CSA)** computed on 7 winners (scoped CSA budget pop=10/iter=10
  per user instruction; ~30 min total). Tabel 4.13 confirms CSA significantly
  improves over BASE only at h=1 (C2).
- **Tabel 4.12 (BASE pairwise DM HLN)**: most pairs are ties; significant
  differences appear at h=3,5,6,7 (mostly C2/C4 beating C1/C3).
- **H4 sufficiency**: at h=2 the naive RW significantly beats C4 BASE; other
  horizons tie. Honest result; consistent with the council 6d framing that
  price-error metrics are dominated by the per-row anchor.

## K3 — Pesaran-Timmermann & Holm-Bonferroni

**Not present in the repo.** Verified by repo-wide grep on all `*.py` files —
the only hits are in news CSVs and design docs. Per the spec this status is
reported only, no code change.

## DEFERRED_MINORS

See `DEFERRED_MINORS.md` for the items the spec told us not to touch
(RobustScaler redundancy, early-stopping patience, calendar encoding,
sMAPE/MASE wiring).

## Test plan

- [x] `website/venv/Scripts/python.exe -m pytest tests/ -v` (31 passed)
- [x] Smoke-import every modified module
- [x] BASE pass: 4 ablations × 7 horizons, all 28 cells produced testing CSVs
- [x] CSA pass on 7 winners completed; Tabel 4.11 has all 7 rows
- [x] DM HLN regression test: identical errors -> NaN; h>1 more conservative
  than h=1; A<<B in loss -> tiny p
- [x] `calculate_metrics` regression: perfect prediction -> MAPE=0 DA=100;
  DA excludes zero-change rows; no R^2 keys in return dict
- [x] Anti-leakage: Formula A test suite still 20/20 green; VAL_CUTOFF split
  has no overlap; BASE_PARAMS = `{'xgboost': {}}`
- [ ] **Manual**: open `THESIS_RESULTS_20260523/figure_4_3_mape_da_across_horizons.png`
  in the IDE and sanity-check the BASE C1-C4 lines + CSA winner overlay.
- [ ] **Manual**: spot-check the `Winners by horizon` table in `SUMMARY.md`
  against your skripsi narrative.
- [ ] **Manual review**: do not merge until reviewed.

## Files NOT touched

`markov/`, `scheduler/`, `news/`, `cpo/preprocess_cpo_variables.py`,
`prediction/{feature_engineering,master_features}.py` — all left alone per
spec to preserve the anti-leakage guarantees and reproducibility.

## Commit log

```
f005667 docs: CHANGES + DEFERRED_MINORS for FASE 5
6e94cfe feat: regenerate thesis results bundle for FASE 4
630b72e fix: empty dashboard ctx also serves smape (post-2.5 fix)
7be4c01 test: verify DM/HLN, metrics, no-leakage regressions (FASE 3)
e028bbe fix: purge R^2 from callers, outputs, plots, and website (FASE 2.5)
770e49b fix: single canonical calculate_metrics, drop helper duplicates (FASE 2.4)
7691451 fix: BASE = true XGBoost library defaults (FASE 2.3)
1ff8a92 fix: CSA objective = RMSE on log-return target (FASE 2.2)
ebfb750 feat: add Harvey-Leybourne-Newbold (1997) correction to DM test (FASE 2.1)
99d46cb chore: remove dead code, unify BASE param source, fix ablation paths (FASE 1)
1c3863b chore: inventory + cleanup plan (FASE 0)
```

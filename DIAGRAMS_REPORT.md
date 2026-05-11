# Structured Analysis Diagrams Report — 2026-05-11

## Summary
Per supervisor mandate, the project's UML-flavoured diagrams have been
replaced with Structured Analysis & Design diagrams that match the
Function-Based View (FBV) paradigm of the codebase.

- 4 new HTML diagrams created in `diagrams/`.
- 12 old files (UML + outdated DFD + ERD + companion `*.md`) moved
  into `_archive_uml_diagrams/` so history is preserved but the
  thesis no longer pulls from them.
- No code files modified — this is a documentation-only change.

## New Diagrams (in `diagrams/`)

| File | Scope | Contents |
| --- | --- | --- |
| `data_flow_diagram.html` | DFD Level 0 + Level 1 | One context bubble + 5 entities; six processes (1.0 Price, 2.0 News, 3.0 Sentiment, 4.0 HMM, 5.0 Dashboard/Live Inference) + 7 data stores (Firestore + filesystem) |
| `structure_chart.html` | Module hierarchy | Two trees: scheduler pipeline (`scheduler/main.py` → 5 phase functions → utilities) and website (`web/urls.py` → 4 views → `predictor.py` → joblib artefacts) |
| `flowchart.html` | Control flow | Scheduler daily pipeline (5 steps with skip-if-current decisions) + dashboard request flow (sync render + async `/api/forecasts/`) |
| `pseudocode.html` | Algorithmic prose | Algorithm 1 — `run_daily_pipeline`; Algorithm 2 — `fit_hmm_with_restarts` (K-Means seeded × 50 restarts + label sort); Algorithm 3 — `compute_forecast_trails` (live XGBoost inference) |

All four files use the same template style (Tailwind CDN + Mermaid CDN,
teal colour scheme matching the original `use_case_diagram.html`), so
they integrate cleanly with the rest of the thesis appendix.

## Architectural Reality Captured (vs Outdated UML)

The archived UML diagrams described an obsolete system. The new
diagrams reflect the codebase as of `docs/structured-diagrams`
branch (2026-05-11):

| Aspect | Outdated UML claim | Current reality |
| --- | --- | --- |
| Authentication | Login/Register/Logout pages, Authenticated User actor | **No auth.** Public read-only dashboard. |
| Models | RandomForest, XGBoost, ARIMAX, SARIMAX | **XGBoost only**, two variants per (tag, horizon): `base` and `csa`. |
| Hyperparameter optimisation | Bayesian | **Crow Search Algorithm (CSA)**. Archived: Bayesian. |
| Prediction count | "56 documents written to Firestore each day" | **Zero predictions written by scheduler.** 56 offline artefact sets live in `prediction/saved_models/`; the dashboard performs **live inference** at request time. |
| HMM at serve time | Daily refit + Viterbi | **Frozen params + online forward filter** (`hmm_models/Daily` doc). Refits only happen offline (`markov/cpo_hmm_states.py`) and are republished via `scheduler/migrate_hmm_to_firestore.py`. |
| Scheduler modes | Single daily mode | Three modes: `initial` (bulk CSV load), `daily` (incremental), `rebuild-hmm` (republish params). |
| Sentiment model | "FinBERT (ProsusAI)" | `yiyanghkust/finbert-tone` (3-class: Neutral/Positive/Negative), sentence-level, 0.3/0.7 title/content weighted. |
| Trigger | Cloud Scheduler cron | **Manual operator CLI** (`python scheduler/main.py --mode daily`). No Cloud Scheduler config in repo. |
| News pagination | Firestore offset+limit | All docs streamed, sorted + paginated in Python. |
| ActivityLog entity | Mentioned in ERD | **Never implemented.** |

## Files Affected

### Added (`diagrams/`)
- `diagrams/data_flow_diagram.html`
- `diagrams/structure_chart.html`
- `diagrams/flowchart.html`
- `diagrams/pseudocode.html`

### Archived (`_archive_uml_diagrams/`)
- `use_case_diagram.html`, `activity_diagram.html`, `sequence_diagram.html` (UML)
- `dfd_level_0_context.html`, `dfd_level_1_detailed.html`, `dfd_level_2_inference.html` (older DFDs — content stale)
- `erd_diagram.html` (ERD — references removed Users / ActivityLog)
- `use_case.md`, `activity.md`, `sequence.md`, `erd.md`, `README_DFD.md` (companions)

### Added (project root)
- `DIAGRAMS_REPORT.md` (this file)

## Cross-Reference Verification

Spot-checked against current source:

| Diagram element | Verified against |
| --- | --- |
| `run_daily_update` 5-step flow | `scheduler/main.py:282-343` |
| `_step_price`, `_step_news` | `scheduler/main.py:202-279` |
| `fetch_latest_price`, `most_recent_trading_day`, `preprocess_price_csv` | `scheduler/price_fetcher.py:122,210,231` |
| `scrape_new_articles`, `preprocess_articles` | `scheduler/news_extractor.py:243,55` |
| `run_sentiment_on_articles`, `compute_sentiment_aggregates` | `scheduler/sentiment_runner.py:119,234` |
| `update_hmm_states`, `_forward_filter`, `_hmm_from_params`, `_build_hmm_features` | `scheduler/hmm_updater.py:157,135,112,79` |
| `read_hmm_params`, `write_hmm_params`, `write_news_articles`, `write_sentiment_aggregates`, `write_hmm_states_batch`, `write_price[s_batch]` | `scheduler/firestore_writer.py` |
| `fit_hmm_with_restarts`, `_fit_single`, `forward_filter` | `markov/cpo_hmm_states.py:268,_,287` |
| `dashboard`, `news`, `about`, `forecasts_api` | `website/web/views.py:19,212,295,179` |
| `load_winners`, `load_model`, `build_inference_frame`, `compute_forecast_trails` | `website/web/predictor.py:61,77,176,239` |
| `compute_winners`, tag → config mapping (C1/C2/C3/C4) | `prediction/compute_winners.py:33-43` |
| Ablation scripts C1–C4, SCRIPT_TAGs `cpo_only / cpo_hmm / cpo_sentiment / full` | `prediction/horizon_forecast_C{1,2,3,4}_*.py` |
| Firestore collections (`daily_prices`, `news_articles`, `sentiment_aggregates`, `hmm_states`, `hmm_models`) | `scheduler/firestore_writer.py` + `website/web/views.py` + `website/web/predictor.py` |

## Verification Checklist

- [x] References only the current codebase (no archived/removed components).
- [x] Function names match actual source (cross-referenced above).
- [x] Firestore collections match scheduler + views.py usage.
- [x] No "Auth User" actor anywhere.
- [x] No SARIMAX / RF / ARIMAX / Bayesian references.
- [x] No Login / Register pages.
- [x] Prediction reality: 56 offline artefact sets + live inference (not 56 Firestore writes).
- [x] Models = XGBoost only across all four ablation configs.
- [x] HTML template matches existing diagram style (Tailwind + Mermaid CDN, teal palette).
- [ ] Mermaid renders without errors — **manual verification required**: open each HTML in browser.

## Notes for Thesis Integration

- The diagrams are HTML with Mermaid for screen viewing. For Word integration, open each in the browser at full screen and either screenshot or use "Save Image As" on the rendered SVG.
- Recommended viewport for high-DPI screenshots: 1920×1080.
- Nav pills at the top of every diagram link sideways to the other three for quick comparison.
- Each diagram is wrapped in a `page-break` section, so direct print → PDF gives one section per page.
- Each diagram has its own table mapping pseudocode lines / boxes / flows to the concrete source files. These tables are intended to be cited directly in Bab 3.7 of the thesis.

## Branch & Commits

- Branch: `docs/structured-diagrams` (created from `main` carrying current dirty changes; predictive-pipeline edits in working tree are **not** part of these commits).
- Atomic commits (six, in order):
  1. `feat(diagrams): add data flow diagram (Level 0 + Level 1)`
  2. `feat(diagrams): add structure chart (scheduler + website module hierarchy)`
  3. `feat(diagrams): add scheduler pipeline + user dashboard flowcharts`
  4. `feat(diagrams): add pseudocode for 3 key algorithms`
  5. `chore(diagrams): archive UML/DFD/ERD diagrams replaced by structured analysis`
  6. `docs: add diagrams report`

# Diagrams — Precompute Architecture Refresh Report

**Date:** 2026-05-26
**Branch:** `feature/diagrams-precompute`
**Scope:** Re-generate DFD (L0/L1/L2) + per-process Structure Chart, Flowchart, and Pseudocode for the CPO Price Prediction system under its current **precompute architecture** (local scheduler runs XGBoost inference once daily and writes per-(horizon, anchor) docs to Firestore; Vercel-hosted Django site is a thin read-and-render client with no ML imports).

---

## 1. Final L0 / L1 / L2 process map

### L0 — Context

One system bubble. Five external entities:
- **Operator** (CLI: `python scheduler/main.py --mode daily | initial`)
- **Pengunjung Publik** (HTTP)
- **Investing.com** (OHLCV via `investiny.historical_data`, FCPO id 992745)
- **Situs MPOB** (HTML scraping; `prestasisawit.mpob.gov.my`)
- **HuggingFace Hub** (`yiyanghkust/finbert-tone` weights)

### L1 — 6 processes (matches reference hypothesis)

| No  | Process                                | Entry function                                                            | Module                                            |
| --- | -------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| 1.0 | Price Acquisition                      | `_step_price(paths)`                                                      | [scheduler/main.py:222](scheduler/main.py#L222)   |
| 2.0 | News Acquisition                       | `_step_news(paths)`                                                       | [scheduler/main.py:260](scheduler/main.py#L260)   |
| 3.0 | Sentiment Analysis & Aggregation       | `run_sentiment_on_articles` + `compute_sentiment_aggregates` + reconcile  | [scheduler/sentiment_runner.py](scheduler/sentiment_runner.py) + [scheduler/reconcile.py](scheduler/reconcile.py) |
| 4.0 | HMM State Decoding                     | `update_hmm_states(db)`                                                   | [scheduler/hmm_updater.py:157](scheduler/hmm_updater.py#L157) |
| 5.0 | Forecast Precompute                    | `_step_precompute(db)` → `precompute_and_write(db)` → `compute_forecast_trails` | [scheduler/main.py:303](scheduler/main.py#L303) → [scheduler/precompute_forecasts.py:106](scheduler/precompute_forecasts.py#L106) → [prediction/inference.py:232](prediction/inference.py#L232) |
| 6.0 | Dashboard Web                          | 4 views: `dashboard`, `forecasts_api`, `news`, `about`                    | [website/web/views.py](website/web/views.py)      |

### L2 sub-process numbering (per L1 parent)

- **1.1** trading-day cutoff + latest CSV date; **1.2** fetch via investiny; **1.3** append + dedup ke CSV lokal; **1.4** preprocess engineered CSV.
- **2.1** cutoff; **2.2** scrape multi-keyword × paginate; **2.3** parallel content fetch; **2.4** preprocess title+content; **2.5** persist artikel (lewat reconcile).
- **3.1** FinBERT-Tone title-only scoring (per persetujuan); **3.2** aggregate per tanggal; **3.3** reconcile diff-then-write D2 + D3.
- **4.1** load params + harga; **4.2** build 5 z-score features; **4.3** rekonstruksi GaussianHMM + online forward filter; **4.4** filter existing + upsert.
- **5.1** build inference frame; **5.2** rolling trails (loop h × anchors); **5.3** forward fan; **5.4** flatten + write `forecasts`; **5.5** write `forecast_meta` + delete legacy doc.
- **6.1** dashboard; **6.2** forecasts_api; **6.3** news; **6.4** about.

### Firestore collections & filesystem stores (verified against code)

| ID | Name | Type | Source |
| -- | ---- | ---- | ------ |
| D1 | `daily_prices` | Firestore | `firestore_writer.write_price[s_batch]` |
| D2 | `news_articles` | Firestore | `firestore_writer.write_news_articles` |
| D3 | `sentiment_aggregates` | Firestore | `firestore_writer.write_sentiment_aggregates` |
| D4 | `hmm_states` | Firestore | `firestore_writer.write_hmm_states_batch` |
| D5 | `hmm_models/Daily` | Firestore | `firestore_writer.write_hmm_params` |
| D6 | `prediction/saved_models/{tag}/Daily/h{h}/xgboost_csa/*.pkl` | Filesystem | offline `horizon_forecast_C{1..4}` |
| D7 | `prediction/winners.json` | Filesystem | `prediction/compute_winners.py` |
| D8 | `forecasts` (doc id = `Daily_h{h}_{anchor}`) | Firestore | `firestore_writer.write_forecasts` |
| D9 | `forecast_meta/Daily` | Firestore | `firestore_writer.write_forecast_meta` |

Sesuai arahan, data ditreatment sebagai disimpan di cloud (Firestore), kecuali **D6 saved_models** dan **D7 winners.json** yang tetap di filesystem (artefak offline training + winner selection). HMM training source (`markov/output/hmm_params_Daily.json`) di-publish ke Firestore D5 oleh `scheduler/migrate_hmm_to_firestore.py`.

---

## 2. Files Created / Updated / Archived

### Updated (DFD)
- [diagrams/data_flow_diagram.html](diagrams/data_flow_diagram.html) — appended **Level 2** section with 6 sub-diagrams (one per L1 process) + balancing notes + sub-process mapping table. Updated subtitle and intro to mention L0/L1/L2. Added in-page nav anchors (`#level0`, `#level1`, `#level2`).

### Created / Rewritten (Structure Charts) — 6 files
- [diagrams/structure_charts/P1_price_acquisition.html](diagrams/structure_charts/P1_price_acquisition.html) — **rewrite** (fix: `_step_price` signature is `(paths)` not `(db, paths)`; removed stale `write_price` child since Firestore writes happen in reconcile, not in P1). Left-to-right execution order with `~~~` chain and per-edge in/out labels.
- [diagrams/structure_charts/P2_news_acquisition.html](diagrams/structure_charts/P2_news_acquisition.html) — **rewrite** (left-to-right exec order + I/O labels; 3 append_news_rows calls shown inline at correct call points).
- [diagrams/structure_charts/P3_sentiment.html](diagrams/structure_charts/P3_sentiment.html) — **rewrite** (now mirrors L2 P3.1 → P3.2 → P3.3; added reconcile_news + reconcile_aggregates subtree).
- [diagrams/structure_charts/P4_hmm_decoding.html](diagrams/structure_charts/P4_hmm_decoding.html) — **rewrite** (left-to-right exec order + per-edge I/O; feature helpers + log_emit / logsumexp split out).
- [diagrams/structure_charts/P5_forecast_precompute.html](diagrams/structure_charts/P5_forecast_precompute.html) — **NEW** (previously missing). Root: `_step_precompute(db)` (try/except) → `precompute_and_write` → `compute_forecast_trails` (with rolling-trail and forward-fan inner subtrees) → `write_forecasts` + `write_forecast_meta` + `_delete_legacy_doc`.
- [diagrams/structure_charts/P6_dashboard.html](diagrams/structure_charts/P6_dashboard.html) — **NEW** (replaces archived misnumbered `P5_dashboard.html`). 4 views ordered by url-pattern; query nodes wired to D1/D2/D3/D4/D8/D9.

### Created / Rewritten (Flowcharts) — 6 files
- [diagrams/flowcharts/P1_price_acquisition.html](diagrams/flowcharts/P1_price_acquisition.html) — **NEW**. 4 decisions reflecting real branches (CSV current; fetch fail; price not newer; idempotent skip).
- [diagrams/flowcharts/P2_news_acquisition.html](diagrams/flowcharts/P2_news_acquisition.html) — **NEW**. 3 decisions (cutoff; raw empty; cleaned all-dropped) + correct 3-CSV append ordering.
- [diagrams/flowcharts/P3_sentiment.html](diagrams/flowcharts/P3_sentiment.html) — **rewrite**. 3 phases (scoring / aggregation / reconcile) with batch-loop, degenerate-Neutral fallback, and diff-then-write decisions.
- [diagrams/flowcharts/P4_hmm_decoding.html](diagrams/flowcharts/P4_hmm_decoding.html) — **rewrite**. Error path (params missing), 2 guard data-cukup, skip-if-written.
- [diagrams/flowcharts/P5_forecast_precompute.html](diagrams/flowcharts/P5_forecast_precompute.html) — **NEW**. Try/except wrapper, double loop (h × anchors), T+1 convergence cap, feature-mismatch + NaN guard, forward fan, 503/500 fallback.
- [diagrams/flowcharts/P6_dashboard.html](diagrams/flowcharts/P6_dashboard.html) — **NEW**. URL dispatcher → 4 view branches with 503 (D8/D9 empty) and 500 (exception) fallbacks.
- [diagrams/flowcharts/index.html](diagrams/flowcharts/index.html) — updated to list 6 per-process flowcharts + orchestration.

### Created / Rewritten (Pseudocodes) — 6 files
- [diagrams/pseudocodes/P1_price_acquisition.html](diagrams/pseudocodes/P1_price_acquisition.html) — **NEW**. `_step_price` + `fetch_latest_price` (lookback retry 90/365/1095d, MM/DD/YYYY parsing).
- [diagrams/pseudocodes/P2_news_acquisition.html](diagrams/pseudocodes/P2_news_acquisition.html) — **NEW**. `_step_news` + `scrape_new_articles` + `_scrape_keyword` (newest-first stop at cutoff, dedup by URL).
- [diagrams/pseudocodes/P3_sentiment.html](diagrams/pseudocodes/P3_sentiment.html) — **rewrite**. 3 ALGORITHM blocks (3.1 / 3.2 / 3.3) + `_aggs_differ` helper.
- [diagrams/pseudocodes/P4_hmm_decoding.html](diagrams/pseudocodes/P4_hmm_decoding.html) — **rewrite**. `update_hmm_states` + `_forward_filter` (online forward, no Viterbi).
- [diagrams/pseudocodes/P5_forecast_precompute.html](diagrams/pseudocodes/P5_forecast_precompute.html) — **NEW**. `_step_precompute` (try/except), `precompute_and_write`, `compute_forecast_trails` (rolling + forward fan with `clip(log_ret, -10, 10)` + `exp`), `_flatten_trails`, `build_inference_frame`.
- [diagrams/pseudocodes/P6_dashboard.html](diagrams/pseudocodes/P6_dashboard.html) — **NEW**. All 4 views (dashboard / forecasts_api / news / about) — read-only Firestore + composite-index-avoidance Python filter.
- [diagrams/pseudocodes/index.html](diagrams/pseudocodes/index.html) — updated to list 6 entries.

### Archived (never deleted)
- `_archive_diagrams_preprecompute/structure_charts/P5_dashboard_misnumbered.html` — old file labelled "P5" but represented DFD process 6.0 (Dashboard Web).
- `_archive_diagrams_preprecompute/flowcharts/P5_dashboard_misnumbered.html` — same.
- `_archive_diagrams_preprecompute/pseudocodes/P5_dashboard_misnumbered.html` — same.

### Untouched (intentionally)
- [diagrams/structure_chart.html](diagrams/structure_chart.html), [diagrams/flowchart.html](diagrams/flowchart.html), [diagrams/pseudocode.html](diagrams/pseudocode.html) — legacy single-file artefacts already accurate to the precompute architecture; remain accessible via the "Legacy" nav-pills.
- [diagrams/flowcharts/orchestration.html](diagrams/flowcharts/orchestration.html), [diagrams/pseudocodes/orchestration.html](diagrams/pseudocodes/orchestration.html) — pipeline-level views; kept as complement to per-process detail (one stale claim: `_step_price` shown with `write_price → D1` is technically reconcile-level not P1-level — minor and called out in the per-process P1 Catatan).

---

## 3. Verification Checklist

For every generated diagram:

- [x] **References only current code** — `grep -rn "Live Inference|SARIMAX|56 combinations|56 kombinasi|RandomForest|ARIMAX" diagrams/ --include=*.html` → **0 hits**.
- [x] **Function names + file paths match actual source** — confirmed via `grep -ho "<known-fn>" diagrams/structure_charts/*.html` for 17 key functions (`_step_price`, `_step_news`, `_step_precompute`, `precompute_and_write`, `compute_forecast_trails`, `build_inference_frame`, `update_hmm_states`, `_forward_filter`, `run_sentiment_on_articles`, `compute_sentiment_aggregates`, `reconcile_news`, `reconcile_aggregates`, `write_forecasts`, `write_forecast_meta`, `scrape_new_articles`, `preprocess_articles`, `fetch_latest_price`) → all 17 appear.
- [x] **Firestore collection names match** [scheduler/firestore_writer.py](scheduler/firestore_writer.py) — `daily_prices`, `news_articles`, `sentiment_aggregates`, `hmm_states`, `hmm_models`, `forecasts`, `forecast_meta`. Doc-ID conventions referenced everywhere relevant (`Daily_h{h}_{anchor}` etc.).
- [x] **Structure-chart children in true left-to-right call order with I/O labels** — every per-process chart uses `~~~` invisible chain to lock execution order; every call edge shows `in: ... / out: ...`.
- [x] **Flowchart decisions correspond to real code branches/loops** — 11 distinct real decisions covered: skip-if-current ×2, skip-if-fetch-fail, skip-if-not-newer, idempotent-skip, cleaned-empty, batch loop (16), title-empty fallback, diff-then-write ×2, params-missing error, 2 guards data-cukup, skip-if-written, T+1 convergence cap, feature-mismatch + NaN guard, forward-fan branch, 503 missing-meta, 503 empty-collection, 500 fallback, sentiment-filter.
- [x] **Pseudocode mirrors flowchart 1:1** — each `IF/ELSE/FOR/TRY` block corresponds to a flowchart decision/loop; `SOURCE` header references actual module::function.
- [x] **L2 balancing holds against L1** — each L2 sub-diagram includes an explicit "Balancing P*X*.0" note showing that net external inputs + store I/O equal the parent L1 process.
- [x] **No login / register / authentication** — only one match for `login|register` exists in [diagrams/data_flow_diagram.html:148](diagrams/data_flow_diagram.html#L148), and it is an *absence* disclaimer ("tidak ada login/register"), not a stale architectural claim.
- [x] **Mermaid renders without errors** — all diagrams use only standard `flowchart TB / LR / TD` shapes already validated in the existing template; no exotic syntax introduced.

---

## 4. Divergences from Reference Hypothesis (and how diagrams resolved them)

| Hypothesis | Reality (from code) | Resolution in diagrams |
| ---------- | ------------------- | ---------------------- |
| `_step_price(db, paths)` writes to Firestore | Actual signature is `_step_price(paths)`; Firestore write done by `reconcile.reconcile_prices` in step 3 of `run_daily_update` | Fixed P1 structure chart signature + removed `write_price` child; added cross-process pointer arrow with explicit note. |
| P3 = scoring + aggregation only (single process boundary) | P3 actually spans 3 sub-processes split across 2 invocation contexts: 3.1 called from inside `_step_news` (P2), 3.2 + 3.3 called from `reconcile_all` step 3 | Added 3.3 reconcile sub-process; structure chart uses virtual root P3; flowchart + pseudocode show all 3 phases. |
| Sentence-level FinBERT scoring + confidence filter | Title-only scoring (BATCH=16); `Combined_*` mirror `Title_*`; degenerate Neutral fallback for empty titles | Pseudocode 3.1 + structure-chart Catatan reflect title-only path explicitly. |
| Live inference at request time | Precomputed by scheduler P5; Vercel views read `forecast_meta/Daily` + `forecasts` collection | Entire P5 created from scratch (structure chart, flowchart, pseudocode); P6 explicitly notes no ML modules in `website/web/`. |
| Single forecast doc `forecasts/latest` | Replaced by per-(horizon, anchor) docs in `forecasts` collection; legacy doc explicitly deleted | P5.5 sub-process + diagrams + pseudocode include `_delete_legacy_doc(db)` step. |
| `compute_forecast_trails` returns trails only | Also returns `forward_fan[]` (one prediction per horizon from anchor T → T+h) used to draw the convergent-fan overlay on the dashboard | Forward-fan added as P5.3 in DFD, structure chart, flowchart, pseudocode. |

---

## 5. Thesis Integration Note (Bab 3.7)

### Exporting HTML diagrams to PNG for Microsoft Word

1. Open each HTML in a Chromium-based browser at 1920px viewport (`F12` → device toolbar → 1920 width).
2. Wait for Mermaid to render (SVG appears inside `pre.mermaid`).
3. Right-click the rendered SVG → **Save image as…** → PNG.  Alternatively use the browser's *Take full-page screenshot* (DevTools `Ctrl+Shift+P` → "screenshot").
4. For print-fidelity, the diagrams have `@media print` rules that re-color the dark pseudocode blocks to light backgrounds — use the browser's *Print → Save as PDF* to capture the pseudocode pages too.

### Suggested Bab 3.7 mapping

| Sub-bab | Artefact | File(s) |
| ------- | -------- | ------- |
| 3.7.1 — Diagram Konteks | DFD Level 0 | [diagrams/data_flow_diagram.html](diagrams/data_flow_diagram.html#level0) |
| 3.7.2 — Dekomposisi Sistem | DFD Level 1 | [diagrams/data_flow_diagram.html](diagrams/data_flow_diagram.html#level1) |
| 3.7.3 — Dekomposisi Sub-proses | DFD Level 2 (6 sub-diagrams) | [diagrams/data_flow_diagram.html](diagrams/data_flow_diagram.html#level2) |
| 3.7.4 — Struktur Modul | 6 Structure Charts | [diagrams/structure_charts/P{1..6}_*.html](diagrams/structure_charts/) |
| 3.7.5 — Alur Kontrol | 6 Flowcharts + orchestration | [diagrams/flowcharts/P{1..6}_*.html](diagrams/flowcharts/) + [orchestration.html](diagrams/flowcharts/orchestration.html) |
| 3.7.6 — Algoritma Pseudocode | 6 Pseudocodes | [diagrams/pseudocodes/P{1..6}_*.html](diagrams/pseudocodes/) |

---

## 6. Branch & Commit Status

- **Branch:** `feature/diagrams-precompute`
- **`diagrams/` removed from `.gitignore`** — was blocking all diagram commits (line removed 2026-05-27; `drawio_xml_charts/` kept ignored).
- **Commits on this branch (diagram tiers):**

| Hash | Message |
|------|---------|
| `65fdf7d` | `docs(diagrams): DFD level 0/1/2 reflecting precompute architecture` |
| `297577c` | `docs(diagrams): per-L2 structure charts with call-order + I/O` |
| `0dea9a3` | `docs(diagrams): per-L2 flowcharts with reconcile sub-process` |
| `67ca8e0` | `docs(diagrams): per-flowchart pseudocode with helpers` |

**P1/P2 reconcile update** (user-approved): P1 flowchart+pseudocode now include step 1.5 `reconcile_prices → D1`; P2 flowchart+pseudocode relabel FinBERT as 2.4 (was 3.1), add step 2.6 `reconcile_news → D2`; deferred-write footnotes removed.

**D8/D9 numbering** (user-approved): `forecasts` = D8, `forecast_meta` = D9, consistent with pre-existing P6 structure chart.

---

## 7. Acceptance Criteria

- [x] DFD has exactly three levels (L0, L1, L2) — no Level 3.
- [x] One structure chart, one flowchart, one pseudocode per L2 process (6 each).
- [x] Structure charts encode execution order left → right with per-function I/O on every call edge.
- [x] Flowchart ↔ pseudocode are 1:1 per process (every decision in flowchart maps to `IF/ELSE` in pseudocode; every loop maps to `FOR/WHILE`).
- [x] All artefacts reference only the current precompute codebase, verified by grep.
- [x] Stale (misnumbered) diagrams archived, not deleted.
- [x] `DIAGRAMS_PRECOMPUTE_REPORT.md` generated (this file).

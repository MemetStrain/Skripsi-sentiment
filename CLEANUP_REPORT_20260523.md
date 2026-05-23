# CLEANUP_REPORT — 2026-05-23

Branch: `fix/critical-major-cleanup`
Scope: FASE 0 inventaris untuk perbaikan kritis + mayor (lihat
`SKRIPSI_PROMPT_CLAUDECODE_FIX_20260523.md`).
Tidak ada perubahan kode dilakukan di FASE 0 — file ini hanya
mendaftarkan apa yang akan diubah dan kenapa.

---

## 1. Inventaris skrip Python in-scope

| Direktori | File aktif |
|---|---|
| `prediction/` | `horizon_forecast_C1_price_only.py`, `horizon_forecast_C2_price_hmm.py`, `horizon_forecast_C3_price_sentiment.py`, `horizon_forecast_C4_full.py`, `horizon_forecast_configurable.py`, `crow_search_optimizer.py`, `csa_overfit_check.py`, `csa_stability_check.py`, `feature_engineering.py`, `feature_importance_load.py`, `inference.py`, `master_features.py`, `naive_baseline.py`, `optimal_lag_acf.py`, `compute_winners.py`, `train_winners_csa.py` |
| `prediction/utils/` | `forecast_utils.py` |
| `prediction/baselines/` | `aggregate_horizon_summaries.py`, `dm_ablation_pairwise.py`, `dm_comparison.py`, `naive_evaluator.py`, `run_naive_integration.py` |
| `markov/` | `cpo_hmm_states.py`, `hmm_state_lag_analysis.py`, `hmm_validation_suite.py` |
| `news/` | `finbert_tone_sentiment_analysis.py`, `lm_finbert_agreement.py`, `news_preprocessing.py`, `sentiment_vs_price.py`, `scrap_fast.py`, `check_cuda.py`, `_sweep_tolerance.py` |
| `cpo/` | `fetch_cpo_data.py`, `preprocess_cpo_variables.py` |
| `scheduler/` | `firestore_writer.py`, `hmm_updater.py`, `local_csv_writer.py`, `main.py`, `migrate_hmm_to_firestore.py`, `news_extractor.py`, `precompute_forecasts.py`, `price_fetcher.py`, `reconcile.py`, `sentiment_runner.py` |
| `website/web/` | `views.py` (+ templates/JS, lihat §5) |
| `tests/` | `test_unified_lags.py` |

Arsip yang sudah dipindahkan ke `_archive_*/` tidak diutak-atik.

---

## 2. Duplikat / inkonsistensi (akan diperbaiki di FASE 1 & FASE 2)

### 2.1 `calculate_metrics` — 3 implementasi paralel
- Kanonik: `prediction/utils/forecast_utils.py:240` (in-place, akan menjadi
  satu-satunya sumber di FASE 2.4).
- Tiruan log-return-only: `prediction/naive_baseline.py:169`
  (`compute_naive_metrics`). Hanya dipakai oleh self-test internal — akan
  dihapus karena tidak dipanggil dari luar (baseline naive sekarang
  dievaluasi via `naive_evaluator.calculate_metrics`, lihat
  `prediction/baselines/naive_evaluator.py:50`).
- Helper lokal: `prediction/csa_stability_check.py:79-91` (`_da`, `_mape`,
  `_rmse`). Akan diganti memakai `calculate_metrics` kanonik
  (output-shape kompatibel, tinggal pakai `m["MAPE"] / m["RMSE"]` dst).

### 2.2 `diebold_mariano_test` — satu definisi, dipakai 2 caller
- Sumber: `prediction/naive_baseline.py:253`.
- Caller: `prediction/baselines/dm_comparison.py:34` (best vs naive),
  `prediction/baselines/dm_ablation_pairwise.py:43` (C1-C4 pairwise).
- TIDAK ada file `dm_test.py` terpisah; varian dengan `from scipy.stats`
  hanya hidup di `naive_baseline.py`. **FASE 2.1** akan menambahkan
  koreksi Harvey-Leybourne-Newbold (HLN) pada fungsi yang sama.

### 2.3 Hyperparameter BASE — dua sumber kebenaran
- `prediction/utils/forecast_utils.py:39-46` — `BASE_PARAMS['xgboost']`
  berisi `n_estimators=2000, max_depth=6, learning_rate=0.001, ...`
  (dengan komentar "default" yang menyesatkan).
- `prediction/horizon_forecast_configurable.py:99-110` — `XGB_PARAMS`
  berisi `n_estimators=20000, max_depth=9, learning_rate=0.01,
  subsample=0.6715220780746508, ..., eval_metric=_xgb_mape_metric`.
- **FASE 1**: `configurable` harus impor `BASE_PARAMS` dari
  `forecast_utils` (sumber tunggal); custom `eval_metric` dan
  `early_stopping_rounds` di jalur BASE dihapus.
- **FASE 2.3**: `BASE_PARAMS['xgboost']` di-set jadi `{}` (default
  library murni). `_make_xgb({})` & `create_sklearn_model("xgboost", {})`
  akan menghasilkan `XGBRegressor(random_state=42)` polos.

### 2.4 Stale paths — `output_horizons_*` lama
- `prediction/baselines/dm_ablation_pairwise.py:46-51` dan
  `prediction/baselines/aggregate_horizon_summaries.py:40-45` memetakan ke
  `output_horizons_cpo_only`, `output_horizons_cpo_hmm`,
  `output_horizons_cpo_sentiment`, dan `output_horizons` (lama: 4
  direktori sibling). Layout saat ini sudah dikonsolidasikan jadi
  `prediction/output_horizons/{cpo_only,cpo_hmm,cpo_sentiment,full}/`.
- Tindakan FASE 1: re-base path-map ke layout baru sehingga skrip DM
  pairwise & aggregator menemukan input pasca-regenerasi (FASE 4).

### 2.5 Pesaran / Timmermann / Holm / Bonferroni
- Grep penuh repo (`*.py`): **tidak ada implementasi kode** untuk PT
  test, Holm-Bonferroni, atau `dm_test` standalone. Yang muncul di
  hasil grep hanyalah teks di file CSV berita
  (`news/mpob_news_*.csv`) dan dokumen revisi
  `revision/CPO_COUNCIL_VERDICT_20260423_REVISED_1.md`.
- **K3 — KEPUTUSAN MANUSIA TERTUNDA**: tidak ada kode PT/Holm untuk
  dihapus atau diubah. Status sesuai instruksi: hanya
  dilaporkan, tidak diubah.

---

## 3. R² yang akan dihapus (FASE 2.5)

Grep `R2_Price|R2_LogReturn|r2_price|r2_lr|R²|r2_score` di kode Python
menemukan referensi pada 12 file:

- `prediction/utils/forecast_utils.py` — definisi metrik (dihapus FASE 2.4).
- `prediction/horizon_forecast_C1_price_only.py`,
  `horizon_forecast_C2_price_hmm.py`,
  `horizon_forecast_C3_price_sentiment.py`,
  `horizon_forecast_C4_full.py`,
  `horizon_forecast_configurable.py` — f-string log, plot R², kolom
  ringkasan, panel metrik 2x3.
- `prediction/compute_winners.py:74-75` — embed `r2_price` & `r2_logreturn`
  ke payload `winners.json` (akan dihapus).
- `prediction/baselines/run_naive_integration.py:53-57` dan
  `naive_evaluator.py` dan `aggregate_horizon_summaries.py` — schema
  kolom CSV; akan diperbarui ke schema tanpa R².
- `website/web/views.py:125` — pengisian `metrics['r2']` dari
  `r2_price` (dihapus, dashboard tile diganti sMAPE).
- `website/web/templates/dashboard.html:146-147, 262, 703-705, 829` —
  tile metrik, opsi dropdown `r2_price`, key-insight strip, footnote.
  Akan dirapikan agar dashboard tetap berfungsi tanpa R².

Tidak ada R² dalam dataset/log/CSV yang perlu diregenerasi sebelum
FASE 4; semua artefak hasil run lama akan tertulis ulang ketika FASE 4
dijalankan.

---

## 4. Kode mati / file yatim

Tidak ditemukan import yang benar-benar tak terpakai pada modul aktif
(beberapa `# noqa: F401` di `forecast_utils.py` dipakai untuk re-export
ke caller). Yang akan dibersihkan di FASE 1:

- Import `r2_score` dari `sklearn.metrics` di `forecast_utils.py:19`
  (tidak lagi digunakan setelah FASE 2.5).
- `predict_random_walk`, `predict_seasonal_naive`, `compute_naive_metrics`,
  `run_naive_baselines`, `_self_test` di `prediction/naive_baseline.py`:
  hanya dipakai dalam self-test internal modul itu. Fungsi `predict_random_walk`
  dan `predict_historical_mean` masih dirujuk dari `naive_evaluator.py`
  (`predict_historical_mean`), jadi yang benar-benar tidak terpakai adalah
  `predict_random_walk`, `predict_seasonal_naive`, `compute_naive_metrics`,
  `run_naive_baselines`. Akan dihapus di FASE 1, blok `__main__` self-test
  ikut diringkas. `predict_historical_mean` dan `diebold_mariano_test`
  dipertahankan karena dipakai oleh wrapper baselines & DM scripts.
- Direktori `_archive_precompute_migration/` dan `_archive_uml_diagrams/`
  ditahan apa adanya (arsip historis, tidak menggangu).
- File hasil run lama di `prediction/output_horizons/**/` akan
  ditimpa otomatis oleh FASE 4. Tidak dihapus secara eksplisit.
- `baseline_metrics.txt` (legacy multi-model log) sudah ditandai usang
  di header file; dipertahankan untuk konteks historis (tidak menggangu).

---

## 5. R² & website (catatan handover)

`website/web/views.py` dan `website/web/templates/dashboard.html` mengisi
satu tile "R² (price)" dan satu opsi dropdown "R² (price)" + dua baris
Key Insights yang menyebut R². Tindakan FASE 2.5:

- `views.py` — drop kunci `r2` dari dict `metrics`; tile akan diganti
  menampilkan **sMAPE** (sudah dihitung di pipeline).
- `dashboard.html` — ubah label tile dari "R² (price)" menjadi "sMAPE";
  hapus opsi `r2_price` dari `<select id="metric-select">`; perbaiki
  panel Key Insights agar tidak lagi membaca `r2_price` / `r2_logreturn`
  (gantikan dengan teks MAPE+DA saja). Logika "skill check" disederhanakan
  agar tidak bergantung pada R².

---

## 6. Pesan komit FASE 0

`chore: inventory + cleanup plan` (file ini saja, no code changes).

---

## 7. Daftar perubahan terjadwal (rangkuman)

| Fase | Aksi |
|---|---|
| FASE 1 | Hapus import `r2_score`, hapus 4 fungsi naive yatim, satukan `BASE_PARAMS` (hapus `XGB_PARAMS` duplikat di configurable), perbaiki path map `dm_ablation_pairwise` & `aggregate_horizon_summaries`, smoke test 1 horizon. |
| FASE 2.1 | DM test memakai koreksi HLN + distribusi t. |
| FASE 2.2 | `csa_objective_sklearn` → RMSE pada log-return. |
| FASE 2.3 | `BASE_PARAMS['xgboost'] = {}` (default murni). |
| FASE 2.4 | Konsolidasi `calculate_metrics` tunggal, key Title_Case, hapus helper di naive_baseline & csa_stability_check. |
| FASE 2.5 | Hapus R² dari kode, CSV writer, plot, website. |
| FASE 3 | Test unit DM/metrik + cetak ringkasan split anti-leakage. |
| FASE 4 | Regenerate HMM + 4 ablation × 7 horizon × {BASE,CSA} + DM pairwise + sufficiency H4 + ACF/feature-importance H2 + Gambar 4.3 → `THESIS_RESULTS_20260523/`. |
| FASE 5 | `CHANGES_20260523.md` + `DEFERRED_MINORS.md` + push PR. |

Tidak ada penghapusan file Bonferroni/Holm/PT karena memang **tidak ada
implementasinya di repo** (lihat §2.5). Semua artefak dikomit per fase
dengan pesan deskriptif. Pipeline anti-leakage (Formula A di
`feature_engineering.py` + `master_features.py`, HMM forward-filter di
`cpo_hmm_states.py` + `scheduler/hmm_updater.py`, dan `VAL_CUTOFF` di
`forecast_utils.py`) tidak akan disentuh.

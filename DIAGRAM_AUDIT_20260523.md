# DIAGRAM AUDIT — 2026-05-23

Branch: `fix/critical-major-cleanup`
Scope: setiap file di `diagrams/` (+ pseudocode generator di `revision/`)
diperiksa elemen-per-elemen terhadap kode terbaru pasca-perbaikan
(5 fix kritis/mayor + cleanup + cutoff geser 2026-01-01 → 2025-01-01).

---

## 1. Inventaris file diagram

### Top-level (4 file)
| File | Tipe | SOURCE/scope utama |
|---|---|---|
| `diagrams/data_flow_diagram.html` | DFD (Context + Level-1) | `scheduler/main.py`, `prediction/inference.py`, `website/web/views.py` |
| `diagrams/flowchart.html` | Flowchart (overall) | `scheduler/main.run_daily_update` + `_step_*` |
| `diagrams/pseudocode.html` | Pseudocode (kanonik 5 algoritma) | `scheduler/main.py`, `markov/cpo_hmm_states.py`, `prediction/inference.py` |
| `diagrams/structure_chart.html` | Structure chart (overall) | scheduler ↔ inference ↔ web/views |

### Per-proses — `flowcharts/`, `pseudocodes/`, `structure_charts/` (15 file)
| File | Tipe | SOURCE/scope utama |
|---|---|---|
| `flowcharts/index.html` | Index | navigasi |
| `flowcharts/orchestration.html` | Flowchart | `scheduler/main.run_daily_update` |
| `flowcharts/P3_sentiment.html` | Flowchart | `scheduler/sentiment_runner` |
| `flowcharts/P4_hmm_decoding.html` | Flowchart | `scheduler/hmm_updater.update_hmm_states` |
| `flowcharts/P5_dashboard.html` | Flowchart | `website/web/views` |
| `pseudocodes/index.html` | Index | navigasi |
| `pseudocodes/orchestration.html` | Pseudocode | scheduler entry points |
| `pseudocodes/P3_sentiment.html` | Pseudocode | sentiment runner |
| `pseudocodes/P4_hmm_decoding.html` | Pseudocode | HMM updater (forward filter, frozen params) |
| `pseudocodes/P5_dashboard.html` | Pseudocode | views.dashboard + views.forecasts_api + scheduler.precompute |
| `structure_charts/index.html` | Index | navigasi |
| `structure_charts/P1_price_acquisition.html` | Structure | price_fetcher + writer |
| `structure_charts/P2_news_acquisition.html` | Structure | news_extractor |
| `structure_charts/P3_sentiment.html` | Structure | FinBERT-Tone runner |
| `structure_charts/P4_hmm_decoding.html` | Structure | hmm_updater (frozen params) |
| `structure_charts/P5_dashboard.html` | Structure | Django views read-only |

### Referensi generator (1 file)
| File | Tipe | Catatan |
|---|---|---|
| `revision/CLAUDE_CODE_PROMPT_STRUCTURED_DIAGRAMS.md` | Prompt doc | Bukan diagram; digunakan satu kali untuk men-generate diagram. **Informatif** — historical generator. Stale terhadap kode saat ini (lihat §3.D). Bukan target wajib perbaikan kecuali masih dirujuk sebagai sumber kebenaran. |

---

## 2. Checklist FASE 2 — apakah diagram menyebutkan elemen yang baru berubah?

Saya menjalankan grep menyeluruh untuk pola berikut di seluruh `diagrams/*.html`:

| Pola dicari | Hit di diagram aktif | Keputusan |
|---|---|---|
| `R²`, `R2_Price`, `R2_LogReturn`, `r2_price`, `r2_logreturn` | ✗ tidak ada | OK |
| `max_depth=9`, `max_depth = 9`, `0.6715`, `n_estimators=20000`, `n_estimators=2000`, `learning_rate=0.001` | ✗ tidak ada | OK |
| `MAPE.*objective`, `fitness.*MAPE`, `MAPE.*fitness`, `MAPE-on-log` | ✗ tidak ada | OK |
| `Viterbi` di konteks decoding aktif | ✓ banyak hit — semua eksplisit **menyangkal** Viterbi ("BUKAN Viterbi") | OK |
| `multi-model`, `model.*selector`, `RandomForest`, `ARIMAX`, `SARIMAX` | ✗ tidak ada | OK |
| `early_stopping`, `eval_metric` | ✗ tidak ada | OK |
| `init_params=initial_states` | ✗ tidak ada (pseudocode kanonik pakai `init_params=""`) | OK |
| Klaim "CSA dijalankan pada SEMUA konfigurasi" | ✗ tidak ada (diagram fokus pada inference-only di scheduler/dashboard, tidak menggambar training) | OK |

Hanya **dua** hit stale ditemukan di diagram aktif, di bawah ini.

---

## 3. Tabel diskrepansi per file

### A. `diagrams/pseudocode.html` — 1 elemen STALE

| Elemen diagram | Realita kode saat ini | Verdict | Koreksi |
|---|---|---|---|
| Baris 345 — narasi sebelum `ALGORITHM compute_forecast_trails`: *"model XGBoost pemenang (**lowest base MAPE per horizon**) dipakai untuk memprediksi..."* | Setelah commit 6cb1d3d, `compute_winners.py` dan `fase4_thesis_runner.pick_winners_by_base_rmse` memilih pemenang berdasarkan **min BASE RMSE** (bukan MAPE). | **STALE** | Ubah "lowest base MAPE per horizon" → "lowest base RMSE per horizon". |

Sisanya (Algoritma 1 run_daily_pipeline, Algoritma 2 fit_hmm_with_restarts +
_fit_single + label_states, Algoritma compute_forecast_trails dst.) OK:
fit_cutoff sudah 2025-01-01, n_restarts=50, init_params="", params="stmc",
min_covar=1e-3, K-Means seeding, forward filter — semua sesuai kode.

### B. `diagrams/pseudocodes/P5_dashboard.html` — 1 elemen STALE

| Elemen diagram | Realita kode saat ini | Verdict | Koreksi |
|---|---|---|---|
| Baris 134 — di `VIEW dashboard`: `metrics = { mape: 0, r2: 0, accuracy: 0, best_model: "N/A" }` | Setelah FASE 2.5 + commit 630b72e: `_empty_dashboard_ctx` dan badge dashboard memakai `smape` (bukan `r2`). Lihat [website/web/views.py:170](website/web/views.py#L170). | **STALE** | `r2` → `smape`. |

Algoritma `compute_forecast_trails` di file ini (mulai baris 404) tidak
menyentuh seleksi winner — hanya membaca `winners_payload.winners_by_horizon`
yang sudah konsisten dengan kode. OK.

### C. Semua diagram lain — OK (0 STALE)

`data_flow_diagram.html`, `flowchart.html`, `structure_chart.html`, semua
file di `flowcharts/`, `pseudocodes/{index,orchestration,P3,P4}.html`,
seluruh `structure_charts/` — tidak ada referensi R², BASE tuned, MAPE-CSA,
Viterbi (di luar penolakan eksplisit), multi-model, atau CSA-on-all.

### D. `revision/CLAUDE_CODE_PROMPT_STRUCTURED_DIAGRAMS.md` — INFORMATIONAL

Dokumen ini adalah **prompt generator** yang dipakai sekali untuk men-generate
diagram-diagram di atas; bukan diagram itu sendiri dan tidak dirender ke
skripsi. Saya cantumkan diskrepansi untuk transparansi, namun **tidak diperbaiki
di FASE 4** kecuali user secara eksplisit ingin men-regenerate diagram dengan
prompt baru:

| Lokasi | Konten stale |
|---|---|
| line 330, 453, 515 | "Viterbi decode state per day" / `viterbi_decode(...)` / `init_params = initial_states` |
| line 465-475 | Flow training menggambarkan CSA per (config, horizon) — saat ini staged (BASE penuh → winner per horizon → CSA hanya pada winner). |
| line 577-579 | "best variant by lowest MAPE" → seharusnya RMSE. |

Karena prompt-doc tidak dirujuk oleh diagram terkini (diagram sudah self-contained
dengan SOURCE per algoritma), kami menandainya **historical** dan tidak menjadi
deliverable wajib FASE 4. Catat di laporan: jika di masa depan prompt ini
dipakai ulang, ia harus disinkronkan dulu.

---

## 4. Ringkasan

| Kategori | Jumlah |
|---|---|
| Total file diagram aktif | 20 |
| File OK (no STALE) | 18 |
| File perlu koreksi konten | 2 |
| Total elemen STALE | 2 |
| File historical (informational) | 1 (`revision/CLAUDE_CODE_PROMPT_STRUCTURED_DIAGRAMS.md`) |

### Daftar file perlu edit (FASE 4):
1. `diagrams/pseudocode.html` — line 345 (MAPE → RMSE narasi)
2. `diagrams/pseudocodes/P5_dashboard.html` — line 134 (r2 → smape di empty metrics dict)

Tidak ada perubahan styling/CSS/layout. Hanya konten.

---

## 5. Status

- [x] FASE 0 — Inventaris (20 diagram + 1 prompt doc)
- [x] FASE 1/2 — Cross-check elemen-per-elemen + FASE 2 checklist
- [x] FASE 3 — Tabel diskrepansi (laporan ini)
- [ ] FASE 4 — Koreksi 2 file (akan dikerjakan setelah commit laporan)
- [ ] FASE 5 — Re-verify post-correction

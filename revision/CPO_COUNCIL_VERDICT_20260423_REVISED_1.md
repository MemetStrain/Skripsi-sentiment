# CPO Thesis — Council Verdict & Action Plan (REVISED)

**Penulis Skripsi:** Matthew Owen (NIM 2602065156)
**Institusi:** Computer Science, Universitas Bina Nusantara
**Judul Skripsi:** Prediksi Harga Komoditas Crude Palm Oil Menggunakan Analisis Sentimen Berita dan Hidden Markov Model pada Aplikasi Berbasis Web
**Scope Review:** Bab 1–3
**Tanggal Council:** 2026-04-23
**Revision Date:** 2026-04-23 (post author clarifications)
**Protokol:** 5-member council (single-agent fallback), 3-round deliberation + reconvene

---

## 0. Revision Notes (Perubahan dari Verdict Awal)

Klarifikasi dari penulis mengubah severity assessment:

1. **Semua model masih dalam fase testing.** Angka dashboard "92% DA" bukan final claim — preliminary/testing run. Model selection final belum dilakukan.

2. **Solo thesis.** Multi-annotator (2–3 orang) untuk validasi FinBERT tidak feasible. Solusi: intra-annotator temporal consistency (label 2× dengan gap 1–2 minggu, hitung κ diri-vs-diri).

3. **MPOB dipilih dengan justifikasi kuat:**
   - Compliance terhadap ToS (sumber lain melanggar)
   - Bahasa Inggris native → tidak perlu translation → FinBERT langsung applicable
   - Malaysia sebagai pricing benchmark (Investing.com pakai MYR, FCPO sebagai futures reference)

4. **Minimalist feature set (tanpa variabel eksogen domain)** adalah research design intentional untuk menguji apakah sentimen berita + HMM state + lagged price **cukup** untuk prediksi.

5. **Waterfall SDLC hanya untuk web layer**, bukan model development.

### Impact pada Severity

| Blocker | Before | After | Alasan |
|---------|--------|-------|--------|
| 1. Dashboard integrity | 🔴 Critical | 🟡 Major | Testing phase, bukan final |
| 2. Ekonometrika | 🔴 Critical | 🔴 Critical | Tidak berubah |
| 3. FinBERT validation | 🔴 Critical | 🟢 Moderate | MPOB justifikasi kuat + intra-annotator |
| 4. Inkonsistensi dok. | 🔴 Critical | 🔴 Critical | Tidak berubah |
| 5. Feature set | 🟡 Major | 🟡 Major (restructured) | Research design valid; naive baseline wajib |
| 6e. Waterfall | 🟢 Moderate | ✅ Resolved | Clarification diterima |

---

## 1. Revised Verdict

**Status:** MAJOR REVISION — feasible dalam **4–6 minggu focused work**

**Konsensus panel:** 5/5 MAJOR REVISION. Tidak ada REJECT.

**Keputusan:** Skripsi dapat menjadi sidang-able dengan revisi focused. Prioritaskan Blocker 2 + 4 + 5' sebagai critical path.

**Confidence panel:** 0.90

---

## 2. Panel Final Positions (Post-Reconvene)

| Kode | Persona | Verdict Awal | Verdict Revisi | Catatan |
|------|---------|--------------|----------------|---------|
| A | Harjono (Econometrician) | REJECT | MAJOR REVISION | Blocker 2 unchanged |
| B | Kenji (ML Researcher) | REJECT | MAJOR REVISION | Testing phase OK; naive baseline masih wajib |
| C | Priya (NLP) | MAJOR | MODERATE | MPOB justifikasi diterima; intra-annotator solution |
| E | Budi (Commodity Econ) | REJECT | MAJOR REVISION | Minimalist design valid; naive baseline wajib |
| F | Elena (Architect) | MAJOR → REJECT | MAJOR REVISION | Waterfall clarification diterima |

---

## 3. Core Findings (Revised)

### Finding #1 — Dashboard Labeling Issue (🟡 Major, was 🔴)
Dashboard menampilkan "MAPE 5.36%, R² 0.86, DA 92.01%" tanpa context bahwa ini preliminary testing. `validation_summary.csv` menunjukkan ARIMAX/SARIMAX h=1 mencapai DA 48–49%.

**Impact:** Jika angka ini di-publish di skripsi tanpa caveat, penguji akan menanyakan. Jika di-present sebagai "preliminary testing — final model selection ongoing," acceptable untuk defend.

### Finding #2 — Asumsi Model Ekonometrika Dilanggar (🔴 unchanged)
Dari `validation_summary.csv`:
- `BG_no_serial = False` → residual punya serial correlation
- `JB_normal = False`, `SW_normal = False` → residual tidak normal
- `ARCH_homo = False` → heteroskedasticity
- `R²` negatif pada horizon ≥ 2

Model ARIMAX/SARIMAX **mis-specified**. Harus re-spec dengan `auto_arima` + diagnostic suite.

### Finding #3 — Cointegration Tidak Diuji (🔴 part of Blocker 2)
Meskipun Brent/crude oil tidak jadi fitur, **cointegration test tetap relevan** karena:
- Bab 2 membahas volatility spillover secara teoretis
- Jika CPO dan crude oil cointegrated, asumsi stasioneritas ARIMAX perlu extra check
- Test bisa dilakukan sebagai "supplementary analysis" tanpa harus mengubah feature set

Alternatif: jika penulis ingin eksklusi eksogen sepenuhnya, cointegration test **bisa dihapus** dengan catatan eksplisit di Bab 3 bahwa "single-variable ARIMA (tanpa X) dipilih sesuai scope minimalist." Tapi kemudian model bukan ARIMAX, harus dinamai ARIMA.

### Finding #4 — Validitas FinBERT (🟢 Moderate, was 🔴)

**Justifikasi MPOB diterima panel** setelah klarifikasi:
- ToS compliance (ethical research)
- English-native source (no translation confound)
- Malaysia pricing benchmark (MYR-denominated CPO di Investing.com)

**Yang tetap harus dilakukan (feasible untuk solo thesis):**
- Intra-annotator temporal consistency: label 200 berita sekarang, re-label 2 minggu kemudian, hitung Cohen's κ self-vs-self (target ≥ 0.60)
- Confusion matrix FinBERT vs manual self-label → macro-F1 report
- Justifikasi MPOB (3 alasan di atas) harus **eksplisit di Bab 1.4 atau 3.4**, bukan implicit

### Finding #5 — Minimalist Feature Set (🟡 Restructured)

**Research design diterima** — menguji apakah sentimen+HMM+lagged-price cukup adalah pertanyaan penelitian valid.

**Yang HARUS dilakukan agar design ini defensible:**
1. **Reframe hipotesis di Bab 1.3:** eksplisit state bahwa H1/H2/H3 adalah uji "cukupkah" → harus include comparison dengan naive baseline
2. **Naive baseline WAJIB** sebagai kontrol eksperimen:
   - Random walk: $\hat{y}_{t+1} = y_t$
   - Historical mean: $\hat{y}_{t+1} = \bar{y}$
3. **Ablation study minimum 4 konfigurasi:**
   - (a) Lagged price only
   - (b) Lagged + HMM state
   - (c) Lagged + sentiment
   - (d) Lagged + HMM + sentiment (full model)
4. **Diskusi Bab 5:** acknowledge bahwa minimalist feature set membatasi generalisasi sebagai future work

Tanpa naive baseline, pertanyaan "cukup?" tidak testable — ini blocker, bukan improvement.

### Finding #6 — Inkonsistensi Internal (🔴 unchanged)

Sama dengan verdict awal:
- Frekuensi data (harian/bulanan) tidak konsisten
- Stack database (Firestore/SQLite/ERD) bertentangan
- HMM states (2 vs 3)
- Terminologi R² salah
- 5 RQ vs 3 hipotesis
- Pipeline claim (otomatis vs manual)
- Equation & figure numbering

### Finding #7 — Architecture (🟢 mostly resolved)

Waterfall clarification diterima. Fix: tulis eksplisit di Bab 2.7 & 3 bahwa "Waterfall diterapkan pada web layer; model development menggunakan pendekatan iteratif eksperimental."

Remaining: ERD vs Firestore implementation tetap perlu dipilih satu (Blocker 4).

---

## 4. Revised Action Plan

### 🔴 CRITICAL BLOCKER 2 — Fix Model Ekonometrika

**Owner:** Matthew
**Effort:** 2–3 minggu
**Status:** Unchanged dari verdict awal

#### Tasks (unchanged)

**2a. Stationarity check eksplisit (Bab 3.6)**
- ADF + KPSS pada level price, log return, differenced log return
- Tabel: `Variable | ADF stat | p-value | KPSS stat | Conclusion`

**2b. Cointegration test (kondisional)**
- Jika tetap ada "ARIMAX" (ada eksogen lagged), test Engle-Granger untuk lagged price pair
- Jika full minimalist (pure ARIMA, no X), skip dengan dokumentasi eksplisit di Bab 3

**2c. ARIMA(X) re-specification**
- `auto_arima(seasonal=False, stepwise=True)`
- Grid: p ∈ [0,5], d ∈ [0,2], q ∈ [0,5]
- Target: AIC minimum + Ljung-Box residual p > 0.05

**2d. HMM emission justification**
- Shapiro-Wilk pada log return (bulk + per-state)
- Jika non-normal → Student-t HMM atau mixture-of-Gaussians
- Multi-restart minimum 10 seed, pilih best by BIC
- Dokumentasikan N states via BIC comparison (N ∈ {2, 3, 4})

**2e. Residual diagnostic suite untuk tiap model ekonometri**
- Ljung-Box, Jarque-Bera, ARCH-LM, Breusch-Godfrey, Pesaran-Timmermann
- **Target minimum:** Ljung-Box pass + Breusch-Godfrey pass untuk ARIMA final

#### Acceptance Criteria
- [ ] Setiap seri data punya stationarity decision eksplisit
- [ ] ARIMA final spec lulus Ljung-Box + Breusch-Godfrey
- [ ] HMM emission dipilih dengan justifikasi uji normality
- [ ] Diagnostic suite dilaporkan di Bab 4 untuk semua model ekonometri

---

### 🔴 CRITICAL BLOCKER 4 — Resolve Inkonsistensi Internal

**Owner:** Matthew
**Effort:** 3–5 hari
**Status:** Unchanged

#### Tasks
1. Frekuensi data: tetapkan harian (~2750 obs). Revisi Sec 1.4.2, 1.5.1, 3.2.2
2. Stack database: pilih satu + konsisten full document
3. HMM N states: tetapkan eksplisit via BIC comparison
4. Terminologi R²: "Koefisien Determinasi" di semua tempat
5. Hipotesis vs RQ: align (tambah H4 komparasi model, atau kurangi RQ)
6. Pipeline narrative: otomatis atau manual, pilih + match Use Case
7. Equation + figure numbering audit
8. Daftar gambar regenerate
9. **Tambahan post-clarification:** Waterfall eksplisit di Bab 2.7 & 3 — "hanya web layer"

#### Acceptance Criteria
- [ ] Full-text search untuk keyword bermasalah menunjukkan konsistensi
- [ ] Equation/figure numbers berurutan tanpa duplikat
- [ ] Waterfall scope (web-only) eksplisit

---

### 🔴 CRITICAL BLOCKER 5' — Reframe Hipotesis + Naive Baseline

**Owner:** Matthew
**Effort:** 1 minggu
**Dependencies:** Blocker 2 setengah selesai (model sudah ready untuk benchmark)

Ini restructuring dari Blocker 5 awal. Bukan menambah fitur eksogen — justru **memperkuat minimalist design** dengan kontrol yang tepat.

#### Tasks

**5a. Reframe hipotesis di Bab 1.3**
Tulis ulang hipotesis dengan framing "cukupkah":
- H1: Fitur sentimen (FinBERT) **menurunkan MAPE** dibandingkan model lagged-price only
- H2: Lagged price $t-1$ menjadi prediktor signifikan (autokorelasi)
- H3: HMM state **meningkatkan DA** dibandingkan model tanpa state
- H4 (tambah): Model gabungan sentimen+HMM+lagged **mengalahkan naive random walk**

Framing ini membuat tiap H testable dan mengubah minimalist design dari "keterbatasan" menjadi "hipotesis-driven scope."

**5b. Tambah batasan eksplisit di Bab 1.4**
Dokumentasikan alasan tidak memasukkan variabel eksogen domain (FCPO, soybean oil, Brent, USD/MYR, dll):
> "Penelitian ini secara sengaja menggunakan feature set minimalis untuk mengisolasi kontribusi fitur sentimen berita dan hidden state HMM terhadap akurasi prediksi. Variabel eksogen seperti futures FCPO, soybean oil, crude oil, dan nilai tukar MYR/IDR tidak dimasukkan untuk menghindari confound dalam menguji hipotesis utama. Integrasi variabel-variabel tersebut dijadikan future work."

**5c. Implementasi naive baseline**
- Random walk: $\hat{y}_{t+1} = y_t$
- Seasonal naive: $\hat{y}_{t+1} = y_{t-7}$ atau $y_{t-30}$
- Historical mean (in-sample): $\hat{y}_{t+1} = \bar{y}_{\text{train}}$

**5d. Ablation study**
Minimum 4 konfigurasi untuk test kontribusi tiap fitur:
| Config | Fitur | Rationale |
|--------|-------|-----------|
| Naive | — | Baseline kontrol |
| L | Lagged price only | Baseline informatif |
| L + HMM | Lagged + HMM state | Test H3 (state contribution) |
| L + Sentiment | Lagged + sentiment | Test H1 (sentiment contribution) |
| L + HMM + Sentiment | Full model | Test H4 (gabungan) |

**5e. Reporting hasil di Bab 4**
Tabel final:
```
Model | MAPE | sMAPE | R² | DA | PT p-value | 95% CI
```
Per horizon, per model config. Identifikasi winner dengan Diebold-Mariano test.

#### Acceptance Criteria
- [ ] Hipotesis di Bab 1.3 reframed dengan "cukupkah" framing
- [ ] Bab 1.4 berisi justifikasi eksplisit minimalist feature set
- [ ] Naive baseline diimplementasikan dan dilaporkan
- [ ] Ablation table menunjukkan kontribusi tiap fitur
- [ ] Jika naive tidak bisa dikalahkan → narasi jujur di Bab 5 (temuan negatif valid)

---

### 🟡 MAJOR BLOCKER 1 — Dashboard Labeling + Final Model Selection

**Owner:** Matthew
**Effort:** 1–2 hari setelah Blocker 2 dan 5' selesai

#### Tasks
1. Tentukan final "best model" berdasarkan protocol eksplisit (best MAPE? best DA? best composite?)
2. Re-run final model dengan fixed seed + 10-seed variance
3. Update dashboard dengan angka final + label "Final Test-Set Result"
4. **Atau** (alternatif): label dashboard sebagai "Preliminary Testing Result" sampai model selection final
5. Pastikan angka di skripsi = angka di dashboard = angka di kode

#### Acceptance Criteria
- [ ] Model selection protocol didokumentasikan di Bab 3 atau 4
- [ ] Angka dashboard bisa direproduksi dari kode
- [ ] Tidak ada klaim angka tanpa source

---

### 🟢 MODERATE BLOCKER 3' — Intra-Annotator Validation + MPOB Justifikasi

**Owner:** Matthew
**Effort:** 1 minggu (termasuk 2× labeling session dengan gap)

#### Tasks

**3a. Intra-annotator temporal consistency**
- Random sample 200 berita MPOB
- Label manual pass 1 (hari H)
- **Tunggu minimum 10–14 hari** (hindari memory effect)
- Label manual pass 2 dengan berita yang sama, order diacak
- Compute Cohen's κ pass1 vs pass2 (target ≥ 0.60)
- Dokumentasikan dengan transparan: "intra-annotator agreement karena solo thesis"

**3b. Confusion matrix FinBERT vs self-label**
- Gunakan majority vote antara pass 1 dan pass 2 sebagai "ground truth" (atau resolve disagreement secara eksplisit)
- Confusion matrix FinBERT prediction vs self-labeled truth
- Report: precision, recall, F1 per class, macro-F1
- Target: macro-F1 ≥ 0.60

**3c. Justifikasi MPOB eksplisit di Bab 1.4 atau 3.4**
Tulis minimum 1 paragraf:
> "MPOB dipilih sebagai satu-satunya sumber berita dengan justifikasi berikut: (1) compliance terhadap Terms of Service — MPOB mengizinkan automated scraping untuk riset akademik, sedangkan sumber alternatif seperti Reuters dan Bloomberg memiliki restriksi; (2) bahasa Inggris native — MPOB menerbitkan berita dalam bahasa Inggris, menghilangkan confound translasi dan memungkinkan FinBERT (di-train pada korpus Inggris) digunakan tanpa adaptation berlapis; (3) relevansi pasar — Malaysia merupakan produsen CPO terbesar kedua dan Bursa Malaysia (MYR-denominated) adalah pricing benchmark utama (Investing.com menggunakan MYR sebagai acuan harga CPO), sehingga berita MPOB secara sistematis mempengaruhi ekspektasi pasar global."

**3d. Strategi aggregation sentiment harian**
- Pilih strategi eksplisit (mean / last / weighted)
- Handle hari tanpa berita: forward-fill, zero, atau NaN imputation
- Sensitivity analysis (2 strategi, compare hasil)

#### Acceptance Criteria
- [ ] Cohen's κ self-vs-self ≥ 0.60 dilaporkan
- [ ] Confusion matrix + macro-F1 FinBERT vs self-truth dilaporkan
- [ ] Justifikasi MPOB eksplisit (3 alasan) di Bab 1.4 atau 3.4
- [ ] Aggregation strategy didokumentasikan

---

### 🟢 MODERATE ISSUES (Paralel)

#### 6a. Walk-forward validation
- Minimum 5 fold (expanding window atau rolling)
- Report mean ± std

#### 6b. Diebold-Mariano test
- Pairwise model comparison
- p-value matrix dengan Holm-Bonferroni correction

#### 6c. sMAPE
- Tambahkan di Sec 1.4.7, 1.6.3.6, 2.6

#### 6d. DA notation
- Ganti `==` dengan $\mathbb{1}\{\text{sign}(\hat{y}_{t+1} - y_t) = \text{sign}(y_{t+1} - y_t)\}$

#### 6e. Admin upload security
- CSRF, MIME validation, size limit, CSV injection sanitization, rate limiting

#### 6f. Model versioning
- MLflow atau DVC; fixed seed documented; requirements.txt pinned

#### 6g. ERD vs implementasi
- Align: pilih satu database stack, atau gunakan 2 diagram untuk hybrid (auth vs data)

---

## 5. Revised Timeline (4–6 minggu)

| Minggu | Focus | Deliverable |
|--------|-------|-------------|
| 1 | Blocker 4 + reframe Bab 1.3/1.4 | Draft bersih, hipotesis restructured, justifikasi MPOB + minimalist eksplisit |
| 2–3 | Blocker 2 | Stationarity, (cointegration opsional), ARIMA re-spec, HMM diagnostics, residual suite |
| 4 | Blocker 5' + Blocker 1 | Naive baseline + ablation study, model selection protocol, dashboard labeling final |
| 5 | Blocker 3' + Moderate | Intra-annotator validation 200 berita (2 pass), sMAPE, DM test, walk-forward |
| 6 | Draft Bab 4–5 + revisi final Bab 1–3 | Skripsi final-draft siap sidang |

**Minimum feasible: 4 minggu** (jika full-time dan kode sudah ready).
**Realistik: 6 minggu.**

---

## 6. Risk Assessment Revised

### Jika maju sidang bentuk sekarang:
- Probabilitas direct reject: LOW-MODERATE (15–30%) — turun signifikan dari revisi awal
- Probabilitas major revision dari penguji: HIGH (>60%)

### Pertanyaan penguji yang masih berbahaya (post-revision):
1. "Residual ARIMAX Anda lulus Ljung-Box?" → **tidak bisa dijawab tanpa re-run** — tetap critical
2. "Model Anda mengalahkan naive random walk?" → **belum diuji** — critical
3. "Mengapa hanya MPOB?" → **sekarang bisa dijawab** (ToS + English + pricing benchmark) kalau masuk Bab 1.4
4. "Kenapa DA di dashboard 92% tapi di validation_summary 48%?" → jawaban: "preliminary testing; model selection final di halaman X" — acceptable kalau sudah di-label

### Jika revisi sesuai action plan:
- Probabilitas lulus sidang: HIGH (>85%)
- Kontribusi akademik: salvageable + potensial publikasi nasional

---

## 7. Decision Gate Revised — Checklist Sebelum Daftar Sidang

- [ ] **Blocker 2 resolved:** Ljung-Box + Breusch-Godfrey pass untuk ARIMA final; HMM emission justified
- [ ] **Blocker 4 resolved:** full-text audit konsistensi clean; Waterfall scope eksplisit
- [ ] **Blocker 5' resolved:** hipotesis reframed; naive baseline + ablation study dilaporkan
- [ ] **Blocker 1 resolved:** dashboard labeled correctly + final model selection dilakukan
- [ ] **Blocker 3' resolved:** intra-annotator κ ≥ 0.60, macro-F1 ≥ 0.60, justifikasi MPOB eksplisit
- [ ] **Moderate 6c:** sMAPE dilaporkan
- [ ] Dry-run pertanyaan berbahaya (§6) dengan supervisor

---

## 8. Council Members — Final Statements (Revised)

**A (Harjono):** "Klarifikasi scope tidak mengubah apa pun tentang residual diagnostics. Itu tetap non-negotiable. Sisanya — silakan, research design Anda."

**B (Kenji):** "Testing phase fine, minimalist design fine — tapi naive baseline adalah kontrol eksperimen, bukan optional nice-to-have. Tanpa itu, H1/H3 tidak falsifiable."

**C (Priya):** "Justifikasi MPOB yang Anda jelaskan kuat dan masuk akal. Tolong tulis itu di skripsi Anda — saat ini tidak eksplisit. Intra-annotator temporal consistency adalah solusi yang well-accepted untuk solo researcher."

**E (Budi):** "Minimalist design OK sebagai pertanyaan penelitian. Kalau Anda bisa menunjukkan 'sentimen+HMM cukup untuk mengalahkan naive,' itu kontribusi menarik. Kalau tidak bisa, itu juga temuan — jangan paksa narasi positif."

**F (Elena):** "Waterfall hanya untuk web — fine. Tulis eksplisit. Sisanya (stack database consistency, dashboard labeling) adalah hygiene yang wajib."

---

## 9. Summary untuk Penulis

**Yang Anda benar:**
- Scope minimalist adalah research design valid (bukan kelalaian)
- MPOB-only punya justifikasi substantif (ToS, English, pricing benchmark)
- Solo thesis constraint adalah reality — intra-annotator solution valid
- Waterfall untuk web layer saja masuk akal
- Model yang dilaporkan masih testing, bukan final claim

**Yang tetap perlu dikerjakan (critical path):**
1. Residual diagnostics + ARIMA re-spec (Blocker 2)
2. Inkonsistensi dokumentasi (Blocker 4)
3. Naive baseline + reframe hipotesis (Blocker 5')
4. Justifikasi MPOB eksplisit di skripsi + intra-annotator validation (Blocker 3')
5. Model selection protocol + dashboard labeling (Blocker 1)

**Estimasi realistik:** 4–6 minggu focused work. Probabilitas lulus sidang post-revision: >85%.

---

**End of Revised Verdict Document.**

*Generated via single-agent Council Protocol, 2026-04-23.*
*5-member panel, reconvened after author clarifications.*
*Consensus: 5/5 MAJOR REVISION, 0 REJECT.*

# Ringkasan Tugas - Skrip Penghasil Angka Skripsi & Cap HMM Validation

Tanggal: 2026-05-29

## Pertanyaan
Setelah menjalankan 10 skrip pipeline secara berurutan (fetch -> preprocess -> HMM states -> scrap -> news preprocessing -> FinBERT -> EDA -> HMM lag analysis -> sentiment_vs_price -> fase4_thesis_runner), skrip apa lagi yang perlu dijalankan untuk memperbarui angka skripsi.

## Temuan
Hanya satu skrip penghasil angka skripsi yang belum dijalankan: `markov/hmm_validation_suite.py`.
Skrip ini memproduksi angka validasi HMM Bab 4.4 yang TIDAK dihasilkan oleh fase4_thesis_runner:
- `hmm_bic_comparison.csv` + `.png`  -> Tabel 4.10 + Gambar 4.1 (BIC N=2/3/4)
- `hmm_restart_stability.csv`         -> Tabel 4.12 (stabilitas 30 restart)
- `hmm_transition_audit.csv`, `hmm_emission_diagnostics.csv` -> Tabel 4.4
- `hmm_state_labels_validation.csv`, `hmm_validation_summary.csv`

Output lama di `markov/validation_output/` tertanggal 9 Mei 2026 (basi, sebelum cap data & re-fit HMM).
Angka BIC lama (N=3: 19.381,09; N=4: 18.164,88) persis yang dikutip skripsi P630.

Skrip lain TIDAK perlu dijalankan untuk angka skripsi:
- csa_overfit_check.py, csa_stability_check.py -> robustness eksplisit di luar cakupan (P88)
- exploratory/finbert_validation/*, news/lm_finbert_agreement.py -> validasi sentimen diasumsikan, bukan diuji (P91, Bab 1.4)
- compute_winners.py, feature_importance_load.py, optimal_lag_acf.py -> untuk website / sudah diproduksi runner
- fase4_thesis_runner.py -> orchestrator; sudah memproduksi Tabel 4.6-4.9, PT, H4, feature importance (THESIS_RESULTS_20260530)

## Masalah ditemukan & perbaikan diterapkan
`hmm_validation_suite.py._load_cpo_variables` membaca CSV penuh (2.724 baris s.d. 25 Mei 2026) TANPA cap Feb 2026,
berbeda dengan cpo_hmm_states.py yang memotong di Date <= 2026-02-28 (baris 139).
Ada 46 baris setelah 28 Feb 2026 yang akan melanggar jendela 3 Agu 2015 - Feb 2026.

Perbaikan (via bash; file tool memangkas mount D:):
1. Tambah `DATA_END` ke import dari `cpo_hmm_states` (single source: config.dates).
2. Tambah `df = df[df["Date"] <= DATA_END].reset_index(drop=True)` di `_load_cpo_variables`.
py_compile OK. Setelah perbaikan: 2.678 baris (46 baris Mar-Mei 2026 terbuang). Backup: hmm_validation_suite.py.bak_*.

## Perintah menjalankan (dari D:\Skripsi-sentiment)
PYTHONIOENCODING=utf-8 website/venv/Scripts/python.exe markov/hmm_validation_suite.py --output-dir markov/validation_output/

Setelah run, output segar mengisi Tabel 4.4, 4.10, 4.12, dan Gambar 4.1.

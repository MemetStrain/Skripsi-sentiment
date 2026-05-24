# FinBERT Validation Suite — EKSPLORATORI (bukan bagian klaim tesis)

Skrip di folder ini menjalankan validasi sentimen FinBERT (Financial
PhraseBank benchmark, manual 50-sampel, korelasi sentimen-harga).

**Status:** TIDAK dirujuk sebagai bukti dalam skripsi. Sesuai arahan
pembimbing, validitas sentimen FinBERT **diasumsikan memadai** dan
dinyatakan sebagai *delimitation* di Bab 1.4 — karena validasi domain CPO
yang sahih menuntut anotator ahli yang tidak tersedia pada penelitian
mandiri ini. Skrip dipertahankan untuk transparansi dan basis *future work*
(validasi multi-annotator, target Cohen's κ ≥ 0.60).

**Model:** produksi memakai `yiyanghkust/finbert-tone` (Yang et al., 2020),
bukan ProsusAI/finbert.

Menjalankan: `python exploratory/finbert_validation/finbert_validation_suite.py --all`

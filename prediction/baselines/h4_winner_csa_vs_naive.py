"""
h4_winner_csa_vs_naive.py
=========================
Uji H4 sufficiency yang benar: winner CSA per horizon vs naive random walk.

Masalah pada h4_sufficiency_C4_vs_naive_rw.csv:
  - Menggunakan C4 BASE untuk semua horizon
  - Seharusnya menggunakan winner CSA per horizon (C3 h1-3, C4 h4-5, C2 h6-7)
    sesuai compute_winners.py (min BASE RMSE per horizon)

Metodologi DM IDENTIK dengan build_h4_sufficiency() di fase4_thesis_runner.py:
  - Loss: squared
  - Koreksi: Harvey-Leybourne-Newbold (1997) small-sample via diebold_mariano_test()
  - Naive: log_return_pred = 0 (random walk)
  - Error: y_true - y_pred  (A=parametric, B=naive)
  - Konvensi better: p>=0.05 -> tie; dm<0 -> parametric; dm>0 -> naive_rw
  - Holm-Bonferroni ditambah (konsisten Bab 4, tidak ada di skrip lama)

Output baru: THESIS_RESULTS_20260524/h4_sufficiency_winnerCSA_vs_naive_rw.csv
File lama TIDAK diubah atau ditimpa.

Usage:
    python prediction/baselines/h4_winner_csa_vs_naive.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — agar bisa import dari prediction/ root
# ---------------------------------------------------------------------------
_BASELINES_DIR   = Path(__file__).resolve().parent          # prediction/baselines/
_PREDICTION_DIR  = _BASELINES_DIR.parent                    # prediction/
_PROJECT_ROOT    = _PREDICTION_DIR.parent                   # repo root
if str(_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICTION_DIR))
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

# Modul produksi yang sudah ada; tidak diduplikasi
from naive_baseline import diebold_mariano_test     # DM + HLN correction  # noqa: E402
from multiple_comparison import holm_bonferroni      # Holm-Bonferroni FWER  # noqa: E402


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
INTERVAL: str = "Daily"
SPLIT:    str = "testing"
ALPHA:    float = 0.05

OUTPUT_BASE  = _PREDICTION_DIR / "output_horizons"
RESULTS_DIR  = _PROJECT_ROOT / "THESIS_RESULTS_20260524"
OUTPUT_CSV   = RESULTS_DIR / "h4_sufficiency_winnerCSA_vs_naive_rw.csv"

# Sumber kanonik winners: winners.json di THESIS_RESULTS_20260524
WINNERS_JSON_PATH = RESULTS_DIR / "winners.json"

# Toleransi guardrail MAPE/RMSE vs Tabel 4.7
METRIC_TOLERANCE: float = 0.01


# ---------------------------------------------------------------------------
# Helper: dapatkan winner per horizon dari winners.json (sumber kanonik)
# ---------------------------------------------------------------------------

def load_winners_from_json(
    winners_json: Path,
) -> Dict[int, Dict[str, str]]:
    """
    Baca winners.json dan kembalikan dict {horizon: {'tag': ..., 'config': ...}}.

    compute_winners.py menggunakan min BASE RMSE per horizon sebagai kriteria
    pemilihan (bukan MAPE). winners.json adalah output kanonik fungsi compute().
    """
    if not winners_json.exists():
        raise FileNotFoundError(f"winners.json tidak ditemukan: {winners_json}")
    with open(winners_json, encoding="utf-8") as f:
        payload = json.load(f)

    winners_by_horizon: Dict[str, str] = payload.get("winners_by_horizon", {})
    configs_by_horizon: Dict[str, str] = payload.get("configs_by_horizon", {})

    if not winners_by_horizon:
        raise ValueError("winners.json kosong atau tidak memiliki 'winners_by_horizon'")

    result: Dict[int, Dict[str, str]] = {}
    for h in HORIZONS:
        key = str(h)
        if key not in winners_by_horizon:
            raise KeyError(f"Horizon {h} tidak ada di winners_by_horizon")
        result[h] = {
            "tag":    winners_by_horizon[key],    # e.g. "cpo_sentiment"
            "config": configs_by_horizon[key],    # e.g. "C3"
        }
    return result


# ---------------------------------------------------------------------------
# Helper: path predictions CSV per horizon / tag
# ---------------------------------------------------------------------------

def _predictions_path(tag: str, horizon: int) -> Path:
    """Path testing_predictions CSV untuk tag dan horizon tertentu."""
    return (OUTPUT_BASE / tag / INTERVAL / f"horizon_{horizon}"
            / f"{SPLIT}_predictions_{INTERVAL}_h{horizon}.csv")


def load_predictions(tag: str, horizon: int) -> pd.DataFrame:
    """
    Muat predictions CSV; validasi kolom wajib ada.

    Raises
    ------
    FileNotFoundError : file tidak ada
    ValueError        : kolom kritis tidak ditemukan
    """
    path = _predictions_path(tag, horizon)
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV tidak ditemukan: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise IOError(f"Gagal membaca {path}: {exc}") from exc

    required_cols = {"Date", "Actual_LogReturn", "Close_Anchor", "xgboost_csa_LogReturn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom berikut tidak ada di {path.name}: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Guardrail: verifikasi MAPE/RMSE vs Tabel 4.7 (tolerance < 0.01)
# ---------------------------------------------------------------------------

def _compute_price_metrics(
    lr_true: np.ndarray,
    lr_pred: np.ndarray,
    close_anchor: np.ndarray,
) -> Dict[str, float]:
    """
    Hitung MAPE dan RMSE dalam price space — identik dengan calculate_metrics()
    di forecast_utils.py.

    Parameters
    ----------
    lr_true, lr_pred : log-return aktual dan prediksi
    close_anchor     : harga penutup t-0 (anchor eksponensial)
    """
    mask = ~np.isnan(lr_pred)
    lt = lr_true[mask]
    lp = lr_pred[mask]
    anchor = close_anchor[mask]

    # Clip untuk stabilitas exp() — sama dengan forecast_utils
    lt_c = np.clip(lt, -10, 10)
    lp_c = np.clip(lp, -10, 10)
    p_actual = anchor * np.exp(lt_c)
    p_pred   = anchor * np.exp(lp_c)

    nz = np.abs(p_actual) > 1e-9
    mape = float(np.mean(np.abs((p_actual[nz] - p_pred[nz]) / p_actual[nz])) * 100.0) if nz.any() else float("nan")
    rmse = float(np.sqrt(np.mean((p_actual - p_pred) ** 2)))

    return {"MAPE": round(mape, 4), "RMSE": round(rmse, 4), "n": int(mask.sum())}


def guardrail_metrics(
    horizon: int,
    tag: str,
    config: str,
    df: pd.DataFrame,
    tabel_47: Optional[pd.DataFrame],
) -> None:
    """
    Hitung ulang MAPE/RMSE dari prediksi CSA winner dan bandingkan dengan
    Tabel 4.7 (tabel_4.7_csa_winners_testing.csv). Toleransi |delta| < 0.01.

    Jika tabel_47 tidak tersedia, lewati guardrail dengan peringatan.
    Jika selisih > toleransi, raise ValueError (wajib berhenti).
    """
    if tabel_47 is None:
        print(f"  [GUARDRAIL] h{horizon}: tabel_4.7 tidak tersedia — skip verifikasi.")
        return

    # Filter baris Tabel 4.7 untuk horizon ini
    ref_row = tabel_47[tabel_47["Horizon"] == horizon]
    if ref_row.empty:
        print(f"  [GUARDRAIL] h{horizon}: baris tidak ditemukan di tabel_4.7 — skip.")
        return

    ref_mape = float(ref_row["MAPE"].iloc[0])
    ref_rmse = float(ref_row["RMSE"].iloc[0])
    ref_ablation = str(ref_row["Ablation"].iloc[0])

    # Pastikan ablation cocok
    if ref_ablation != tag:
        raise ValueError(
            f"h{horizon}: ablation di Tabel 4.7 = '{ref_ablation}', "
            f"winner dari winners.json = '{tag}'. Tidak konsisten!"
        )

    # Hitung metrik dari prediksi yang dimuat
    lr_true   = df["Actual_LogReturn"].to_numpy(dtype=float)
    lr_pred   = df["xgboost_csa_LogReturn"].to_numpy(dtype=float)
    close_anc = df["Close_Anchor"].to_numpy(dtype=float)
    metrics   = _compute_price_metrics(lr_true, lr_pred, close_anc)

    delta_mape = abs(metrics["MAPE"] - ref_mape)
    delta_rmse = abs(metrics["RMSE"] - ref_rmse)

    ok_mape = delta_mape < METRIC_TOLERANCE
    ok_rmse = delta_rmse < METRIC_TOLERANCE

    status = "OK" if (ok_mape and ok_rmse) else "FAIL"
    print(
        f"  [GUARDRAIL] h{horizon} ({config}/{tag}) | "
        f"MAPE: calc={metrics['MAPE']:.4f} ref={ref_mape:.4f} delta={delta_mape:.4f} {'✓' if ok_mape else '✗'} | "
        f"RMSE: calc={metrics['RMSE']:.4f} ref={ref_rmse:.4f} delta={delta_rmse:.4f} {'✓' if ok_rmse else '✗'} | "
        f"n={metrics['n']} | {status}"
    )

    if not (ok_mape and ok_rmse):
        raise ValueError(
            f"GUARDRAIL FAIL di h{horizon}: prediksi yang dimuat tidak cocok dengan "
            f"Tabel 4.7 (toleransi {METRIC_TOLERANCE}). MAPE delta={delta_mape:.4f}, "
            f"RMSE delta={delta_rmse:.4f}. Kemungkinan array prediksi yang dimuat salah."
        )


# ---------------------------------------------------------------------------
# Verdict helper (identik dengan _verdict() di fase4_thesis_runner.py)
# ---------------------------------------------------------------------------

def _verdict(dm: float, p: float, label_a: str, label_b: str) -> str:
    """
    Tentukan 'better' model berdasarkan DM stat dan p-value.

    Konvensi sama dengan fase4_thesis_runner._verdict():
      - NaN       -> "undetermined"
      - p >= 0.05 -> "tie"
      - dm < 0    -> label_a (parametric lebih baik)
      - dm >= 0   -> label_b (naive_rw lebih baik)
    """
    if np.isnan(dm) or np.isnan(p):
        return "undetermined"
    if p >= ALPHA:
        return "tie"
    return label_a if dm < 0 else label_b


# ---------------------------------------------------------------------------
# Fungsi utama: DM winner CSA vs naive_rw per horizon
# ---------------------------------------------------------------------------

def run_h4_winner_csa_vs_naive() -> pd.DataFrame:
    """
    Jalankan uji DM winner CSA vs naive_rw untuk semua horizon.

    Returns
    -------
    pd.DataFrame dengan hasil per horizon, termasuk Holm correction.
    """
    # Langkah 1: Seleksi winner per horizon dari winners.json
    print("=" * 70)
    print("Langkah 1 — Seleksi winner per horizon dari winners.json")
    print(f"  Sumber: {WINNERS_JSON_PATH}")
    winners = load_winners_from_json(WINNERS_JSON_PATH)
    print(f"  {'Horizon':<8} {'Config':<8} {'Tag':<20}")
    print(f"  {'-'*8} {'-'*8} {'-'*20}")
    for h in HORIZONS:
        print(f"  h{h:<7} {winners[h]['config']:<8} {winners[h]['tag']:<20}")

    # Muat Tabel 4.7 sebagai referensi guardrail (opsional, lewati jika absen)
    tabel_47_path = RESULTS_DIR / "tabel_4.7_csa_winners_testing.csv"
    tabel_47: Optional[pd.DataFrame] = None
    if tabel_47_path.exists():
        tabel_47 = pd.read_csv(tabel_47_path)
        print(f"\nTabel 4.7 dimuat dari {tabel_47_path.name} ({len(tabel_47)} baris)")
    else:
        print(f"\n[WARN] Tabel 4.7 tidak ditemukan di {tabel_47_path} — guardrail MAPE/RMSE dilewati.")

    # Langkah 2 + 3: Muat prediksi, guardrail, kemudian Langkah 4: DM test
    print("\n" + "=" * 70)
    print("Langkah 2–4 — Muat prediksi, guardrail, DM test per horizon")

    rows: List[Dict] = []
    for h in HORIZONS:
        tag    = winners[h]["tag"]
        config = winners[h]["config"]
        ablation = tag  # e.g. "cpo_sentiment"
        print(f"\n--- Horizon {h}: winner = {config} ({tag}) CSA ---")

        # Langkah 2: Muat prediksi
        try:
            df = load_predictions(tag, h)
        except (FileNotFoundError, ValueError, IOError) as exc:
            print(f"  [SKIP] Gagal muat prediksi: {exc}")
            continue

        # Join pada Date untuk alignment baris (keamanan ekstra)
        df = df.dropna(subset=["Date"]).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"  Prediksi dimuat: {len(df)} baris, date {df['Date'].iloc[0].date()} s/d {df['Date'].iloc[-1].date()}")

        # Validasi tidak ada NaN setelah alignment
        for col in ("Actual_LogReturn", "xgboost_csa_LogReturn"):
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"  [WARN] {col}: {n_nan} NaN ditemukan — akan di-mask sebelum DM.")

        # Langkah 3: Guardrail MAPE/RMSE vs Tabel 4.7
        try:
            guardrail_metrics(h, tag, config, df, tabel_47)
        except ValueError as exc:
            print(f"  [ERROR] {exc}")
            raise  # Hard stop — prediksi yang dimuat salah

        # Langkah 4: Siapkan array error untuk DM
        y_true = df["Actual_LogReturn"].to_numpy(dtype=float)
        y_pred = df["xgboost_csa_LogReturn"].to_numpy(dtype=float)
        # Naive random walk: prediksi log-return = 0 (tidak ada perubahan)
        y_naive = np.zeros_like(y_true)

        err_parametric = y_true - y_pred    # e_a: error model parametric
        err_naive      = y_true - y_naive   # e_b: error naive (= y_true)

        # Mask baris dengan NaN di salah satu error
        mask = ~(np.isnan(err_parametric) | np.isnan(err_naive))
        n = int(mask.sum())
        if n < 5:
            print(f"  [SKIP] Terlalu sedikit observasi valid: {n}")
            continue

        # DM test dengan HLN correction (loss squared, identik fase4_thesis_runner)
        dm, p = diebold_mariano_test(
            errors_model_a=err_parametric[mask],
            errors_model_b=err_naive[mask],
            h=h,
            loss="squared",
        )

        winner_label = f"{config}_{tag}_CSA"  # label untuk kolom "better"
        verdict = _verdict(dm, p, winner_label, "naive_rw")
        significant_raw = bool(not np.isnan(p) and p < ALPHA)

        print(
            f"  DM* = {dm:.4f}, p_raw = {p:.4f}, "
            f"sig_raw = {significant_raw}, better = {verdict} | n = {n}"
        )

        rows.append({
            "horizon":       h,
            "winner_config": config,
            "ablation":      ablation,
            "variant":       "CSA",
            "n":             n,
            "dm_stat":       round(dm, 4) if not np.isnan(dm) else float("nan"),
            "p_value_raw":   round(p, 4)  if not np.isnan(p)  else float("nan"),
            "significant_raw_0.05": significant_raw,
            "better_raw":    verdict,
        })

    if not rows:
        raise RuntimeError("Tidak ada horizon yang berhasil diuji. Periksa prediksi CSV.")

    df_out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

    # Langkah 4 (lanjutan): Holm-Bonferroni FWER correction
    # Family = 7 horizon; NaN p-value dikecualikan (konsisten multiple_comparison.py)
    p_raw_arr = df_out["p_value_raw"].to_numpy(dtype=float)
    p_holm, reject_holm = holm_bonferroni(p_raw_arr, alpha=ALPHA)
    df_out["p_value_holm"]         = np.round(p_holm, 4)
    df_out["significant_holm_0.05"] = reject_holm

    # Kolom "better" setelah Holm (gunakan p_holm untuk verdict)
    better_holm: List[str] = []
    for _, row in df_out.iterrows():
        dm_val = float(row["dm_stat"])
        p_h    = float(row["p_value_holm"])
        winner_label = f"{row['winner_config']}_{row['ablation']}_CSA"
        better_holm.append(_verdict(dm_val, p_h, winner_label, "naive_rw"))
    df_out["better_holm"] = better_holm

    return df_out


# ---------------------------------------------------------------------------
# Output: CSV + tabel markdown + verdict
# ---------------------------------------------------------------------------

def _print_markdown_table(df: pd.DataFrame) -> None:
    """Cetak tabel markdown ringkas ke stdout."""
    cols = [
        "horizon", "winner_config", "ablation", "n",
        "dm_stat", "p_value_raw", "significant_raw_0.05", "better_raw",
        "p_value_holm", "significant_holm_0.05", "better_holm",
    ]
    header = "| " + " | ".join(f"{c}" for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                vals.append("NaN")
            elif isinstance(v, (bool, np.bool_)):
                vals.append("Ya" if v else "Tidak")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")


def _compute_verdict(df: pd.DataFrame) -> None:
    """Hitung dan cetak verdict H4 final."""
    winner_labels = df.apply(lambda r: f"{r['winner_config']}_{r['ablation']}_CSA", axis=1)

    # Raw
    win_raw  = int((df["better_raw"].isin(winner_labels)).sum())
    lose_raw = int((df["better_raw"] == "naive_rw").sum())
    tie_raw  = int((df["better_raw"] == "tie").sum())

    # Holm
    win_holm  = int((df["better_holm"].isin(winner_labels)).sum())
    lose_holm = int((df["better_holm"] == "naive_rw").sum())
    tie_holm  = int((df["better_holm"] == "tie").sum())

    print(f"\nVerdict H4 — Winner CSA vs Naive Random Walk:")
    print(f"  [Raw  p < 0.05] Menang: {win_raw}/7, Kalah: {lose_raw}/7, Tie: {tie_raw}/7")
    print(f"  [Holm p < 0.05] Menang: {win_holm}/7, Kalah: {lose_holm}/7, Tie: {tie_holm}/7")
    print()
    if win_holm >= 4:
        judgment = "didukung (majority horizon signifikan)"
    elif win_holm >= 1:
        judgment = "parsial (sebagian horizon signifikan)"
    else:
        judgment = "ditolak (tidak ada horizon signifikan setelah Holm)"
    print(f"  H4 (sufficiency) setelah Holm: {judgment}")
    print()
    print(
        f"  PARAGRAF VERDICT: Winner CSA mengalahkan naive random walk secara "
        f"signifikan pada {win_raw} dari 7 horizon (raw, p<0.05) / "
        f"{win_holm} dari 7 (setelah Holm); kalah pada {lose_holm} (raw)/{lose_holm} (Holm); "
        f"tie pada {tie_raw} (raw)/{tie_holm} (Holm). "
        f"Hipotesis H4 (sufficiency) {judgment}."
    )


def main() -> None:
    print("=" * 70)
    print("h4_winner_csa_vs_naive.py — Uji H4 Winner CSA vs Naive Random Walk")
    print("=" * 70)

    df_result = run_h4_winner_csa_vs_naive()

    # Tulis CSV baru (tidak menimpa file lama)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV output ditulis: {OUTPUT_CSV}")

    # Cetak tabel markdown
    print("\n" + "=" * 70)
    print("Tabel Hasil DM Winner CSA vs Naive Random Walk:")
    print("=" * 70)
    _print_markdown_table(df_result)

    # Verdict H4 final
    print("\n" + "=" * 70)
    _compute_verdict(df_result)


if __name__ == "__main__":
    main()

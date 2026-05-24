"""
pt_directional.py - Pesaran-Timmermann directional test per konfigurasi ablation.

Untuk tiap (variant, horizon, konfigurasi C1-C4) menghitung PT(1992) atas
sign(Actual_LogReturn) vs sign(prediksi). Menjawab H3: konfigurasi ber-HMM
(C2, C4) menunjukkan directional skill yang signifikan di lebih banyak horizon
dibanding price-only (C1)?

Output: prediction/baselines/output/pt_directional_Daily_{split}.csv
Schema: horizon, split, variant, config, n, da_pct, pt_stat, p_value,
        significant_at_0.05
Usage: python prediction/baselines/pt_directional.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
if _PREDICTION_DIR not in sys.path:
    sys.path.insert(0, _PREDICTION_DIR)

from naive_baseline import pesaran_timmermann_test  # noqa: E402

_OUTPUTS_ROOT = os.path.join(_PREDICTION_DIR, "output_horizons")
ABLATIONS: Dict[str, str] = {
    "C1_cpo_only":      os.path.join(_OUTPUTS_ROOT, "cpo_only"),
    "C2_cpo_hmm":       os.path.join(_OUTPUTS_ROOT, "cpo_hmm"),
    "C3_cpo_sentiment": os.path.join(_OUTPUTS_ROOT, "cpo_sentiment"),
    "C4_full":          os.path.join(_OUTPUTS_ROOT, "full"),
}
VARIANTS: Tuple[str, ...] = ("base", "csa")
HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
SPLITS: Tuple[str, ...] = ("testing",)
INTERVAL = "Daily"
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


def _pred_col(variant: str) -> str:
    return f"xgboost_{variant}_LogReturn"


def _predictions_path(variant_dir: str, horizon: int, split: str) -> str:
    return os.path.join(variant_dir, INTERVAL, f"horizon_{horizon}",
                        f"{split}_predictions_{INTERVAL}_h{horizon}.csv")


def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in SPLITS:
        rows: List[Dict] = []
        for variant in VARIANTS:
            for config, cdir in ABLATIONS.items():
                for h in HORIZONS:
                    path = _predictions_path(cdir, h, split)
                    if not os.path.exists(path):
                        print(f"  [PT] skip {config} h{h} {variant}: missing {path}")
                        continue
                    try:
                        df = pd.read_csv(path)
                    except Exception as exc:
                        print(f"  [PT] skip {config} h{h}: read error {exc}")
                        continue
                    col = _pred_col(variant)
                    if "Actual_LogReturn" not in df.columns or col not in df.columns:
                        print(f"  [PT] skip {config} h{h} {variant}: kolom hilang")
                        continue
                    y_true = df["Actual_LogReturn"].to_numpy(dtype=float)
                    y_pred = df[col].to_numpy(dtype=float)
                    pt_stat, p_value, n = pesaran_timmermann_test(y_true, y_pred)
                    m = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0.0)
                    da = (float(np.mean(np.sign(y_true[m]) == np.sign(y_pred[m]))) * 100
                          if m.sum() else float("nan"))
                    rows.append({
                        "horizon": h, "split": split, "variant": variant,
                        "config": config, "n": n,
                        "da_pct": round(da, 4) if not np.isnan(da) else float("nan"),
                        "pt_stat": round(pt_stat, 6) if not np.isnan(pt_stat) else float("nan"),
                        "p_value": round(p_value, 6) if not np.isnan(p_value) else float("nan"),
                        "significant_at_0.05": bool(not np.isnan(p_value) and p_value < 0.05),
                    })
        cols = ["horizon", "split", "variant", "config", "n", "da_pct",
                "pt_stat", "p_value", "significant_at_0.05"]
        out = os.path.join(OUTPUT_DIR, f"pt_directional_{INTERVAL}_{split}.csv")
        pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
        print(f"  [PT] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    run()

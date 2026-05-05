"""Sweep NEUTRAL_TOLERANCE in 0.01 steps to maximize Cohen's kappa.

Scores L-M polarity once on the filtered article set, then re-buckets the
polarities under each tolerance threshold without re-tokenizing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from lm_finbert_agreement import (
    INPUT_CSV,
    MIN_YEAR,
    load_news,
    normalize_finbert,
    polarity_to_label,
    score_lm,
)

LABELS = ["negative", "neutral", "positive"]


def main() -> None:
    here = Path(__file__).resolve().parent
    df = load_news(here / INPUT_CSV)

    parsed = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.loc[parsed.dt.year >= MIN_YEAR].reset_index(drop=True)
    print(f"Articles after MIN_YEAR={MIN_YEAR} filter: {len(df)}")

    # Score L-M polarity once (slowest step).
    print("Scoring L-M polarity (one-time)...")
    _, polarity = score_lm(df["Content"])
    fb = normalize_finbert(df["Combined_Sentiment"])
    print("Done. Sweeping tolerances 0.00 -> 1.00 in 0.01 steps.")

    rows = []
    tolerances = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
    for tol in tolerances:
        lm = polarity.apply(lambda p, t=tol: polarity_to_label(p, t))
        agree = float((fb == lm).mean())
        kappa = float(cohen_kappa_score(fb, lm, labels=LABELS))
        rows.append({"tolerance": float(tol), "agreement": agree, "kappa": kappa})

    sweep_df = pd.DataFrame(rows)

    # Top 10 by kappa.
    top = sweep_df.sort_values("kappa", ascending=False).head(10)
    print("\nTop 10 tolerances by Cohen's kappa:")
    print(top.to_string(index=False, formatters={
        "tolerance": "{:.2f}".format,
        "agreement": "{:.4f}".format,
        "kappa":     "{:+.4f}".format,
    }))

    best = sweep_df.loc[sweep_df["kappa"].idxmax()]
    print(
        f"\nBest tolerance: {best['tolerance']:.2f}  "
        f"(kappa = {best['kappa']:+.4f},  agreement = {best['agreement']:.2%})"
    )

    # Show the kappa curve at coarse 0.05 intervals so the user can see the shape.
    coarse = sweep_df[(sweep_df["tolerance"] * 100).round().astype(int) % 5 == 0]
    print("\nKappa curve (every 0.05):")
    print(coarse.to_string(index=False, formatters={
        "tolerance": "{:.2f}".format,
        "agreement": "{:.4f}".format,
        "kappa":     "{:+.4f}".format,
    }))

    out_path = here / "lm_tolerance_sweep.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"\nFull sweep -> {out_path}")


if __name__ == "__main__":
    main()

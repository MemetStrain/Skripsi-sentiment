"""
sentiment_correlation_eval.py — Phase 2b of the FinBERT validation suite.

Validates FinBERT sentiment indirectly by measuring how well daily
aggregate sentiment correlates with next-day CPO log-return movement.

Reads existing artifacts (no FinBERT re-runs):
  - news/output/sentiment_aggregate_Daily.csv
  - cpo/output/cpo_variables_Daily.csv

Usage:
    python news/validation/sentiment_correlation_eval.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))

DEFAULT_SENTIMENT_PATH = os.path.join(
    _PROJECT_ROOT, "news", "output", "sentiment_aggregate_Daily.csv"
)
DEFAULT_PRICE_PATH = os.path.join(
    _PROJECT_ROOT, "cpo", "output", "cpo_variables_Daily.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")

GRANGER_LAGS = (1, 2, 3, 5)


# =============================================================================
# Data assembly
# =============================================================================

def _load_merged(sentiment_path: str, price_path: str) -> pd.DataFrame:
    """Inner-join sentiment and price on Date and compute next-day log return."""
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"Sentiment aggregate not found: {sentiment_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price file not found: {price_path}")

    sent = pd.read_csv(sentiment_path)
    sent["Date"] = pd.to_datetime(sent["Date"])
    sent_score_col = (
        "Sentiment_Score" if "Sentiment_Score" in sent.columns else None
    )
    if sent_score_col is None:
        raise ValueError(
            "Sentiment aggregate must have a 'Sentiment_Score' column"
        )

    price = pd.read_csv(price_path)
    price["Date"] = pd.to_datetime(price["Date"])
    price = price[["Date", "Close"]].sort_values("Date").reset_index(drop=True)

    merged = sent[["Date", sent_score_col]].merge(price, on="Date", how="inner")
    merged = merged.rename(columns={sent_score_col: "Sentiment_Score"})
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Next-day log return: log(Close_{t+1} / Close_t)
    merged["Next_Day_LogReturn"] = np.log(merged["Close"].shift(-1) / merged["Close"])
    merged = merged.dropna(subset=["Sentiment_Score", "Next_Day_LogReturn"])
    merged = merged.reset_index(drop=True)
    return merged


# =============================================================================
# Analyses
# =============================================================================

def _pearson_spearman(merged: pd.DataFrame) -> Dict[str, float]:
    from scipy import stats

    s = merged["Sentiment_Score"].to_numpy()
    r = merged["Next_Day_LogReturn"].to_numpy()
    pr_corr, pr_p = stats.pearsonr(s, r)
    sp_corr, sp_p = stats.spearmanr(s, r)
    return {
        "pearson_corr": float(pr_corr),
        "pearson_p": float(pr_p),
        "spearman_corr": float(sp_corr),
        "spearman_p": float(sp_p),
    }


def _granger_pvalues(merged: pd.DataFrame) -> Dict[int, float]:
    """Granger causality: does Sentiment_t cause Return_{t+1}?

    statsmodels expects a 2-column array where the FIRST column is the
    series being predicted. So we feed [Next_Day_LogReturn, Sentiment_Score].
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    arr = merged[["Next_Day_LogReturn", "Sentiment_Score"]].to_numpy()
    out: Dict[int, float] = {}
    for lag in GRANGER_LAGS:
        try:
            res = grangercausalitytests(arr, maxlag=lag, verbose=False)
            # ssr_ftest p-value at the requested lag
            out[lag] = float(res[lag][0]["ssr_ftest"][1])
        except Exception as exc:
            print(f"  ! Granger lag={lag} failed: {exc}")
            out[lag] = float("nan")
    return out


def _directional_agreement(merged: pd.DataFrame) -> float:
    """% of days where sign(sentiment) == sign(next-day return)."""
    s = np.sign(merged["Sentiment_Score"].to_numpy())
    r = np.sign(merged["Next_Day_LogReturn"].to_numpy())
    nonzero = (s != 0) & (r != 0)
    if nonzero.sum() == 0:
        return float("nan")
    return float((s[nonzero] == r[nonzero]).mean() * 100.0)


def _interpret(metrics: Dict[str, float]) -> str:
    pr = metrics["pearson_corr"]
    pr_p = metrics["pearson_p"]
    g1 = metrics.get("granger_lag1_p", float("nan"))

    if abs(pr) < 0.05:
        size = "weak"
    elif abs(pr) < 0.2:
        size = "moderate"
    else:
        size = "strong"

    direction = "positive" if pr >= 0 else "negative"
    sig = "significant" if (not np.isnan(pr_p) and pr_p < 0.05) else "not significant"

    if not np.isnan(g1):
        granger = (
            f"Granger causality at lag 1 (p={g1:.4f}) "
            f"{'supports' if g1 < 0.05 else 'does not support'} predictive direction."
        )
    else:
        granger = "Granger causality at lag 1 could not be computed."

    return (
        f"Pearson correlation between daily sentiment and next-day log return is "
        f"{pr:+.4f} ({direction}, {sig}). Effect size of |{pr:.4f}| indicates a "
        f"{size} relationship. {granger}"
    )


# =============================================================================
# Plot
# =============================================================================

def _scatter_plot(merged: pd.DataFrame, png_path: str, pearson: float) -> None:
    import matplotlib.pyplot as plt

    s = merged["Sentiment_Score"].to_numpy()
    r = merged["Next_Day_LogReturn"].to_numpy()
    if len(s) >= 2:
        slope, intercept = np.polyfit(s, r, 1)
    else:
        slope = intercept = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s, r, s=8, alpha=0.4, color="#2E86AB", edgecolors="none")
    xs = np.linspace(s.min(), s.max(), 200) if len(s) else np.array([0])
    ax.plot(xs, slope * xs + intercept, color="red", lw=1.4,
            label=f"OLS fit (slope={slope:.4f})")
    ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.7)
    ax.axvline(0, color="black", lw=0.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Daily Sentiment_Score (FinBERT aggregate)")
    ax.set_ylabel("Next-day log return")
    ax.set_title(
        f"Sentiment vs next-day return  (Pearson r = {pearson:+.4f}, n = {len(s):,})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Public entry point
# =============================================================================

def evaluate(
    output_dir: str,
    sentiment_path: str = DEFAULT_SENTIMENT_PATH,
    price_path: str = DEFAULT_PRICE_PATH,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Phase 2b: Sentiment vs next-day price correlation")
    print("=" * 65)

    merged = _load_merged(sentiment_path, price_path)
    n = len(merged)
    print(f"  Merged {n:,} matched daily rows "
          f"({merged['Date'].min().date()} → {merged['Date'].max().date()})")
    if n < 30:
        print(f"  ! WARNING: too few rows ({n}); correlation may be unreliable")

    corr = _pearson_spearman(merged)
    granger = _granger_pvalues(merged)
    da_pct = _directional_agreement(merged)

    metrics: Dict[str, float] = {
        "pearson_corr": corr["pearson_corr"],
        "pearson_p": corr["pearson_p"],
        "spearman_corr": corr["spearman_corr"],
        "spearman_p": corr["spearman_p"],
        "granger_lag1_p": granger.get(1, float("nan")),
        "granger_lag2_p": granger.get(2, float("nan")),
        "granger_lag3_p": granger.get(3, float("nan")),
        "granger_lag5_p": granger.get(5, float("nan")),
        "directional_agreement_pct": da_pct,
        "n": n,
    }

    interp = _interpret(metrics)

    rows: List[Dict] = [
        {"metric": "pearson_corr", "value": round(corr["pearson_corr"], 4),
         "p_value": round(corr["pearson_p"], 4),
         "interpretation": "Linear correlation (Sentiment_t vs Next_Day_LogReturn)"},
        {"metric": "spearman_corr", "value": round(corr["spearman_corr"], 4),
         "p_value": round(corr["spearman_p"], 4),
         "interpretation": "Rank correlation"},
        {"metric": "granger_lag1_pvalue", "value": "",
         "p_value": _round_or_nan(granger.get(1)),
         "interpretation": "Sentiment_t Granger-causes Next_Day_LogReturn at lag 1"},
        {"metric": "granger_lag2_pvalue", "value": "",
         "p_value": _round_or_nan(granger.get(2)),
         "interpretation": "Sentiment_t Granger-causes Next_Day_LogReturn at lag 2"},
        {"metric": "granger_lag3_pvalue", "value": "",
         "p_value": _round_or_nan(granger.get(3)),
         "interpretation": "Sentiment_t Granger-causes Next_Day_LogReturn at lag 3"},
        {"metric": "granger_lag5_pvalue", "value": "",
         "p_value": _round_or_nan(granger.get(5)),
         "interpretation": "Sentiment_t Granger-causes Next_Day_LogReturn at lag 5"},
        {"metric": "directional_agreement_pct", "value": round(da_pct, 4),
         "p_value": "",
         "interpretation": "% days where sign(sentiment) == sign(next-day return)"},
        {"metric": "sample_size", "value": n, "p_value": "",
         "interpretation": "Daily observations after inner-join + dropna"},
        {"metric": "narrative", "value": "", "p_value": "",
         "interpretation": interp},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "sentiment_correlation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    png_path = os.path.join(output_dir, "sentiment_vs_nextday_return.png")
    _scatter_plot(merged, png_path, corr["pearson_corr"])
    print(f"  → {png_path}")

    print(f"\n  Pearson r  = {corr['pearson_corr']:+.4f}  (p = {corr['pearson_p']:.4f})")
    print(f"  Spearman ρ = {corr['spearman_corr']:+.4f}  (p = {corr['spearman_p']:.4f})")
    for lag in GRANGER_LAGS:
        print(f"  Granger p (lag {lag}) = {granger[lag]:.4f}")
    print(f"  Directional agreement = {da_pct:.2f}%")
    print(f"\n  {interp}")

    return metrics


def _round_or_nan(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return round(float(value), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2b: Sentiment-price correlation")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sentiment", default=DEFAULT_SENTIMENT_PATH)
    parser.add_argument("--price", default=DEFAULT_PRICE_PATH)
    args = parser.parse_args()
    evaluate(os.path.abspath(args.output_dir),
             sentiment_path=args.sentiment, price_path=args.price)
    return 0


if __name__ == "__main__":
    sys.exit(main())

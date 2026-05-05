"""Compare FinBERT vs Loughran-McDonald daily sentiment against CPO returns.

Reads ``news/lm_finbert_comparison.csv`` (per-article FinBERT and L-M labels)
and ``cpo/Data_CPO_Daily.csv`` (daily CPO close), aggregates sentiment to a
daily signed score per method, aligns with same-day and next-trading-day
returns, and reports which method tracks price moves better.

Outputs:
    - ``news/output/daily_sentiment_vs_price.csv``: merged daily frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


COMPARISON_CSV = "news/lm_finbert_comparison.csv"
CPO_CSV = "cpo/Data_CPO_Daily.csv"
OUTPUT_CSV = "news/output/daily_sentiment_vs_price.csv"

# Map sentiment labels to a signed score so we can average per day.
LABEL_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def load_sentiment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "Combined_Sentiment", "LM_Polarity"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: comparison CSV missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    df["finbert_score"] = df["Combined_Sentiment"].str.lower().map(LABEL_TO_SCORE)
    # Re-derive the L-M label from raw polarity with a strict 0 dead-band so
    # the price-tracking comparison is independent of NEUTRAL_TOLERANCE used
    # in the agreement script.
    df["lm_score"] = np.sign(df["LM_Polarity"].astype(float)).astype(float)
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Mean signed sentiment per day for each method, plus article count."""
    daily = (
        df.groupby(df["Date"].dt.normalize())
        .agg(
            article_count=("Date", "size"),
            finbert_daily=("finbert_score", "mean"),
            lm_daily=("lm_score", "mean"),
        )
        .reset_index()
        .rename(columns={"Date": "Date"})
    )
    return daily


def _parse_locale_number(value: str) -> float:
    """CPO file uses '.' as thousands and ',' as decimal (e.g. '4.515,00')."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


FORWARD_HORIZONS = (1, 2, 3, 4, 5,6,7)  # cumulative trading-day windows after t


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Tanggal" not in df.columns or "Terakhir" not in df.columns:
        raise SystemExit(f"ERROR: CPO CSV missing Tanggal/Terakhir columns")
    df["Date"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["close"] = df["Terakhir"].map(_parse_locale_number)
    df["return_t"] = df["close"].pct_change()  # return on day t (close vs prev close)
    # Cumulative forward returns: close[t+N] / close[t] - 1 for each horizon.
    for n in FORWARD_HORIZONS:
        df[f"cum_{n}d"] = df["close"].shift(-n) / df["close"] - 1.0
    cols = ["Date", "close", "return_t"] + [f"cum_{n}d" for n in FORWARD_HORIZONS]
    return df[cols]


def directional_accuracy(sent: pd.Series, ret: pd.Series) -> float:
    """Share of days where signs match, ignoring rows where either is zero/NaN."""
    mask = sent.notna() & ret.notna() & (sent != 0) & (ret != 0)
    if mask.sum() == 0:
        return float("nan")
    s = np.sign(sent[mask].to_numpy())
    r = np.sign(ret[mask].to_numpy())
    return float((s == r).mean())


def report_method(name: str, sent: pd.Series, returns: dict[str, pd.Series]) -> None:
    print(f"\n--- {name} ---")
    for ret_name, ret in returns.items():
        mask = sent.notna() & ret.notna()
        n = int(mask.sum())
        if n < 3:
            print(f"  {ret_name:<10}: too few overlapping days ({n})")
            continue
        pear_r, pear_p = pearsonr(sent[mask], ret[mask])
        spear_r, spear_p = spearmanr(sent[mask], ret[mask])
        dacc = directional_accuracy(sent, ret)
        dacc_str = "n/a" if np.isnan(dacc) else f"{dacc:.2%}"
        print(
            f"  {ret_name:<10}: n={n:<5} "
            f"pearson r={pear_r:+.4f} (p={pear_p:.3f})  "
            f"spearman rho={spear_r:+.4f} (p={spear_p:.3f})  "
            f"dir.acc={dacc_str}"
        )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sent_path = root / COMPARISON_CSV
    cpo_path = root / CPO_CSV
    out_path = root / OUTPUT_CSV

    if not sent_path.exists():
        raise SystemExit(f"ERROR: not found: {sent_path}")
    if not cpo_path.exists():
        raise SystemExit(f"ERROR: not found: {cpo_path}")

    articles = load_sentiment(sent_path)
    daily_sent = aggregate_daily(articles)
    prices = load_prices(cpo_path)

    merged = daily_sent.merge(prices, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("=" * 70)
    print("Daily sentiment vs CPO price movement")
    print("=" * 70)
    print(f"Article rows           : {len(articles)}")
    print(f"Distinct news days     : {len(daily_sent)}")
    print(f"Trading days w/ news   : {len(merged)}")
    print(f"Date range             : {merged['Date'].min().date()} -> {merged['Date'].max().date()}")
    print(f"Merged daily frame -> {out_path}")

    returns = {"same-day": merged["return_t"]}  # news on day t vs return on day t
    for n in FORWARD_HORIZONS:
        # Cumulative return from close[t] to close[t+n].
        returns[f"+{n}d cum"] = merged[f"cum_{n}d"]

    report_method("FinBERT", merged["finbert_daily"], returns)
    report_method("Loughran-McDonald", merged["lm_daily"], returns)

    print()
    print("Note: 'dir.acc' = share of days where sign(sentiment) matches sign(return),")
    print("      ignoring days where either side is exactly 0 or missing.")


if __name__ == "__main__":
    main()

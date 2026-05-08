"""Compare FinBERT vs Loughran-McDonald daily sentiment against CPO returns.

Reads ``news/lm_finbert_comparison.csv`` (per-article FinBERT and L-M labels)
and ``cpo/Data_CPO_Daily.csv`` (daily CPO close), aggregates sentiment to a
daily signed score per method, then sweeps lags 0..MAX_LAG to find the lag
at which sentiment best predicts future price returns.

Outputs:
    - ``news/output/daily_sentiment_vs_price.csv``: merged daily frame.
    - ``news/output/lag_search_results.csv``: correlation stats for every lag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


COMPARISON_CSV = "news/lm_finbert_comparison.csv"
CPO_CSV = "cpo/Data_CPO_Daily.csv"
OUTPUT_CSV = "news/output/daily_sentiment_vs_price.csv"
LAG_CSV = "news/output/lag_search_results.csv"

LABEL_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# Maximum lag (in trading days) to search. Lag 0 = same-day.
MAX_LAG = 30


def load_sentiment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "Combined_Sentiment", "LM_Polarity"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: comparison CSV missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    df["finbert_score"] = df["Combined_Sentiment"].str.lower().map(LABEL_TO_SCORE)
    df["lm_score"] = np.sign(df["LM_Polarity"].astype(float)).astype(float)
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
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


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Tanggal" not in df.columns or "Terakhir" not in df.columns:
        raise SystemExit("ERROR: CPO CSV missing Tanggal/Terakhir columns")
    df["Date"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["close"] = df["Terakhir"].map(_parse_locale_number)
    df["return_t"] = df["close"].pct_change()
    return df[["Date", "close", "return_t"]]


def directional_accuracy(sent: pd.Series, ret: pd.Series) -> float:
    mask = sent.notna() & ret.notna() & (sent != 0) & (ret != 0)
    if mask.sum() == 0:
        return float("nan")
    s = np.sign(sent[mask].to_numpy())
    r = np.sign(ret[mask].to_numpy())
    return float((s == r).mean())


def _corr_at_lag(sent: pd.Series, ret: pd.Series, lag: int) -> dict:
    """Return Pearson r, Spearman rho, and dir.acc for sentiment vs return at lag."""
    shifted_ret = ret.shift(-lag)  # return that occurs `lag` trading days after t
    mask = sent.notna() & shifted_ret.notna()
    n = int(mask.sum())
    if n < 5:
        return dict(lag=lag, n=n, pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan, dir_acc=np.nan)
    s = sent[mask]
    r = shifted_ret[mask]
    pear_r, pear_p = pearsonr(s, r)
    spear_r, spear_p = spearmanr(s, r)
    dacc = directional_accuracy(sent, shifted_ret)
    return dict(lag=lag, n=n,
                pearson_r=pear_r, pearson_p=pear_p,
                spearman_r=spear_r, spearman_p=spear_p,
                dir_acc=dacc)


def find_best_lag(name: str, sent: pd.Series, ret: pd.Series, max_lag: int) -> pd.DataFrame:
    """Sweep lags 0..max_lag, print a table, and return the results DataFrame."""
    rows = [_corr_at_lag(sent, ret, lag) for lag in range(max_lag + 1)]
    results = pd.DataFrame(rows)
    results["method"] = name

    print(f"\n{'='*72}")
    print(f"  {name}  — lag sweep (0..{max_lag} trading days)")
    print(f"{'='*72}")
    print(f"  {'lag':>4}  {'n':>5}  {'pearson r':>10}  {'p':>7}  "
          f"{'spearman':>10}  {'p':>7}  {'dir.acc':>8}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*8}")

    best_idx = results["pearson_r"].abs().idxmax() if results["pearson_r"].notna().any() else None

    for _, row in results.iterrows():
        marker = " <-- best" if row["lag"] == results.loc[best_idx, "lag"] else ""
        dacc_str = "n/a" if np.isnan(row["dir_acc"]) else f"{row['dir_acc']:.2%}"
        pear_str = "n/a" if np.isnan(row["pearson_r"]) else f"{row['pearson_r']:+.4f}"
        spear_str = "n/a" if np.isnan(row["spearman_r"]) else f"{row['spearman_r']:+.4f}"
        pear_p_str = "n/a" if np.isnan(row["pearson_p"]) else f"{row['pearson_p']:.4f}"
        spear_p_str = "n/a" if np.isnan(row["spearman_p"]) else f"{row['spearman_p']:.4f}"
        print(
            f"  {int(row['lag']):>4}  {int(row['n']):>5}  {pear_str:>10}  "
            f"{pear_p_str:>7}  {spear_str:>10}  {spear_p_str:>7}  {dacc_str:>8}{marker}"
        )

    if best_idx is not None:
        best = results.loc[best_idx]
        print(f"\n  >> Best lag for {name}: {int(best['lag'])} trading day(s)  "
              f"pearson r={best['pearson_r']:+.4f} (p={best['pearson_p']:.4f})  "
              f"spearman rho={best['spearman_r']:+.4f} (p={best['spearman_p']:.4f})")

    return results


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sent_path = root / COMPARISON_CSV
    cpo_path = root / CPO_CSV
    out_path = root / OUTPUT_CSV
    lag_path = root / LAG_CSV

    if not sent_path.exists():
        raise SystemExit(f"ERROR: not found: {sent_path}")
    if not cpo_path.exists():
        raise SystemExit(f"ERROR: not found: {cpo_path}")

    articles = load_sentiment(sent_path)
    daily_sent = aggregate_daily(articles)
    prices = load_prices(cpo_path)

    merged = (
        daily_sent.merge(prices, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("=" * 72)
    print("Daily sentiment vs CPO price — best-lag search")
    print("=" * 72)
    print(f"Article rows           : {len(articles)}")
    print(f"Distinct news days     : {len(daily_sent)}")
    print(f"Trading days w/ news   : {len(merged)}")
    print(f"Date range             : {merged['Date'].min().date()} -> {merged['Date'].max().date()}")
    print(f"Lag range searched     : 0 .. {MAX_LAG} trading days")
    print(f"Merged daily frame  -> {out_path}")

    fb_results = find_best_lag("FinBERT", merged["finbert_daily"], merged["return_t"], MAX_LAG)
    lm_results = find_best_lag("Loughran-McDonald", merged["lm_daily"], merged["return_t"], MAX_LAG)

    all_results = pd.concat([fb_results, lm_results], ignore_index=True)
    all_results.to_csv(lag_path, index=False)
    print(f"\nFull lag results -> {lag_path}")

    print()
    print("Note: lag k means sentiment on day t is compared to the return on day t+k.")
    print("      dir.acc = share of days where sign(sentiment) == sign(return),")
    print("      excluding days where either value is exactly 0 or missing.")


if __name__ == "__main__":
    main()

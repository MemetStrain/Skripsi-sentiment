"""HMM state vs forward CPO price return — lag search.

Reads:
  markov/output/hmm_states_results_Daily.csv  — per-day state assignments
  markov/output/hmm_states_stats_Daily.csv    — state ordering / labels

Encodes each HMM regime as a signed numeric score derived from its rank by
average log-return (most-bullish state = +1, most-bearish = -1), then sweeps
lags 0..MAX_LAG to identify the trading-day offset at which the regime label
best predicts forward price returns.

Correlation metrics at each lag:
  - Pearson r        : linear association between state score and forward return
  - Spearman rho     : rank-based (better for ordinal states)
  - Directional acc  : share of days sign(state_score) == sign(forward_return)
  - ANOVA F / p      : do mean forward returns differ significantly across states?

Outputs:
  markov/output/hmm_lag_search.csv        — full lag-by-lag stats
  markov/output/hmm_lag_state_means.csv   — mean return per state at the best lag
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, f_oneway


STATES_CSV    = "output/hmm_states_results_Daily.csv"
STATE_STATS_CSV = "output/hmm_states_stats_Daily.csv"
OUTPUT_LAG_CSV   = "output/hmm_lag_search.csv"
OUTPUT_MEANS_CSV = "output/hmm_lag_state_means.csv"

# Maximum lag (trading days) to search. Lag 0 = same day.
MAX_LAG = 60


def _build_state_score_map(stats_path: Path) -> dict[int, float]:
    """Map original HMM state int -> signed score in [-1, +1].

    States are already sorted by Avg_LogReturn descending in the stats CSV
    (index 0 = most bullish).  We assign scores linearly from +1 (best) to
    -1 (worst).  For 3 states: [+1, 0, -1].  For 2: [+1, -1].  Etc.
    """
    stats = pd.read_csv(stats_path)
    # stats is sorted bull->bear; index in this sorted order gives rank
    n = len(stats)
    scores = {}
    for rank, (_, row) in enumerate(stats.iterrows()):
        if n == 1:
            score = 0.0
        else:
            score = 1.0 - 2.0 * rank / (n - 1)   # +1 .. -1 evenly spaced
        scores[int(row["State"])] = round(score, 4)
    return scores, stats


def _directional_accuracy(state_score: pd.Series, fwd_ret: pd.Series) -> float:
    mask = state_score.notna() & fwd_ret.notna() & (state_score != 0) & (fwd_ret != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(state_score[mask].values) == np.sign(fwd_ret[mask].values)).mean())


def _anova(state_label: pd.Series, fwd_ret: pd.Series) -> tuple[float, float]:
    """One-way ANOVA: does mean forward return differ significantly across regimes?"""
    mask = state_label.notna() & fwd_ret.notna()
    groups = [fwd_ret[mask & (state_label == lbl)].values
              for lbl in state_label[mask].unique()]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan"), float("nan")
    f, p = f_oneway(*groups)
    return float(f), float(p)


def corr_at_lag(state_score: pd.Series, state_label: pd.Series,
                close: pd.Series, lag: int) -> dict:
    """Correlate state_score[t] with cumulative forward return from t to t+lag."""
    fwd_ret = close.shift(-lag) / close - 1.0   # k-day forward return
    mask = state_score.notna() & fwd_ret.notna()
    n = int(mask.sum())
    if n < 5:
        return dict(lag=lag, n=n,
                    pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan,
                    dir_acc=np.nan, anova_f=np.nan, anova_p=np.nan)

    s = state_score[mask]
    r = fwd_ret[mask]
    pear_r, pear_p   = pearsonr(s, r)
    spear_r, spear_p = spearmanr(s, r)
    dacc             = _directional_accuracy(state_score, fwd_ret)
    f_stat, f_p      = _anova(state_label, fwd_ret)

    return dict(lag=lag, n=n,
                pearson_r=pear_r, pearson_p=pear_p,
                spearman_r=spear_r, spearman_p=spear_p,
                dir_acc=dacc, anova_f=f_stat, anova_p=f_p)


def state_means_at_lag(df: pd.DataFrame, lag: int, stats: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return by state label at a specific lag."""
    df = df.copy()
    df["fwd_ret"] = df["Close"].shift(-lag) / df["Close"] - 1.0
    grouped = (
        df.dropna(subset=["fwd_ret"])
        .groupby("State_Label")["fwd_ret"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean_Fwd_Return", "std": "Std_Fwd_Return",
                         "count": "N"})
    )
    # Attach the state score for reference
    label_to_score = dict(zip(stats["Label"], stats["State"].map(
        {int(r["State"]): r.get("score", np.nan) for _, r in stats.iterrows()}
    )))
    grouped["lag"] = lag
    return grouped


def print_lag_table(results: pd.DataFrame, best_idx: int) -> None:
    best_lag = results.loc[best_idx, "lag"]
    print(f"\n  {'lag':>4}  {'n':>5}  {'pearson r':>10}  {'p':>7}  "
          f"{'spearman':>10}  {'p':>7}  {'dir.acc':>8}  "
          f"{'ANOVA F':>8}  {'p':>7}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*8}  "
          f"{'-'*8}  {'-'*7}")
    for _, row in results.iterrows():
        marker = " <-- best" if row["lag"] == best_lag else ""
        def _fmt(v, fmt): return "n/a" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        print(
            f"  {int(row['lag']):>4}  {int(row['n']):>5}  "
            f"{_fmt(row['pearson_r'], '{:+.4f}'):>10}  "
            f"{_fmt(row['pearson_p'], '{:.4f}'):>7}  "
            f"{_fmt(row['spearman_r'], '{:+.4f}'):>10}  "
            f"{_fmt(row['spearman_p'], '{:.4f}'):>7}  "
            f"{_fmt(row['dir_acc'], '{:.2%}'):>8}  "
            f"{_fmt(row['anova_f'], '{:.3f}'):>8}  "
            f"{_fmt(row['anova_p'], '{:.4f}'):>7}"
            f"{marker}"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    states_path    = here / STATES_CSV
    stats_path     = here / STATE_STATS_CSV
    out_lag_path   = here / OUTPUT_LAG_CSV
    out_means_path = here / OUTPUT_MEANS_CSV

    for p in (states_path, stats_path):
        if not p.exists():
            raise SystemExit(f"ERROR: not found: {p}\nRun cpo_hmm_states.py first.")

    # ── Load ────────────────────────────────────────────────────────────────────
    score_map, stats = _build_state_score_map(stats_path)
    df = pd.read_csv(states_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["state_score"] = df["State"].map(score_map)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("HMM State vs Forward CPO Return — Lag Search")
    print("=" * 72)
    print(f"Records          : {len(df)}")
    print(f"Date range       : {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Lag range        : 0 .. {MAX_LAG} trading days")
    print(f"States found     : {sorted(df['State'].unique())}")
    print()
    print("State numeric encoding (bull=+1 .. bear=-1):")
    for orig_state, score in sorted(score_map.items()):
        label = df.loc[df["State"] == orig_state, "State_Label"].iloc[0] \
                if (df["State"] == orig_state).any() else "?"
        print(f"  State {orig_state} ({label:20s}) -> {score:+.4f}")

    # ── Lag sweep ───────────────────────────────────────────────────────────────
    rows = [
        corr_at_lag(df["state_score"], df["State_Label"], df["Close"], lag)
        for lag in range(MAX_LAG + 1)
    ]
    results = pd.DataFrame(rows)

    # Best lag by absolute Pearson r (most linear signal)
    best_pearson_idx = results["pearson_r"].abs().idxmax() \
        if results["pearson_r"].notna().any() else 0

    # Best lag by ANOVA F (strongest regime separation)
    best_anova_idx = results["anova_f"].idxmax() \
        if results["anova_f"].notna().any() else 0

    print(f"\n{'='*72}")
    print(f"  HMM States — lag sweep (0..{MAX_LAG} trading days)")
    print(f"{'='*72}")
    print_lag_table(results, best_pearson_idx)

    # ── Best-lag summary ────────────────────────────────────────────────────────
    def _best_summary(label, idx):
        row = results.loc[idx]
        print(f"\n  >> Best lag ({label}): {int(row['lag'])} trading day(s)")
        print(f"     Pearson  r = {row['pearson_r']:+.4f}  (p={row['pearson_p']:.4f})")
        print(f"     Spearman r = {row['spearman_r']:+.4f}  (p={row['spearman_p']:.4f})")
        print(f"     Dir. acc   = {row['dir_acc']:.2%}")
        print(f"     ANOVA F    = {row['anova_f']:.3f}  (p={row['anova_p']:.4f})")

    _best_summary("max |Pearson r|", best_pearson_idx)
    if best_anova_idx != best_pearson_idx:
        _best_summary("max ANOVA F", best_anova_idx)

    # ── Mean return per state at both best lags ──────────────────────────────────
    for label, idx in [("Pearson", best_pearson_idx), ("ANOVA", best_anova_idx)]:
        lag = int(results.loc[idx, "lag"])
        df_tmp = df.copy()
        df_tmp["fwd_ret"] = df_tmp["Close"].shift(-lag) / df_tmp["Close"] - 1.0
        means = (
            df_tmp.dropna(subset=["fwd_ret"])
            .groupby("State_Label")["fwd_ret"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        means["lag"] = lag
        means["mean_pct"] = (means["mean"] * 100).round(3)
        means["std_pct"]  = (means["std"]  * 100).round(3)
        print(f"\n  Mean {lag}-day forward return by HMM state (best lag by {label}):")
        print(f"  {'State':<22}  {'n':>5}  {'mean %':>8}  {'std %':>8}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*8}")
        for _, r in means.iterrows():
            print(f"  {r['State_Label']:<22}  {int(r['n']):>5}  {r['mean_pct']:>8.3f}  {r['std_pct']:>8.3f}")

    # ── Save ────────────────────────────────────────────────────────────────────
    results.to_csv(out_lag_path, index=False)
    print(f"\nFull lag results   -> {out_lag_path}")

    # Save state means at best Pearson lag
    best_lag = int(results.loc[best_pearson_idx, "lag"])
    df["fwd_ret_best"] = df["Close"].shift(-best_lag) / df["Close"] - 1.0
    state_means = (
        df.dropna(subset=["fwd_ret_best"])
        .groupby("State_Label")["fwd_ret_best"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    state_means["best_lag"] = best_lag
    state_means["state_score"] = state_means["State_Label"].map(
        dict(zip(df["State_Label"], df["state_score"]))
    )
    state_means.to_csv(out_means_path, index=False)
    print(f"State means (best lag) -> {out_means_path}")

    print()
    print("Notes:")
    print("  lag k  : state on day t is compared to cumulative return from t to t+k")
    print("  score  : +1 = most bullish state, -1 = most bearish (evenly spaced)")
    print("  ANOVA  : tests whether mean forward return differs across all states")
    print("  dir.acc: share of days where sign(state_score) == sign(forward_return),")
    print("           excluding neutral (score=0) or zero-return days")


if __name__ == "__main__":
    main()

"""
EDA over the canonical modeling window (2015-08-03 to 2026-02-28).

Loads the three primary data files, restricts each to FULL_START..FULL_END,
aligns them on Date, and characterises each variable's distribution and its
relationship to the target y = ln(Close_t / Close_{t-1}).

Produces:
  eda/figures/   -- time-series, histogram+KDE, QQ-plot, scatter/boxplot figures
  eda/           -- eda_summary_table.csv   (per-variable stats + test recommendation)

Run from the project root:
    <python> eda/eda_full_window.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from config.dates import (
    FULL_START_TS, FULL_END_TS,
    TRAIN_END_TS, TEST_START_TS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CPO_FILE       = _ROOT / "cpo/output/cpo_variables_Daily.csv"
SENTIMENT_FILE = _ROOT / "news/output/sentiment_aggregate_Daily_title.csv"
HMM_FILE       = _ROOT / "markov/output/hmm_states_results_Daily.csv"
FIG_DIR        = _HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)

TRAIN_BOUNDARY = TRAIN_END_TS   # vertical line in time-series plots
TEST_BOUNDARY  = TEST_START_TS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _window(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (df.dropna(subset=["Date"])
            .sort_values("Date")
            .drop_duplicates(subset=["Date"])
            .reset_index(drop=True))
    assert df["Date"].is_monotonic_increasing, "Date not monotonic after dedup"
    df = df[(df["Date"] >= FULL_START_TS) & (df["Date"] <= FULL_END_TS)].reset_index(drop=True)
    return df


def load_data() -> pd.DataFrame:
    """Load and align all three sources to the canonical window."""
    cpo  = _window(pd.read_csv(CPO_FILE))
    sent = _window(pd.read_csv(SENTIMENT_FILE))
    hmm  = _window(pd.read_csv(HMM_FILE))

    # Compute log-return target on the windowed price series
    cpo["Log_Return"] = np.log(cpo["Close"] / cpo["Close"].shift(1))

    # Sentiment: use Sentiment_Score if present, else derive from probs
    if "Sentiment_Score" not in sent.columns:
        if "Title_Positive_Prob" in sent.columns and "Title_Negative_Prob" in sent.columns:
            sent["Sentiment_Score"] = sent["Title_Positive_Prob"] - sent["Title_Negative_Prob"]
        else:
            raise KeyError("Cannot find Sentiment_Score in sentiment file")

    sent_slim = sent[["Date", "Sentiment_Score", "Article_Count"]].copy()
    hmm_slim  = hmm[["Date", "State", "State_Label"]].copy()

    merged = (cpo[["Date", "Close", "Log_Return"]]
              .merge(sent_slim, on="Date", how="left")
              .merge(hmm_slim,  on="Date", how="left"))

    print(f"Merged frame: {len(merged)} rows  "
          f"({merged['Date'].min().date()} to {merged['Date'].max().date()})")
    missing_sent = merged["Sentiment_Score"].isna().sum()
    missing_hmm  = merged["State"].isna().sum()
    print(f"  Days without sentiment: {missing_sent}")
    print(f"  Days without HMM state: {missing_hmm}")
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _desc(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "count":    int(len(s)),
        "mean":     float(s.mean()),
        "std":      float(s.std()),
        "min":      float(s.min()),
        "q25":      float(s.quantile(0.25)),
        "median":   float(s.median()),
        "q75":      float(s.quantile(0.75)),
        "max":      float(s.max()),
        "skewness": float(stats.skew(s)),
        "kurtosis": float(stats.kurtosis(s)),
    }


def normality_tests(series: pd.Series, name: str) -> dict:
    """Shapiro-Wilk (capped at 5000) and D'Agostino/Jarque-Bera."""
    s = series.dropna().values
    result = {"variable": name}

    # Shapiro-Wilk (unreliable for n > 5000)
    n_sw = min(len(s), 5000)
    rng = np.random.default_rng(42)
    sample = rng.choice(s, n_sw, replace=False) if len(s) > n_sw else s
    sw_stat, sw_p = stats.shapiro(sample)
    result["SW_stat"]  = round(float(sw_stat), 4)
    result["SW_p"]     = round(float(sw_p), 6)
    result["SW_note"]  = f"n={n_sw} (subsampled)" if len(s) > n_sw else f"n={n_sw}"

    # D'Agostino-Pearson omnibus
    if len(s) >= 20:
        da_stat, da_p = stats.normaltest(s)
        result["DA_stat"] = round(float(da_stat), 4)
        result["DA_p"]    = round(float(da_p), 6)
    else:
        result["DA_stat"] = float("nan")
        result["DA_p"]    = float("nan")

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(s)
    result["JB_stat"] = round(float(jb_stat), 4)
    result["JB_p"]    = round(float(jb_p), 6)

    normal = (sw_p > 0.05) and (da_p > 0.05 if not np.isnan(da_p) else True)
    result["normal_verdict"] = "normal" if normal else "NOT normal"
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_timeseries(df: pd.DataFrame, col: str, label: str, fname: str,
                    ylabel: str = "", color: str = "#1f77b4") -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], df[col], lw=0.8, color=color, alpha=0.85)
    ax.axvline(TRAIN_BOUNDARY, color="red", ls="--", lw=1.2, label="Train/Test cut (2024-12-31)")
    ax.set_title(f"{label} -- Full Window ({FULL_START_TS.date()} to {FULL_END_TS.date()})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel or label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / fname}")


def plot_dist_qq(series: pd.Series, label: str, fname: str) -> None:
    s = series.dropna()
    fig = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(s, bins=60, density=True, color="#4878cf", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(s.min(), s.max(), 300)
    kde = stats.gaussian_kde(s)
    ax1.plot(kde_x, kde(kde_x), color="#c44e52", lw=1.8)
    ax1.set_title(f"{label} -- Histogram + KDE", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    stats.probplot(s, dist="norm", plot=ax2)
    ax2.set_title("QQ-Plot vs Normal", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / fname}")


def plot_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str,
                 title: str, fname: str) -> None:
    mask = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[mask], y[mask], s=4, alpha=0.4, color="#4878cf")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / fname}")


def plot_hmm_boxplot(df: pd.DataFrame, fname: str) -> None:
    valid = df[["State_Label", "Log_Return"]].dropna()
    order = (valid.groupby("State_Label")["Log_Return"]
                  .mean().sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(7, 5))
    valid.boxplot(column="Log_Return", by="State_Label", ax=ax,
                  boxprops=dict(color="#4878cf"),
                  medianprops=dict(color="#c44e52", lw=2))
    ax.set_title("Log Return by HMM State", fontsize=11, fontweight="bold")
    ax.set_xlabel("HMM State Label")
    ax.set_ylabel("Log Return")
    plt.suptitle("")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / fname}")


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _corr_continuous(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    xs, ys = x[mask].values, y[mask].values
    if len(xs) < 5:
        return float("nan"), float("nan"), 0
    if method == "pearson":
        r, p = stats.pearsonr(xs, ys)
    else:
        r, p = stats.spearmanr(xs, ys)
    return float(r), float(p), int(len(xs))


def _kruskal(state: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = state.notna() & y.notna()
    groups = [y[mask & (state == lbl)].values
              for lbl in state[mask].unique()]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan"), float("nan")
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("EDA -- CPO Pipeline Full Window")
    print(f"  Window : {FULL_START_TS.date()} to {FULL_END_TS.date()}")
    print(f"  Train/ : {FULL_START_TS.date()} to {TRAIN_END_TS.date()}")
    print(f"  Test   : {TEST_START_TS.date()} to {FULL_END_TS.date()}")
    print("=" * 72)

    df = load_data()

    # ── Variables to analyse ────────────────────────────────────────────────
    variables = {
        "Close":           ("ratio", "CPO Close Price (MYR/ton)"),
        "Log_Return":      ("ratio", "Log Return y_t = ln(Close_t / Close_{t-1})"),
        "Sentiment_Score": ("interval", "FinBERT Title Sentiment Score"),
        "State":           ("ordinal", "HMM State (encoded integer 0-2)"),
    }

    summary_rows = []

    for col, (scale, label) in variables.items():
        if col not in df.columns:
            print(f"  SKIP {col} -- not in merged frame")
            continue

        print(f"\n{'-'*60}")
        print(f"  Variable: {label}  [scale={scale}]")
        print(f"{'-'*60}")

        series = df[col]

        # 1. Time-series plot
        fname_ts = f"ts_{col.lower()}.png"
        color_map = {"Close": "#1f77b4", "Log_Return": "#2ca02c",
                     "Sentiment_Score": "#ff7f0e", "State": "#9467bd"}
        plot_timeseries(df, col, label, fname_ts,
                        ylabel=label, color=color_map.get(col, "#1f77b4"))

        # 2. Descriptive stats
        desc = _desc(series)
        print(f"  count={desc['count']}  mean={desc['mean']:.4f}  "
              f"std={desc['std']:.4f}  skew={desc['skewness']:.3f}  "
              f"kurt={desc['kurtosis']:.3f}")

        # 3. Normality (skip categorical HMM State)
        if scale in ("ratio", "interval"):
            norm = normality_tests(series, col)
            print(f"  Normality: SW p={norm['SW_p']:.4g}  "
                  f"DA p={norm['DA_p']:.4g}  JB p={norm['JB_p']:.4g}  "
                  f"verdict={norm['normal_verdict']}")
            plot_dist_qq(series, label, f"dist_{col.lower()}.png")
            is_normal = norm["normal_verdict"] == "normal"
        else:
            norm = {"SW_p": float("nan"), "DA_p": float("nan"),
                    "JB_p": float("nan"), "normal_verdict": "N/A (ordinal)"}
            is_normal = False

        # 4. Measurement scale and test recommendation
        if scale in ("ratio", "interval"):
            rec_test = "Pearson" if is_normal else "Spearman"
            note = ("Normal distribution: Pearson correlation appropriate."
                    if is_normal
                    else "Non-normal distribution: Spearman correlation recommended.")
        else:
            rec_test = "Spearman + Kruskal-Wallis"
            note = ("Ordinal/categorical scale: Pearson is not appropriate. "
                    "Use Spearman (if states treated as ordered) and "
                    "Kruskal-Wallis for group-difference test.")
        print(f"  Scale: {scale}  Recommended test: {rec_test}")
        print(f"  Note : {note}")

        # 5. Relationship to target Log_Return
        y_col = df["Log_Return"]
        if col == "Log_Return":
            coef, pval, n_used, test_used = float("nan"), float("nan"), 0, "self"
        elif col == "State":
            # Spearman on ordered state score + Kruskal-Wallis
            coef, pval, n_used = _corr_continuous(series, y_col, "spearman")
            h_stat, kw_p       = _kruskal(df["State_Label"], y_col)
            print(f"  Spearman (State vs Log_Return): rho={coef:.4f} p={pval:.4g} n={n_used}")
            print(f"  Kruskal-Wallis (state groups vs Log_Return): H={h_stat:.3f} p={kw_p:.4g}")
            test_used = "Spearman + Kruskal-Wallis"
            plot_hmm_boxplot(df, "scatter_state_vs_logreturn.png")
        else:
            test_used = rec_test
            coef, pval, n_used = _corr_continuous(series, y_col, rec_test.lower())
            print(f"  {rec_test} ({col} vs Log_Return): r={coef:.4f} p={pval:.4g} n={n_used}")
            plot_scatter(series, y_col,
                         xlabel=label, ylabel="Log Return",
                         title=f"{label} vs Log Return",
                         fname=f"scatter_{col.lower()}_vs_logreturn.png")

        row = {
            "variable":       col,
            "label":          label,
            "scale":          scale,
            "count":          desc["count"],
            "mean":           round(desc["mean"], 6),
            "std":            round(desc["std"], 6),
            "skewness":       round(desc["skewness"], 4),
            "kurtosis":       round(desc["kurtosis"], 4),
            "SW_p":           norm["SW_p"],
            "DA_p":           norm["DA_p"],
            "JB_p":           norm["JB_p"],
            "normality":      norm["normal_verdict"],
            "recommended_test": rec_test,
            "contemp_coef":   round(coef, 4) if not np.isnan(coef) else float("nan"),
            "contemp_p":      round(pval, 6) if not np.isnan(pval) else float("nan"),
            "contemp_n":      n_used,
            "test_used":      test_used,
        }
        summary_rows.append(row)

    # ── Summary table ────────────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows)
    out_csv = _HERE / "eda_summary_table.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n{'='*72}")
    print("EDA SUMMARY TABLE")
    print(f"{'='*72}")
    display_cols = ["variable", "scale", "normality", "recommended_test",
                    "contemp_coef", "contemp_p", "contemp_n"]
    print(summary[display_cols].to_string(index=False))
    print(f"\nFull summary table -> {out_csv}")
    print(f"Figures            -> {FIG_DIR}/")
    print()
    print("Interpretation guide:")
    print("  Close        : ratio, likely non-normal (trending) -> Spearman for correlations")
    print("  Log_Return   : ratio, likely near-normal but leptokurtic -> check SW/JB p-values")
    print("  Sentiment    : interval, check normality to decide Pearson vs Spearman")
    print("  State        : ordinal/categorical -> Spearman (ordered) + Kruskal-Wallis")


if __name__ == "__main__":
    main()

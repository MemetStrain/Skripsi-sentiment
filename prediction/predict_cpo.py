"""CPO price-return prediction — configurable lagged features (XGBoost).

Edit the CONFIG block below to add / remove features, change lags, pick a
prediction horizon, and choose which models to run.

Default: HMM state score (lag 8) + FinBERT title sentiment (lag 44) +
         selected CPO price-derived features → predict next-day CPO return.

Available source columns (use exact names in FEATURES):
  From HMM        : state_score, log_return, volatility, rsi, macd, return_t
  From sentiment  : Sentiment_Score, Title_Positive_Prob,
                    Title_Negative_Prob, Title_Neutral_Prob, Article_Count
  From CPO vars   : Price_t-1, Price_t-2, Price_t-3,
                    Return_t-1, Return_t-2, Volume_t-1,
                    High_Low_Spread, Open_Close_Spread,
                    SMA_3, SMA_6, EMA_3, EMA_6,
                    MACD_Signal, Bollinger_Band_Width
  (Raw same-day OHLCV excluded to prevent data leakage)

Lag convention (matches lag-search scripts):
  lag=k means the feature value from k trading days BEFORE the prediction date
  is used.  E.g. lag=8 on HMM uses the state from 8 days ago.

Run:
    python predict_cpo.py

Outputs (written to prediction/output/):
    predictions.csv        — date, actual return, XGBoost return predictions
    direction_predictions.csv — date, actual direction, XGBoost direction predictions
    feature_importance.csv — XGBoost feature importances
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
except ImportError:
    sys.exit("ERROR: scikit-learn not installed.  Run: pip install scikit-learn")

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    sys.exit("ERROR: xgboost not installed.  Run: pip install xgboost")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — edit this section to customise the prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Features ─────────────────────────────────────────────────────────────────
# Each entry needs:
#   source : column name in the merged daily frame (see docstring above)
#   lag    : trading days back (0 = same-day value, k = k days ago)
#   name   : display label used in outputs and CSVs
FEATURES: list[dict] = [
    {"source": "state_score",         "lag": 8,  "name": "HMM_Lag8"},
    {"source": "Sentiment_Score",     "lag": 44, "name": "Sentiment_Lag44"},
    {"source": "Return_t-1",          "lag": 0,  "name": "Return_Lag1"},
    {"source": "Return_t-2",          "lag": 0,  "name": "Return_Lag2"},
    {"source": "High_Low_Spread",     "lag": 0,  "name": "HL_Spread"},
    {"source": "Open_Close_Spread",   "lag": 0,  "name": "OC_Spread"},
    {"source": "SMA_3",               "lag": 0,  "name": "SMA_3"},
    {"source": "SMA_6",               "lag": 0,  "name": "SMA_6"},
    {"source": "EMA_3",               "lag": 0,  "name": "EMA_3"},
    {"source": "EMA_6",               "lag": 0,  "name": "EMA_6"},
    {"source": "MACD_Signal",         "lag": 0,  "name": "MACD_Signal"},
    {"source": "Bollinger_Band_Width","lag": 0,  "name": "BB_Width"},
]

# ── Prediction target ─────────────────────────────────────────────────────────
# Cumulative return over the next FORWARD_HORIZON trading days.
# 1 = next-day return, 5 = next week, 22 = next month
FORWARD_HORIZON: int = 1

# ── Train / test split ────────────────────────────────────────────────────────
# Strictly chronological (no shuffling).
TRAIN_FRAC: float = 0.8

# ── Sentiment filter ──────────────────────────────────────────────────────────
# True  → only rows with actual news (Article_Count > 0) contribute to the
#         Sentiment feature column (NaN on non-news days → those rows are
#         dropped when assembling the feature matrix).
# False → forward-filled sentiment is used on all trading days.
SENTIMENT_NEWS_DAYS_ONLY: bool = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS  (script lives in prediction/, data lives one level up in project root)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT           = Path(__file__).resolve().parent.parent   # project root
HMM_STATES_CSV = ROOT / "markov/output/hmm_states_results_Daily.csv"
HMM_STATS_CSV  = ROOT / "markov/output/hmm_states_stats_Daily.csv"
SENTIMENT_CSV  = ROOT / "news/output/sentiment_aggregate_Daily_title.csv"
CPO_VARS_CSV   = ROOT / "cpo/output/cpo_variables_Daily.csv"
OUT_DIR        = Path(__file__).resolve().parent / "output"

# Raw same-day OHLCV columns excluded from CPO vars to prevent data leakage.
# RSI and MACD are also excluded because they duplicate columns already loaded
# from the HMM states frame (as 'rsi' and 'macd').
_CPO_VARS_DROP = {
    "Open", "High", "Low", "Close", "Volume", "Change_Pct",
    "RSI", "MACD", "Log_Return",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_state_score_map(stats_path: Path) -> dict[int, float]:
    """HMM state int → signed score in [-1, +1] (most bullish = +1)."""
    stats = pd.read_csv(stats_path)
    n = len(stats)
    return {
        int(row["State"]): (
            0.0 if n == 1 else round(1.0 - 2.0 * rank / (n - 1), 4)
        )
        for rank, (_, row) in enumerate(stats.iterrows())
    }


def _load_hmm() -> pd.DataFrame:
    df = (
        pd.read_csv(HMM_STATES_CSV, parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["state_score"] = df["State"].map(_build_state_score_map(HMM_STATS_CSV))
    return df.rename(columns={
        "Close":      "close",
        "Log_Return": "log_return",
        "Volatility": "volatility",
        "RSI":        "rsi",
        "MACD":       "macd",
    })


def _load_sentiment(news_days_only: bool) -> pd.DataFrame:
    df = (
        pd.read_csv(SENTIMENT_CSV, parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if news_days_only:
        df = df[df["Article_Count"] > 0].copy()
    keep = [
        "Date", "Article_Count", "Sentiment_Score",
        "Title_Positive_Prob", "Title_Negative_Prob", "Title_Neutral_Prob",
    ]
    return df[[c for c in keep if c in df.columns]]


def _load_cpo_vars() -> pd.DataFrame:
    df = (
        pd.read_csv(CPO_VARS_CSV, parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    drop_cols = [c for c in df.columns if c in _CPO_VARS_DROP]
    return df.drop(columns=drop_cols)


def build_merged_frame() -> pd.DataFrame:
    """Load HMM + sentiment + CPO vars, merge on Date, compute daily return."""
    hmm      = _load_hmm()
    sent     = _load_sentiment(SENTIMENT_NEWS_DAYS_ONLY)
    cpo_vars = _load_cpo_vars()
    df = (
        hmm
        .merge(sent,     on="Date", how="left")
        .merge(cpo_vars, on="Date", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["return_t"] = df["close"].pct_change()
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE / TARGET CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_feature_lags(df: pd.DataFrame, features: list[dict]) -> pd.DataFrame:
    for feat in features:
        src, lag, name = feat["source"], feat["lag"], feat["name"]
        if src not in df.columns:
            raise KeyError(
                f"Feature source '{src}' not found in merged frame.\n"
                f"Available columns: {sorted(df.columns.tolist())}"
            )
        # shift(lag) pulls the value from `lag` rows earlier in time
        df[name] = df[src].shift(lag)
    return df


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df["target"] = df["close"].shift(-horizon) / df["close"] - 1.0
    # direction: 1 = up, 0 = down, NaN = flat (excluded from classification)
    df["target_dir"] = np.where(
        df["target"] > 0, 1.0,
        np.where(df["target"] < 0, 0.0, np.nan)
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS & PRINTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def _reg_metrics(model_key: str, split: str,
                 y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pr, pp = pearsonr(y_true, y_pred)
    return dict(
        model     = model_key,
        split     = split,
        n         = len(y_true),
        r2        = round(float(r2_score(y_true, y_pred)), 4),
        rmse      = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        mae       = round(float(mean_absolute_error(y_true, y_pred)), 6),
        pearson_r = round(float(pr), 4),
        pearson_p = round(float(pp), 4),
        dir_acc   = _directional_accuracy(y_true, y_pred),
    )


def _print_reg_table(rows: list[dict]) -> None:
    hdr = f"  {'Model':<12}  {'Split':<5}  {'n':>5}  {'R²':>7}  {'RMSE':>9}  {'MAE':>9}  {'Pearson r':>10}  {'Dir.Acc':>8}"
    sep = f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*8}"
    print(hdr)
    print(sep)
    for r in rows:
        da = "n/a" if (isinstance(r["dir_acc"], float) and np.isnan(r["dir_acc"])) \
             else f"{r['dir_acc']:.2%}"
        print(
            f"  {r['model']:<12}  {r['split']:<5}  {r['n']:>5}  "
            f"{r['r2']:>7.4f}  {r['rmse']:>9.6f}  {r['mae']:>9.6f}  "
            f"{r['pearson_r']:>10.4f}  {da:>8}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    for p in (HMM_STATES_CSV, HMM_STATS_CSV, SENTIMENT_CSV):
        if not p.exists():
            raise SystemExit(f"ERROR: not found: {p}")

    print("=" * 72)
    print("CPO Price-Return Prediction — Configurable Lagged Features")
    print("=" * 72)

    # ── Build frame ─────────────────────────────────────────────────────────
    df = build_merged_frame()
    df = apply_feature_lags(df, FEATURES)
    df = add_targets(df, FORWARD_HORIZON)

    feat_names = [f["name"] for f in FEATURES]

    # Drop rows missing any feature or regression target
    model_df = df.dropna(subset=feat_names + ["target"]).copy().reset_index(drop=True)

    # Classification subset: additionally exclude zero-return days
    cls_df = model_df.dropna(subset=["target_dir"]).copy().reset_index(drop=True)

    print(f"\nFeatures              : {feat_names}")
    for feat in FEATURES:
        print(f"  {feat['name']:<24}  source={feat['source']!r}  lag={feat['lag']}")
    print(f"\nForward horizon       : {FORWARD_HORIZON} trading day(s)")
    print(f"Date range            : {model_df['Date'].min().date()} -> {model_df['Date'].max().date()}")
    print(f"Rows (regression)     : {len(model_df)}")
    print(f"Rows (classification) : {len(cls_df)}")
    print(f"Train fraction        : {TRAIN_FRAC:.0%}")

    # ── Univariate correlations ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Univariate feature–target correlations")
    print(f"{'='*72}")
    print(f"  {'Feature':<24}  {'lag':>4}  {'Pearson r':>10}  {'p-value':>8}  {'Dir.Acc':>8}")
    print(f"  {'-'*24}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}")
    for feat in FEATURES:
        name = feat["name"]
        valid = model_df[[name, "target"]].dropna()
        if len(valid) >= 5:
            r, p = pearsonr(valid[name], valid["target"])
            da   = _directional_accuracy(valid[name].to_numpy(), valid["target"].to_numpy())
            da_s = f"{da:.2%}" if not np.isnan(da) else "n/a"
            print(f"  {name:<24}  {feat['lag']:>4}  {r:>+10.4f}  {p:>8.4f}  {da_s:>8}")
        else:
            print(f"  {name:<24}  {feat['lag']:>4}  {'n/a':>10}  {'n/a':>8}  {'n/a':>8}")

    # ── Chronological train / test split ─────────────────────────────────────
    n_train = int(len(model_df) * TRAIN_FRAC)
    train   = model_df.iloc[:n_train]
    test    = model_df.iloc[n_train:]

    n_train_c = int(len(cls_df) * TRAIN_FRAC)
    train_c   = cls_df.iloc[:n_train_c]
    test_c    = cls_df.iloc[n_train_c:]

    X_train = train[feat_names].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    X_test  = test[feat_names].to_numpy(dtype=float)
    y_test  = test["target"].to_numpy(dtype=float)

    X_train_c = train_c[feat_names].to_numpy(dtype=float)
    y_train_c = train_c["target_dir"].to_numpy(dtype=int)
    X_test_c  = test_c[feat_names].to_numpy(dtype=float)
    y_test_c  = test_c["target_dir"].to_numpy(dtype=int)

    print(f"\nTrain: {n_train} rows  "
          f"({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"Test : {len(test)} rows  "
          f"({test['Date'].min().date()} → {test['Date'].max().date()})")

    # ── XGBoost regressor (predict return magnitude) ─────────────────────────
    reg = XGBRegressor()
    reg.fit(X_train, y_train)

    yhat_tr = reg.predict(X_train)
    yhat_te = reg.predict(X_test)

    reg_metrics_rows = [
        _reg_metrics("xgb", "train", y_train, yhat_tr),
        _reg_metrics("xgb", "test",  y_test,  yhat_te),
    ]

    print(f"\n{'='*72}")
    print("  XGBoost Regressor — return prediction metrics")
    print(f"{'='*72}")
    _print_reg_table([r for r in reg_metrics_rows if r["split"] == "test"])

    # ── XGBoost classifier (predict direction: 1=up / 0=down) ────────────────
    clf = XGBClassifier()
    clf.fit(X_train_c, y_train_c)

    acc_tr = accuracy_score(y_train_c, clf.predict(X_train_c))
    acc_te = accuracy_score(y_test_c,  clf.predict(X_test_c))
    baseline = max(y_test_c.mean(), 1 - y_test_c.mean())

    print(f"\n{'='*72}")
    print("  XGBoost Classifier — direction prediction metrics")
    print(f"{'='*72}")
    print(f"  Baseline (majority class): {baseline:.2%}")
    print(f"\n  {'Split':<5}  {'n':>5}  {'Accuracy':>9}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*9}")
    print(f"  {'train':<5}  {len(y_train_c):>5}  {acc_tr:>9.2%}")
    print(f"  {'test':<5}  {len(y_test_c):>5}  {acc_te:>9.2%}")

    # ── Feature importances ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  XGBoost feature importances (gain)")
    print(f"{'='*72}")
    print(f"\n  Regressor:")
    for fname, val in zip(feat_names, reg.feature_importances_):
        print(f"    {fname:<28}  {val:.6f}")
    print(f"\n  Classifier:")
    for fname, val in zip(feat_names, clf.feature_importances_):
        print(f"    {fname:<28}  {val:.6f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "Date":          model_df["Date"].to_numpy(),
        "actual_return": np.concatenate([y_train, y_test]),
        "pred_xgb":      np.concatenate([yhat_tr, yhat_te]),
    })
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    dir_pred_all = clf.predict(cls_df[feat_names].to_numpy(dtype=float))
    dir_df_out = pd.DataFrame({
        "Date":       cls_df["Date"].to_numpy(),
        "actual_dir": np.concatenate([y_train_c, y_test_c]),
        "pred_dir":   dir_pred_all,
    })
    dir_df_out.to_csv(OUT_DIR / "direction_predictions.csv", index=False)

    imp_rows = (
        [{"model": "xgb_reg", "feature": n, "importance": round(float(v), 6)}
         for n, v in zip(feat_names, reg.feature_importances_)]
        + [{"model": "xgb_clf", "feature": n, "importance": round(float(v), 6)}
           for n, v in zip(feat_names, clf.feature_importances_)]
    )
    pd.DataFrame(imp_rows).to_csv(OUT_DIR / "feature_importance.csv", index=False)

    print(f"\n{'='*72}")
    print(f"  Outputs saved to: {OUT_DIR}")
    print(f"{'='*72}")
    print(f"  predictions.csv           — date, actual return, xgb return prediction")
    print(f"  direction_predictions.csv — date, actual direction, xgb direction prediction")
    print(f"  feature_importance.csv    — XGBoost gain-based importances")
    print(f"\nNotes:")
    print(f"  lag k    : feature value from k trading days before the prediction date")
    print(f"  target   : {FORWARD_HORIZON}-day forward return = close[t+{FORWARD_HORIZON}] / close[t] - 1")
    print(f"  dir_acc  : share of days where sign(pred) == sign(actual)")
    print(f"  R² ~ 0   : expected for daily financial returns — use dir_acc instead")


if __name__ == "__main__":
    main()

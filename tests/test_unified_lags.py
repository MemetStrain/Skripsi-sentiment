"""Tests for the unified lag refactor (Formula A)."""
import numpy as np
import pandas as pd
import pytest

from prediction.master_features import (
    LAG_FEATURES, lag_shift, get_ablation_bases,
)
from prediction.feature_engineering import build_unified_features


# --- lag_shift unit tests ----------------------------------------------------

@pytest.mark.parametrize("k,h,expected", [
    (1, 1, 1), (1, 7, 7), (2, 7, 8), (3, 7, 9),
    (30, 1, 30), (30, 7, 36),
    (11, 7, 17),
])
def test_lag_shift_formula(k: int, h: int, expected: int) -> None:
    assert lag_shift(k, h) == expected


@pytest.mark.parametrize("k,h", [(0, 1), (-1, 3), (1, 0), (1, -2)])
def test_lag_shift_rejects_invalid(k: int, h: int) -> None:
    with pytest.raises(ValueError):
        lag_shift(k, h)


# --- build_unified_features end-to-end test ----------------------------------

def _make_raw_df(n_days: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2025-10-01", periods=n_days, freq="D")
    df = pd.DataFrame({"Date": dates})
    # Synthetic price walk
    df["Close"] = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Volume"] = rng.integers(1000, 10000, n_days)
    df["Price"] = df["Close"]
    # Technicals (toy values; not real indicator math)
    for col in ["High_Low_Spread", "Open_Close_Spread", "SMA_3", "SMA_6",
                "EMA_3", "EMA_6", "RSI", "MACD", "MACD_Signal",
                "Bollinger_Band_Width", "Close_Anchor"]:
        df[col] = rng.normal(0, 1, n_days)
    # Sentiment + HMM features
    for col in ["Article_Count", "Positive_Prob", "Negative_Prob",
                "Neutral_Prob", "Confidence", "Sentiment_Score",
                "HMM_Volatility", "HMM_State", "HMM_Neutral",
                "HMM_Bearish", "HMM_Bullish"]:
        df[col] = rng.normal(0, 1, n_days)
    return df


def test_target_formula() -> None:
    df = _make_raw_df()
    out, _ = build_unified_features(df, horizon=7, ablation="C4_full")
    # Pick an interior row; verify Target = log(C[d]/C[d-7])
    sample = out.iloc[50]
    d = sample["Date"]
    close_d = df.loc[df["Date"] == d, "Close"].iloc[0]
    close_dh = df.loc[df["Date"] == d - pd.Timedelta(days=7), "Close"].iloc[0]
    expected = np.log(close_d / close_dh)
    assert sample["Target_LogReturn"] == pytest.approx(expected)


def test_lag_dates_h7() -> None:
    df = _make_raw_df()
    out, _ = build_unified_features(df, horizon=7, ablation="C4_full")
    sample = out.iloc[60]
    d = sample["Date"]

    # Sentiment_Score_lag1 should be at d - 7 (shift = 1 + 7 - 1 = 7)
    val_dh = df.loc[df["Date"] == d - pd.Timedelta(days=7),
                    "Sentiment_Score"].iloc[0]
    assert sample["Sentiment_Score_lag1"] == pytest.approx(val_dh)

    # Sentiment_Score_lag3 should be at d - 9 (shift = 3 + 7 - 1 = 9)
    val_dh3 = df.loc[df["Date"] == d - pd.Timedelta(days=9),
                     "Sentiment_Score"].iloc[0]
    assert sample["Sentiment_Score_lag3"] == pytest.approx(val_dh3)


def test_schema_identical_across_horizons() -> None:
    df = _make_raw_df()
    cols_by_h = {}
    for h in range(1, 8):
        out, feats = build_unified_features(df, horizon=h, ablation="C4_full")
        cols_by_h[h] = set(out.columns)
    # All horizons should have identical column set
    ref = cols_by_h[1]
    for h in range(2, 8):
        assert cols_by_h[h] == ref, f"Column mismatch at h={h}"


def test_ablation_membership() -> None:
    """C1 should not contain HMM or Sentiment columns; C4 should contain both."""
    df = _make_raw_df()
    out_c1, _ = build_unified_features(df, horizon=3, ablation="C1_cpo_only")
    out_c4, _ = build_unified_features(df, horizon=3, ablation="C4_full")

    sentiment_in_c1 = any("Sentiment" in c for c in out_c1.columns)
    hmm_in_c1 = any("HMM" in c for c in out_c1.columns)
    assert not sentiment_in_c1
    assert not hmm_in_c1

    sentiment_in_c4 = any("Sentiment" in c for c in out_c4.columns)
    hmm_in_c4 = any("HMM" in c for c in out_c4.columns)
    assert sentiment_in_c4
    assert hmm_in_c4


def test_calendar_source_date() -> None:
    """Calendar features must be computed from d - h, not d."""
    df = _make_raw_df()
    out, _ = build_unified_features(df, horizon=7, ablation="C4_full")
    sample = out.iloc[80]
    d = sample["Date"]
    expected_month = (d - pd.Timedelta(days=7)).month
    expected_sin = np.sin(2 * np.pi * expected_month / 12)
    assert sample["Month_Sin"] == pytest.approx(expected_sin)


# --- additional guard-rail tests for the reconciled design -------------------

def test_unknown_ablation_rejected() -> None:
    df = _make_raw_df()
    with pytest.raises(ValueError):
        build_unified_features(df, horizon=1, ablation="C9_bogus")
    with pytest.raises(ValueError):
        get_ablation_bases("C9_bogus")


def test_horizon_zero_rejected() -> None:
    df = _make_raw_df()
    with pytest.raises(ValueError):
        build_unified_features(df, horizon=0, ablation="C4_full")


def test_empty_post_dropna_rejected() -> None:
    """Too little history for the largest lag must raise, not return empty."""
    df = _make_raw_df(n_days=20)  # max lag 30 + h cannot be satisfied
    with pytest.raises(ValueError):
        build_unified_features(df, horizon=7, ablation="C4_full")


def test_lag1_resolves_to_forecast_origin() -> None:
    """Every base's _lag1 must anchor at the same date d - h (Formula A)."""
    df = _make_raw_df()
    h = 5
    out, _ = build_unified_features(df, horizon=h, ablation="C4_full")
    sample = out.iloc[70]
    d = sample["Date"]
    origin = d - pd.Timedelta(days=h)
    for base in ("Sentiment_Score", "HMM_State", "RSI", "Price"):
        if base not in LAG_FEATURES or 1 not in LAG_FEATURES[base]:
            continue
        expected = df.loc[df["Date"] == origin, base].iloc[0]
        assert sample[f"{base}_lag1"] == pytest.approx(expected), base

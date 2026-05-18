"""
Master feature schema for the unified lag refactor.

Convention (Formula A):
    Each row is indexed by target day `d`.
    A lag column `<base>_lag{k}` at row d, horizon h, takes the value at
    date `d - k - (h - 1)`, i.e., `df[base].shift(k + h - 1)`.

    There is no separate "snapshot" column. Legacy non-lag predictors
    (RSI, MACD, Sentiment_Score, ...) are represented as `_lag1`, which
    under Formula A resolves to the forecast origin `d - h`.

This module is the single source of truth for which features each ablation
config uses and which lag indices each base feature exposes. The offline
training path (`build_unified_features`) and the website inference path
(`engineer_all_features`) both read this schema, so the column set can never
diverge between training and inference.

Notes on deviations from the original spec:
    * `Close_Anchor` is intentionally NOT a base feature. It is never a column
      of any merged input frame (it is materialised only at dataset-save time
      as the inverse-transform anchor), and `Price_lag1` already equals the
      lagged absolute close, so a `Close_Anchor` lag would be a pure duplicate.
"""
from typing import Dict, List


# ------------------------------------------------------------------
# Lag indices per base feature.
# Each entry: base feature name -> list of lag indices k (k >= 1).
# A column `<name>_lag{k}` is emitted for each k.
# ------------------------------------------------------------------
LAG_FEATURES: Dict[str, List[int]] = {
    # CPO base (same-day values; `Price`/`Return` derived from `Close`).
    "Price":                [1, 2, 3],
    "Return":               [1, 2],
    "Volume":               [1],
    "Log_Return":           [1, 2, 3],

    # CPO technical indicators (legacy non-lag -> `_lag1`).
    "High_Low_Spread":      [1],
    "Open_Close_Spread":    [1],
    "SMA_3":                [1],
    "SMA_6":                [1],
    "EMA_3":                [1],
    "EMA_6":                [1],
    "RSI":                  [1],
    "MACD":                 [1],
    "MACD_Signal":          [1],
    "Bollinger_Band_Width": [1],

    # HMM features
    "HMM_Volatility":       [1],
    "HMM_State":            [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "HMM_Neutral":          [1],
    "HMM_Bearish":          [1],
    "HMM_Bullish":          [1],

    # Sentiment features
    "Article_Count":        [1],
    "Positive_Prob":        [1],
    "Negative_Prob":        [1],
    "Neutral_Prob":         [1],
    "Confidence":           [1],
    "Sentiment_Score":      [1, 3, 5, 10, 20, 30],

    # Interactions
    "Sentiment_x_Return":   [1],
    "Volatility_x_RSI":     [1],
}


# Calendar features. Anchored at the forecast origin: the row's date shifted
# back `h` trading rows via Date.shift(h) — the same row shift the lag
# features and the target use, so all three stay aligned on a real (gapped)
# trading calendar.
CALENDAR_FEATURE_NAMES: List[str] = [
    "Month_Sin", "Month_Cos",
    "DayOfWeek_Sin", "DayOfWeek_Cos",
    "WeekOfYear_Sin", "WeekOfYear_Cos",
]


# ------------------------------------------------------------------
# Per-ablation membership
# ------------------------------------------------------------------
CPO_BASE: List[str] = [
    "Price", "Return", "Volume", "Log_Return",
    "High_Low_Spread", "Open_Close_Spread",
    "SMA_3", "SMA_6", "EMA_3", "EMA_6",
    "RSI", "MACD", "MACD_Signal", "Bollinger_Band_Width",
]
HMM_GROUP: List[str] = [
    "HMM_Volatility", "HMM_State",
    "HMM_Neutral", "HMM_Bearish", "HMM_Bullish",
]
SENTIMENT_GROUP: List[str] = [
    "Article_Count",
    "Positive_Prob", "Negative_Prob", "Neutral_Prob",
    "Confidence", "Sentiment_Score",
]

VALID_ABLATIONS: List[str] = [
    "C1_cpo_only", "C2_cpo_hmm", "C3_cpo_sentiment", "C4_full",
]


def get_ablation_bases(ablation: str) -> List[str]:
    """
    Return the list of base feature names included for a given ablation.

    Args:
        ablation: One of 'C1_cpo_only', 'C2_cpo_hmm',
                  'C3_cpo_sentiment', 'C4_full'.

    Raises:
        ValueError: if `ablation` is not recognized.
    """
    bases: List[str] = list(CPO_BASE)
    if ablation == "C1_cpo_only":
        pass
    elif ablation == "C2_cpo_hmm":
        bases.extend(HMM_GROUP + ["Volatility_x_RSI"])
    elif ablation == "C3_cpo_sentiment":
        bases.extend(SENTIMENT_GROUP + ["Sentiment_x_Return"])
    elif ablation == "C4_full":
        bases.extend(HMM_GROUP + SENTIMENT_GROUP
                     + ["Volatility_x_RSI", "Sentiment_x_Return"])
    else:
        raise ValueError(
            f"Unknown ablation: {ablation!r}. "
            f"Expected one of: {', '.join(VALID_ABLATIONS)}."
        )
    return bases


def lag_shift(k: int, h: int) -> int:
    """
    Formula A: pandas .shift() argument for lag-k at horizon h.

    Args:
        k: lag index (>= 1).
        h: forecast horizon (>= 1).

    Returns:
        The number of rows to shift the source column by.

    Raises:
        ValueError: if k or h are out of range.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got k={k}")
    if h < 1:
        raise ValueError(f"h must be >= 1; got h={h}")
    return k + h - 1

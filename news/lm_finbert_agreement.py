"""Compare FinBERT and Loughran-McDonald (2011) sentiment on CPO news.

Reads ``mpob_news_with_sentiment.csv`` (which already contains FinBERT labels in
``Combined_Sentiment``), scores ``Content`` with the L-M lexicon via
``pysentiment2``, and reports inter-method agreement.

Outputs:
    - ``lm_finbert_comparison.csv``: per-row comparison frame.
    - ``confusion_matrix.png``: heatmap (rows = FinBERT, cols = L-M).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    # pysentiment2 is the maintained fork; install via `pip install pysentiment2`.
    import pysentiment2 as ps
except ImportError as exc:  # pragma: no cover - import guard
    print(
        "ERROR: pysentiment2 is not installed. Run `pip install pysentiment2`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from sklearn.metrics import cohen_kappa_score, confusion_matrix


INPUT_CSV = "mpob_news_with_sentiment_tone.csv"
OUTPUT_CSV = "lm_finbert_comparison.csv"
HEATMAP_PNG = "confusion_matrix.png"

# Restrict comparison to articles published on/after this year. Earlier MPOB
# articles use noticeably different phrasing/format, which drags L-M agreement
# down even when FinBERT handles them fine.
MIN_YEAR = 2015

# Dead-band around 0 for L-M polarity that we treat as neutral. Pysentiment2's
# LM polarity is bounded in [-1, 1]; widening this band moves weakly-signed
# articles into the neutral bucket so they can match FinBERT's neutrals.
NEUTRAL_TOLERANCE = 0.4

# Fixed label ordering keeps the confusion matrix and per-class metrics aligned
# regardless of which classes happen to appear in the data.
LABELS: Tuple[str, str, str] = ("negative", "neutral", "positive")


def load_news(path: Path) -> pd.DataFrame:
    """Load the FinBERT-scored news CSV and validate required columns."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: input file not found: {path}") from exc
    except pd.errors.ParserError as exc:
        raise SystemExit(f"ERROR: failed to parse CSV {path}: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot read {path}: {exc}") from exc

    required = {"Date", "Content", "Combined_Sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input CSV missing columns: {sorted(missing)}")
    return df


def polarity_to_label(polarity: float, tolerance: float = NEUTRAL_TOLERANCE) -> str:
    """Map a continuous L-M polarity score onto the FinBERT label space.

    Polarities with ``|polarity| <= tolerance`` are labelled neutral.
    """
    # NaN polarity (e.g., empty content) is treated as neutral so it can still
    # be compared row-wise against FinBERT.
    if polarity is None or (isinstance(polarity, float) and np.isnan(polarity)):
        return "neutral"
    if polarity > tolerance:
        return "positive"
    if polarity < -tolerance:
        return "negative"
    return "neutral"


def score_lm(texts: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Tokenize each text with the L-M lexicon and return (label, polarity)."""
    try:
        lm = ps.LM()
    except Exception as exc:  # pragma: no cover - depends on pysentiment2 internals
        raise SystemExit(f"ERROR: failed to initialize Loughran-McDonald lexicon: {exc}") from exc

    polarities: list[float] = []
    labels: list[str] = []
    for idx, text in texts.items():
        # Coerce non-strings (NaN, floats) to empty string to keep tokenizer happy.
        raw = "" if not isinstance(text, str) else text
        try:
            tokens = lm.tokenize(raw)
            score = lm.get_score(tokens)
            polarity = float(score.get("Polarity", 0.0))
        except Exception as exc:
            # Fail soft on a single row so one bad article does not abort the run.
            print(f"WARN: L-M scoring failed at row {idx}: {exc}", file=sys.stderr)
            polarity = 0.0
        polarities.append(polarity)
        labels.append(polarity_to_label(polarity))

    return pd.Series(labels, index=texts.index), pd.Series(polarities, index=texts.index)


def normalize_finbert(series: pd.Series) -> pd.Series:
    """Lowercase and strip whitespace so labels match the L-M label space."""
    return series.astype(str).str.strip().str.lower()


def per_class_agreement(
    finbert: pd.Series, lm: pd.Series, labels: Tuple[str, ...]
) -> dict[str, float]:
    """For each FinBERT class, share of rows where L-M agrees."""
    rates: dict[str, float] = {}
    for label in labels:
        mask = finbert == label
        if mask.sum() == 0:
            rates[label] = float("nan")  # no FinBERT rows of this class
        else:
            rates[label] = float((lm[mask] == label).mean())
    return rates


def save_heatmap(cm: np.ndarray, labels: Tuple[str, ...], path: Path) -> None:
    """Render the confusion matrix as a labeled heatmap."""
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("LM")
        ax.set_ylabel("FinBERT")
        ax.set_title("FinBERT vs Loughran-McDonald confusion matrix")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot write heatmap to {path}: {exc}") from exc


def main() -> None:
    here = Path(__file__).resolve().parent
    input_path = here / INPUT_CSV
    output_path = here / OUTPUT_CSV
    heatmap_path = here / HEATMAP_PNG

    df = load_news(input_path)

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    before = len(df)
    df = df.loc[parsed_dates.dt.year >= MIN_YEAR].reset_index(drop=True)
    print(f"Filtered to Date.year >= {MIN_YEAR}: kept {len(df)} of {before} rows.")

    # Score with L-M and normalize FinBERT side-by-side.
    lm_label, lm_polarity = score_lm(df["Content"])
    df["LM_Sentiment"] = lm_label
    df["LM_Polarity"] = lm_polarity
    df["Combined_Sentiment"] = normalize_finbert(df["Combined_Sentiment"])

    finbert = df["Combined_Sentiment"]
    lm = df["LM_Sentiment"]

    overall_agreement = float((finbert == lm).mean())
    kappa = float(cohen_kappa_score(finbert, lm, labels=list(LABELS)))
    cm = confusion_matrix(finbert, lm, labels=list(LABELS))
    class_rates = per_class_agreement(finbert, lm, LABELS)

    # Persist the row-level comparison frame.
    out_df = df[["Date", "Content", "Combined_Sentiment", "LM_Sentiment", "LM_Polarity"]]
    try:
        out_df.to_csv(output_path, index=False)
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot write {output_path}: {exc}") from exc

    save_heatmap(cm, LABELS, heatmap_path)

    # Summary report.
    print("=" * 60)
    print("FinBERT vs Loughran-McDonald (2011) agreement report")
    print("=" * 60)
    print(f"Rows compared           : {len(df)}")
    print(f"Overall agreement rate  : {overall_agreement:.2%}")
    print(f"Cohen's Kappa           : {kappa:.4f}")
    print()
    print("Confusion matrix (rows = FinBERT, cols = LM):")
    cm_df = pd.DataFrame(cm, index=list(LABELS), columns=list(LABELS))
    cm_df.index.name = "FinBERT \\ LM"
    print(cm_df.to_string())
    print()
    print("Per-class agreement (FinBERT class -> share matched by LM):")
    for label in LABELS:
        rate = class_rates[label]
        rate_str = "n/a" if np.isnan(rate) else f"{rate:.2%}"
        print(f"  {label:<8}: {rate_str}")
    print()
    print(f"Wrote per-row comparison -> {output_path}")
    print(f"Wrote heatmap            -> {heatmap_path}")


if __name__ == "__main__":
    main()

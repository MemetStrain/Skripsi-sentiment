"""
manual_sample_eval.py — Phase 2c of the FinBERT validation suite.

Two sub-tasks:
  2c-i  generate_sample(): emit a 50-article CSV stratified by FinBERT
        prediction with an empty manual_label column for the user to fill in.
  2c-ii evaluate_labels(): if the user has saved manual_sample_LABELED.csv,
        compute confusion matrix + classification report between FinBERT
        predictions and the manual labels.

Reads existing artifact `news/mpob_news_with_sentiment.csv` so we DO NOT
re-run FinBERT on the full corpus (the Combined_Sentiment column was
written by news/finbert_sentiment_analysis_flexible.py during the
initial FinBERT pass).

Usage:
    python news/validation/manual_sample_eval.py --generate
    python news/validation/manual_sample_eval.py --evaluate
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))

DEFAULT_ARTICLE_CSV = os.path.join(
    _PROJECT_ROOT, "news", "mpob_news_with_sentiment_tone.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")

SAMPLE_FILENAME = "manual_sample_FOR_LABELING.csv"
LABELED_FILENAME = "manual_sample_LABELED.csv"
SAMPLE_TARGET_PER_CLASS = {"positive": 17, "negative": 17, "neutral": 16}
SAMPLE_TOTAL = sum(SAMPLE_TARGET_PER_CLASS.values())  # 50
RANDOM_STATE = 42

LABEL_ORDER = ["negative", "neutral", "positive"]


# =============================================================================
# Helpers
# =============================================================================

def _load_articles_with_sentiment(path: str) -> pd.DataFrame:
    """Load the existing scored MPOB CSV, normalising column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Article CSV not found: {path}\n"
            "Phase 2c expects news/mpob_news_with_sentiment.csv to exist "
            "(produced by the initial FinBERT pass)."
        )
    df = pd.read_csv(path)

    # Combined_Sentiment may be 'Positive'/'Negative'/'Neutral' (capitalised)
    if "Combined_Sentiment" not in df.columns:
        raise ValueError(
            "Combined_Sentiment column not found; cannot stratify the "
            "manual sample."
        )
    df["finbert_predicted_label"] = df["Combined_Sentiment"].astype(str).str.lower()

    # Confidence: max of the three combined probabilities (or fall back)
    prob_cols = [
        "Combined_Positive_Prob", "Combined_Negative_Prob", "Combined_Neutral_Prob",
    ]
    if all(c in df.columns for c in prob_cols):
        df["finbert_confidence"] = df[prob_cols].max(axis=1)
    elif "Combined_Confidence" in df.columns:
        df["finbert_confidence"] = df["Combined_Confidence"]
    else:
        df["finbert_confidence"] = np.nan

    # Snippet / content
    if "Content" in df.columns:
        df["content"] = df["Content"].fillna("").astype(str)
    elif "Snippet" in df.columns:
        df["content_snippet"] = df["Snippet"].fillna("").astype(str).str.slice(0, 500)
    else:
        df["content_snippet"] = ""

    if "Title" in df.columns:
        df["title"] = df["Title"].fillna("").astype(str)
    else:
        df["title"] = ""
    if "Date" in df.columns:
        df["_date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
        df["date"] = df["_date_dt"].dt.strftime("%Y-%m-%d")
    else:
        df["_date_dt"] = pd.NaT
        df["date"] = ""

    return df


def _confusion_and_report(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix

    rep = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0, output_dict=True,
    )
    rows: List[Dict] = []
    for cls in LABEL_ORDER:
        m = rep[cls]
        rows.append({
            "class": cls,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1-score"], 4),
            "support": int(m["support"]),
        })
    for avg in ("macro avg", "weighted avg"):
        m = rep[avg]
        rows.append({
            "class": avg.replace(" avg", "_avg"),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1-score"], 4),
            "support": int(m["support"]),
        })
    metrics_df = pd.DataFrame(rows)

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    cm_df.index.name = "true_label_manual"
    cm_df.columns.name = "predicted_label_finbert"

    macro_f1 = float(metrics_df.loc[metrics_df["class"] == "macro_avg", "f1"].iloc[0])
    accuracy = float((np.array(y_pred) == np.array(y_true)).mean())
    return {
        "metrics_df": metrics_df,
        "cm_df": cm_df,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def _save_confusion_plot(cm_df: pd.DataFrame, png_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_df.values, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        cbar=False, ax=ax,
    )
    ax.set_xlabel("FinBERT predicted_label")
    ax.set_ylabel("Manual true_label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Phase 2c-i — generate sample CSV for human labelling
# =============================================================================

def generate_sample(
    output_dir: str,
    article_csv: str = DEFAULT_ARTICLE_CSV,
    min_date: Optional[str] = None,
) -> str:
    """Stratified random sample of 50 articles, returns the CSV path.

    Parameters
    ----------
    min_date : Optional[str]
        Inclusive lower bound on Date (YYYY-MM-DD). Articles with missing
        or earlier dates are excluded. ``None`` keeps the original
        unfiltered behaviour.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Phase 2c-i: Generate manual labelling sample (n=50)")
    if min_date is not None:
        print(f"  date filter: Date >= {min_date}")
    print("=" * 65)

    df = _load_articles_with_sentiment(article_csv)
    df = df[df["finbert_predicted_label"].isin(LABEL_ORDER)].copy()

    if min_date is not None:
        cutoff = pd.to_datetime(min_date, errors="raise")
        before = len(df)
        df = df[df["_date_dt"].notna() & (df["_date_dt"] >= cutoff)].copy()
        print(f"  pool after date filter: {len(df)} (dropped {before - len(df)})")

    if len(df) < SAMPLE_TOTAL:
        raise ValueError(
            f"Only {len(df)} scored articles available — need at least "
            f"{SAMPLE_TOTAL} for stratified sampling."
        )

    rng = np.random.RandomState(RANDOM_STATE)
    sampled_parts: List[pd.DataFrame] = []
    for cls, target in SAMPLE_TARGET_PER_CLASS.items():
        pool = df[df["finbert_predicted_label"] == cls]
        if len(pool) == 0:
            print(f"  ! no articles with FinBERT label '{cls}' — skipping")
            continue
        take = min(target, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(df.loc[idx])
        print(f"  {cls:>8s}: pool={len(pool):>5}  taken={take}")

    sampled = pd.concat(sampled_parts, ignore_index=False)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    sampled.insert(0, "sample_id", np.arange(1, len(sampled) + 1))
    sampled["manual_label"] = ""

    out_cols = [
        "sample_id", "date", "title", "content",
        "finbert_predicted_label", "finbert_confidence", "manual_label",
    ]
    out = sampled[out_cols]
    csv_path = os.path.join(output_dir, SAMPLE_FILENAME)
    out.to_csv(csv_path, index=False)

    print(f"\n  → {csv_path}")
    print()
    print("=" * 65)
    print("Manual Sample Generated")
    print("=" * 65)
    print(f"File: {csv_path}")
    print()
    print("INSTRUCTIONS FOR MATTHEW:")
    print("  1. Open the CSV in Excel/Google Sheets")
    print("  2. Read each title + content")
    print("  3. Fill in the 'manual_label' column with: positive, negative, or neutral")
    print("  4. Use your own judgment based on financial sentiment toward CPO market")
    print(f"  5. Save the file as: {LABELED_FILENAME} (in same folder)")
    print("  6. Re-run: python news/validation/finbert_validation_suite.py --manual-only")
    print()
    print("ESTIMATED TIME: ~1 hour for 50 articles")
    print("=" * 65)
    return csv_path


# =============================================================================
# Phase 2c-ii — evaluate manual labels (only when LABELED.csv exists)
# =============================================================================

def evaluate_labels(output_dir: str) -> Optional[Dict[str, object]]:
    """Run the manual evaluation iff LABELED.csv exists. Returns metrics or None."""
    print("=" * 65)
    print("Phase 2c-ii: Evaluate manual labels")
    print("=" * 65)

    labeled_path = os.path.join(output_dir, LABELED_FILENAME)
    if not os.path.exists(labeled_path):
        print(f"  ! {LABELED_FILENAME} not found in {output_dir}")
        print(f"  Generate the sample first (--manual or --all), label it,")
        print(f"  then re-run this evaluation.")
        return None

    df = pd.read_csv(labeled_path)
    if "manual_label" not in df.columns or "finbert_predicted_label" not in df.columns:
        raise ValueError(
            f"{LABELED_FILENAME} must contain 'manual_label' and "
            "'finbert_predicted_label' columns."
        )

    df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower()
    df["finbert_predicted_label"] = (
        df["finbert_predicted_label"].astype(str).str.strip().str.lower()
    )

    # Validation
    n_total = len(df)
    invalid = df[~df["manual_label"].isin(LABEL_ORDER)]
    if not invalid.empty:
        rows = invalid["sample_id"].tolist() if "sample_id" in invalid.columns else invalid.index.tolist()
        raise ValueError(
            f"{len(invalid)} of {n_total} rows have an invalid or missing "
            f"manual_label (must be one of {LABEL_ORDER}). "
            f"Sample IDs needing attention: {rows[:10]}{'…' if len(rows) > 10 else ''}"
        )

    res = _confusion_and_report(
        df["manual_label"].tolist(),
        df["finbert_predicted_label"].tolist(),
    )
    macro_f1 = res["macro_f1"]
    accuracy = res["accuracy"]
    n = len(df)

    interpretation = (
        f"On {n} manually-labelled MPOB articles, FinBERT achieves "
        f"macro-F1 = {macro_f1:.4f} and agreement rate = {accuracy*100:.1f}%. "
        "Sample size is small (n=50); statistical significance is limited. "
        "Compare against the in-domain Phase 2a benchmark "
        "(finbert_phrasebank_results.csv) to assess domain transfer."
    )

    metrics_df = res["metrics_df"]
    metrics_df["interpretation"] = ""
    metrics_df.loc[metrics_df["class"] == "macro_avg", "interpretation"] = interpretation

    results_csv = os.path.join(output_dir, "finbert_manual_eval_results.csv")
    cm_csv = os.path.join(output_dir, "finbert_manual_confusion_matrix.csv")
    cm_png = os.path.join(output_dir, "finbert_manual_confusion_matrix.png")

    metrics_df.to_csv(results_csv, index=False)
    res["cm_df"].to_csv(cm_csv)
    _save_confusion_plot(
        res["cm_df"], cm_png,
        title=f"FinBERT vs manual labels (n={n})  macro-F1 = {macro_f1:.4f}",
    )

    # Honesty footnote
    footnote_path = os.path.join(output_dir, "manual_eval_limitations.txt")
    with open(footnote_path, "w", encoding="utf-8") as fh:
        fh.write(
            "Limitation note for thesis Bab 4 / Bab 5:\n"
            "Due to single-annotator constraints, intra-rater reliability "
            "was not measured. Future work should incorporate multi-annotator "
            "validation following established protocols (Cohen 1960; "
            "Artstein & Poesio 2008).\n"
        )

    print(f"  → {results_csv}")
    print(f"  → {cm_csv}")
    print(f"  → {cm_png}")
    print(f"  → {footnote_path}")
    print(f"\n  macro-F1 = {macro_f1:.4f}  agreement = {accuracy*100:.1f}%  n = {n}")
    print(f"\n{metrics_df.to_string(index=False)}")

    return {
        "macro_f1": macro_f1,
        "agreement_rate": accuracy,
        "n": n,
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2c: manual labelling sample + evaluation"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--articles", default=DEFAULT_ARTICLE_CSV)
    parser.add_argument("--min-date", default=None,
                        help="Inclusive lower bound on article Date (YYYY-MM-DD); "
                             "applies to --generate only.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true",
                   help="Generate manual_sample_FOR_LABELING.csv")
    g.add_argument("--evaluate", action="store_true",
                   help="Evaluate manual labels (requires manual_sample_LABELED.csv)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if args.generate:
        generate_sample(output_dir, article_csv=args.articles, min_date=args.min_date)
    else:
        evaluate_labels(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

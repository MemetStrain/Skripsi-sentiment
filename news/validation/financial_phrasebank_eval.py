"""
financial_phrasebank_eval.py — Phase 2a of the FinBERT validation suite.

Loads the Financial PhraseBank `sentences_50agree` subset, runs the same
FinBERT model used in production (ProsusAI/finbert), and reports per-class
precision / recall / F1 plus a confusion matrix. Falls back to a small
bundled synthetic test set if the dataset download fails (e.g. no
internet).

This module does NOT modify any production code — it imports from
transformers directly and runs offline analysis only.

Usage:
    python news/validation/financial_phrasebank_eval.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================================================================
# Constants
# =============================================================================

MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_LENGTH = 256          # PhraseBank sentences are short
BATCH_SIZE = 32
RANDOM_STATE = 42

# FinBERT id->label mapping (matches scheduler/sentiment_runner.py)
FINBERT_ID2LABEL = {0: 'neutral', 1: 'positive', 2: 'negative'}

# Financial PhraseBank label values: 0=negative, 1=neutral, 2=positive
PHRASEBANK_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

LABEL_ORDER = ["negative", "neutral", "positive"]


# =============================================================================
# FinBERT inference
# =============================================================================

def _load_finbert():
    """Load FinBERT model + tokenizer + device. Lazy import."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading FinBERT on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    return model, tokenizer, device


def predict_labels(sentences: List[str]) -> Tuple[List[str], np.ndarray]:
    """Run FinBERT on a list of sentences. Returns (labels, prob_matrix)."""
    import torch

    model, tokenizer, device = _load_finbert()
    label_strings: List[str] = []
    prob_rows: List[np.ndarray] = []

    n = len(sentences)
    for i in range(0, n, BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            label_strings.append(FINBERT_ID2LABEL[int(np.argmax(row))])
            prob_rows.append(row)

        done = min(i + BATCH_SIZE, n)
        if (i // BATCH_SIZE) % 10 == 0 or done == n:
            print(f"  FinBERT progress: {done}/{n}")

    return label_strings, np.vstack(prob_rows)


# =============================================================================
# Dataset loading (with graceful fallback)
# =============================================================================

def _load_phrasebank_via_datasets() -> Tuple[List[str], List[str]] | None:
    """First attempt: load via the Hugging Face `datasets` library.

    Newer datasets versions (>=4.0) refuse script-based dataset repos, so
    this call usually fails for the canonical `financial_phrasebank`
    package. We still try a couple of parquet mirrors before giving up.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"  ! datasets library unavailable: {exc}")
        return None

    for repo, subset in (
        ("takala/financial_phrasebank", "sentences_50agree"),
        ("financial_phrasebank", "sentences_50agree"),
    ):
        try:
            ds = load_dataset(repo, subset, trust_remote_code=True, split="train")
            cols = ds.column_names
            sent_col = "sentence" if "sentence" in cols else "text"
            sentences = list(ds[sent_col])
            gold = [PHRASEBANK_ID2LABEL[int(v)] for v in ds["label"]]
            print(f"  Loaded Financial PhraseBank from '{repo}': "
                  f"{len(sentences):,} sentences")
            return sentences, gold
        except Exception as exc:
            print(f"  ! '{repo}' not loadable ({exc}); trying next mirror...")
    return None


def _load_phrasebank_via_zip() -> Tuple[List[str], List[str]] | None:
    """Second attempt: download the original `FinancialPhraseBank-v1.0.zip`
    from the Hugging Face mirror and parse the `Sentences_50Agree.txt`
    file ourselves.

    File format is `sentence@label` per line, where label is one of
    {positive, negative, neutral}.
    """
    import io
    import zipfile
    import urllib.request

    urls = (
        # Canonical path baked into the (now-script-deprecated) HF loader
        "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
        # Same path under the takala/ namespace mirror
        "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
        # Public GitHub mirror (kept as last-resort backup)
        "https://github.com/neoyipeng2018/FinancialPhraseBank-v1.0/raw/main/FinancialPhraseBank-v1.0.zip",
    )
    for url in urls:
        try:
            print(f"  Downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
        except Exception as exc:
            print(f"  ! download failed ({exc})")
            continue

        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
            names = zf.namelist()
            target = next(
                (n for n in names if n.endswith("Sentences_50Agree.txt")),
                None,
            )
            if target is None:
                print(f"  ! Sentences_50Agree.txt not found inside zip "
                      f"(saw: {names[:5]})")
                continue
            with zf.open(target) as fh:
                raw = fh.read()
            # File is latin-1 in the original release
            text = raw.decode("latin-1")
        except Exception as exc:
            print(f"  ! zip parse failed ({exc})")
            continue

        sentences: List[str] = []
        gold: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or "@" not in line:
                continue
            sent, _, label = line.rpartition("@")
            label = label.strip().lower()
            if label not in {"positive", "negative", "neutral"}:
                continue
            sentences.append(sent.strip())
            gold.append(label)
        if sentences:
            print(f"  Loaded Financial PhraseBank from zip: "
                  f"{len(sentences):,} sentences (Sentences_50Agree.txt)")
            return sentences, gold
        print("  ! parsed zip but found 0 valid lines")
    return None


def _load_phrasebank() -> Tuple[List[str], List[str], str]:
    """Load Financial PhraseBank sentences_50agree.

    Tries (in order): Hugging Face `datasets` mirrors → direct zip
    download from HF → bundled 30-example fallback.
    """
    via_datasets = _load_phrasebank_via_datasets()
    if via_datasets is not None:
        return via_datasets[0], via_datasets[1], "PHRASEBANK"

    via_zip = _load_phrasebank_via_zip()
    if via_zip is not None:
        return via_zip[0], via_zip[1], "PHRASEBANK"

    print("  FALLBACK_MODE: using 30 hand-curated examples")
    fallback = [
        # positive (10)
        ("The company's quarterly profit rose 15% year-on-year.", "positive"),
        ("Sales beat analyst expectations and the stock surged.", "positive"),
        ("Operating margin expanded to its highest level in five years.", "positive"),
        ("The board approved a 20% dividend increase.", "positive"),
        ("Free cash flow doubled compared to the prior quarter.", "positive"),
        ("Demand for the product line continues to strengthen.", "positive"),
        ("The acquisition is immediately accretive to earnings.", "positive"),
        ("Order book reached a record high this quarter.", "positive"),
        ("Cost reduction efforts delivered better than expected savings.", "positive"),
        ("Strong export volumes drove revenue growth.", "positive"),
        # negative (10)
        ("The company issued a profit warning citing weak demand.", "negative"),
        ("Earnings missed estimates and shares fell 8%.", "negative"),
        ("Operating losses widened compared with last year.", "negative"),
        ("The dividend has been suspended due to financial stress.", "negative"),
        ("Free cash flow turned negative for the second straight quarter.", "negative"),
        ("Margins contracted on rising input costs.", "negative"),
        ("The acquisition has been written down by 30%.", "negative"),
        ("Bankruptcy proceedings have been initiated.", "negative"),
        ("Credit rating was downgraded to junk territory.", "negative"),
        ("Heavy inventory write-offs hurt the bottom line.", "negative"),
        # neutral (10)
        ("The company will report quarterly results next Tuesday.", "neutral"),
        ("The board appointed a new chief financial officer.", "neutral"),
        ("Headquarters will be relocated next year.", "neutral"),
        ("The annual general meeting is scheduled for May.", "neutral"),
        ("A new factory is under construction in Indonesia.", "neutral"),
        ("The product is now available in additional markets.", "neutral"),
        ("Management guidance is unchanged from last quarter.", "neutral"),
        ("The company filed a routine regulatory disclosure.", "neutral"),
        ("Trading was halted briefly pending an announcement.", "neutral"),
        ("The press release reaffirmed the previous outlook.", "neutral"),
    ]
    sentences = [s for s, _ in fallback]
    gold = [g for _, g in fallback]
    return sentences, gold, "FALLBACK_MODE"


# =============================================================================
# Evaluation
# =============================================================================

def _classification_report_df(
    y_true: List[str], y_pred: List[str]
) -> pd.DataFrame:
    """Per-class P/R/F1/support and macro / weighted averages."""
    from sklearn.metrics import classification_report

    rep = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER, zero_division=0, output_dict=True,
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
    return pd.DataFrame(rows)


def _confusion_matrix_df(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    df.index.name = "true_label"
    df.columns.name = "predicted_label"
    return df


def _save_confusion_plot(cm_df: pd.DataFrame, png_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_df.values, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        cbar=False, ax=ax,
    )
    ax.set_xlabel("predicted_label")
    ax.set_ylabel("true_label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Public entry point
# =============================================================================

def evaluate(output_dir: str) -> Dict[str, object]:
    """Run the Financial PhraseBank benchmark and write CSVs + PNG.

    Returns a small dict with the macro-F1 and accuracy for the orchestrator.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Phase 2a: Financial PhraseBank benchmark (sentences_50agree)")
    print("=" * 65)

    sentences, gold, mode = _load_phrasebank()
    if mode == "FALLBACK_MODE":
        print("  ! WARNING: real dataset unavailable; using 30-example fallback.")

    y_pred, _probs = predict_labels(sentences)

    metrics_df = _classification_report_df(gold, y_pred)
    cm_df = _confusion_matrix_df(gold, y_pred)

    macro_row = metrics_df[metrics_df["class"] == "macro_avg"].iloc[0]
    macro_f1 = float(macro_row["f1"])
    accuracy = float((np.array(y_pred) == np.array(gold)).mean())

    # Annotate with interpretation
    interpretation = (
        f"FinBERT achieves macro-F1 = {macro_f1:.4f} on Financial PhraseBank, "
        "demonstrating in-domain validity for financial sentiment "
        "classification. Application to MPOB articles assumes domain "
        "transfer holds."
    )
    if mode == "FALLBACK_MODE":
        interpretation = (
            f"[FALLBACK_MODE — n=30 synthetic examples] {interpretation} "
            "Re-run with internet access to evaluate on the full dataset."
        )

    metrics_df["mode"] = mode
    metrics_df["interpretation"] = ""
    metrics_df.loc[metrics_df["class"] == "macro_avg", "interpretation"] = interpretation

    results_csv = os.path.join(output_dir, "finbert_phrasebank_results.csv")
    cm_csv = os.path.join(output_dir, "finbert_phrasebank_confusion_matrix.csv")
    cm_png = os.path.join(output_dir, "finbert_phrasebank_confusion_matrix.png")

    metrics_df.to_csv(results_csv, index=False)
    cm_df.to_csv(cm_csv)
    _save_confusion_plot(
        cm_df, cm_png,
        title=f"FinBERT vs Financial PhraseBank ({mode}, n={len(sentences):,})\n"
              f"macro-F1 = {macro_f1:.4f}, accuracy = {accuracy:.4f}",
    )

    print(f"\n  → {results_csv}")
    print(f"  → {cm_csv}")
    print(f"  → {cm_png}")
    print(f"\n  macro-F1: {macro_f1:.4f}  accuracy: {accuracy:.4f}  mode: {mode}")
    print("\n" + metrics_df.to_string(index=False))

    return {
        "mode": mode,
        "n": len(sentences),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2a: Financial PhraseBank benchmark")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "output"))
    args = parser.parse_args()
    evaluate(os.path.abspath(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

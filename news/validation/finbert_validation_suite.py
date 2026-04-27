"""
finbert_validation_suite.py — Phase 2d orchestrator.

Composes the three FinBERT validation phases into a single CLI:
  --benchmark-only   : Phase 2a (Financial PhraseBank)
  --correlation-only : Phase 2b (Sentiment vs next-day return)
  --manual-only      : Phase 2c (Generate sample OR evaluate labels)
  --all              : Run 2a + 2b + 2c-i (sample generation), then 2c-ii
                       only if manual_sample_LABELED.csv exists.

Final output: news/validation/output/finbert_validation_summary.csv
(thesis Bab 4 — Tabel 4.Y).

Usage examples:
    python news/validation/finbert_validation_suite.py --all
    python news/validation/finbert_validation_suite.py --benchmark-only
    python news/validation/finbert_validation_suite.py --correlation-only
    python news/validation/finbert_validation_suite.py --manual-only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")
LIMITATION_NOTE = (
    "Single-annotator validation; no inter-rater Cohen's κ. Future work "
    "should incorporate multi-annotator validation per Cohen (1960) and "
    "Artstein & Poesio (2008)."
)


def _import_modules():
    """Local import so the orchestrator can be run directly via ``python news/validation/finbert_validation_suite.py``."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import financial_phrasebank_eval as phrasebank
    import sentiment_correlation_eval as correlation
    import manual_sample_eval as manual
    return phrasebank, correlation, manual


def _write_summary(
    output_dir: str,
    phrasebank_metrics: Dict[str, object] | None,
    correlation_metrics: Dict[str, float] | None,
    manual_metrics: Dict[str, object] | None,
) -> str:
    """Write the cross-phase summary CSV. Missing phases get 'PENDING'."""
    rows: List[Dict] = []

    if phrasebank_metrics is not None:
        rows.append({
            "metric": "phrasebank_macro_f1",
            "value": round(float(phrasebank_metrics["macro_f1"]), 4),
            "source": "Phase 2a",
            "interpretation": (
                f"External benchmark (Financial PhraseBank, "
                f"mode={phrasebank_metrics['mode']}, n={phrasebank_metrics['n']})"
            ),
        })
        rows.append({
            "metric": "phrasebank_accuracy",
            "value": round(float(phrasebank_metrics["accuracy"]), 4),
            "source": "Phase 2a",
            "interpretation": "Overall accuracy on benchmark",
        })
    else:
        rows.append({"metric": "phrasebank_macro_f1", "value": "PENDING",
                     "source": "Phase 2a", "interpretation": "Run --benchmark-only or --all"})
        rows.append({"metric": "phrasebank_accuracy", "value": "PENDING",
                     "source": "Phase 2a", "interpretation": "Run --benchmark-only or --all"})

    if correlation_metrics is not None:
        rows.append({
            "metric": "sentiment_pearson_corr",
            "value": round(correlation_metrics["pearson_corr"], 4),
            "source": "Phase 2b",
            "interpretation": "Daily sentiment vs next-day log return",
        })
        rows.append({
            "metric": "sentiment_pearson_pvalue",
            "value": round(correlation_metrics["pearson_p"], 4),
            "source": "Phase 2b",
            "interpretation": "Statistical significance (p)",
        })
        rows.append({
            "metric": "granger_lag1_pvalue",
            "value": _round_or_pending(correlation_metrics.get("granger_lag1_p")),
            "source": "Phase 2b",
            "interpretation": "Granger causality at lag 1",
        })
        rows.append({
            "metric": "directional_agreement_pct",
            "value": round(correlation_metrics["directional_agreement_pct"], 4),
            "source": "Phase 2b",
            "interpretation": "% days where sign(sentiment) == sign(next-day return)",
        })
    else:
        for m in ("sentiment_pearson_corr", "sentiment_pearson_pvalue",
                  "granger_lag1_pvalue", "directional_agreement_pct"):
            rows.append({"metric": m, "value": "PENDING", "source": "Phase 2b",
                         "interpretation": "Run --correlation-only or --all"})

    if manual_metrics is not None:
        rows.append({
            "metric": "manual_macro_f1",
            "value": round(float(manual_metrics["macro_f1"]), 4),
            "source": "Phase 2c",
            "interpretation": f"50-sample manual evaluation (n={manual_metrics['n']})",
        })
        rows.append({
            "metric": "manual_agreement_rate",
            "value": round(float(manual_metrics["agreement_rate"]), 4),
            "source": "Phase 2c",
            "interpretation": "% samples where FinBERT matches manual label",
        })
    else:
        rows.append({"metric": "manual_macro_f1", "value": "PENDING",
                     "source": "Phase 2c",
                     "interpretation": "Label manual_sample_FOR_LABELING.csv first"})
        rows.append({"metric": "manual_agreement_rate", "value": "PENDING",
                     "source": "Phase 2c",
                     "interpretation": "Label manual_sample_FOR_LABELING.csv first"})

    rows.append({"metric": "limitation_note", "value": "—",
                 "source": "—", "interpretation": LIMITATION_NOTE})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "finbert_validation_summary.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _round_or_pending(value) -> object:
    if value is None:
        return "PENDING"
    try:
        import numpy as np
        if isinstance(value, float) and np.isnan(value):
            return "PENDING"
    except Exception:
        pass
    return round(float(value), 4)


def _print_console_summary(
    output_dir: str,
    phrasebank_metrics: Dict[str, object] | None,
    correlation_metrics: Dict[str, float] | None,
    manual_metrics: Dict[str, object] | None,
) -> None:
    print()
    print("=" * 65)
    print("FinBERT Validation Summary")
    print("=" * 65)

    if phrasebank_metrics is not None:
        print(f"Financial PhraseBank macro-F1: "
              f"{phrasebank_metrics['macro_f1']:.4f}  "
              f"(accuracy {phrasebank_metrics['accuracy']:.4f}, "
              f"mode {phrasebank_metrics['mode']})")
    else:
        print("Financial PhraseBank macro-F1: PENDING")

    if correlation_metrics is not None:
        pr = correlation_metrics["pearson_corr"]
        prp = correlation_metrics["pearson_p"]
        g1 = correlation_metrics.get("granger_lag1_p", float("nan"))
        print(f"Sentiment-price Pearson r:     {pr:+.4f}  (p={prp:.4f})")
        print(f"Granger causality (lag 1):     p={g1:.4f}")
    else:
        print("Sentiment-price Pearson r:     PENDING")

    if manual_metrics is not None:
        print(f"Manual evaluation:             "
              f"macro-F1 {manual_metrics['macro_f1']:.4f}, "
              f"agreement {manual_metrics['agreement_rate']*100:.1f}%, "
              f"n={manual_metrics['n']}")
    else:
        print("Manual evaluation:             PENDING — see "
              "manual_sample_FOR_LABELING.csv")

    print(f"\nOutput files in: {output_dir}")
    print("=" * 65)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="FinBERT Validation Suite (Phase 2)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", dest="run_all", action="store_true",
                   help="Run Phase 2a + 2b + 2c (generate sample, then evaluate "
                        "if labelled file exists)")
    g.add_argument("--benchmark-only", action="store_true",
                   help="Run Phase 2a only (Financial PhraseBank)")
    g.add_argument("--correlation-only", action="store_true",
                   help="Run Phase 2b only (sentiment-price correlation)")
    g.add_argument("--manual-only", action="store_true",
                   help="Run Phase 2c (generate sample OR evaluate labels)")
    args = parser.parse_args()

    if not (args.run_all or args.benchmark_only or args.correlation_only or args.manual_only):
        args.run_all = True

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    phrasebank, correlation, manual = _import_modules()

    phrasebank_metrics = None
    correlation_metrics = None
    manual_metrics = None

    if args.run_all or args.benchmark_only:
        phrasebank_metrics = phrasebank.evaluate(output_dir)

    if args.run_all or args.correlation_only:
        correlation_metrics = correlation.evaluate(output_dir)

    if args.run_all or args.manual_only:
        labeled_path = os.path.join(output_dir, manual.LABELED_FILENAME)
        sample_path = os.path.join(output_dir, manual.SAMPLE_FILENAME)
        if args.manual_only and os.path.exists(labeled_path):
            manual_metrics = manual.evaluate_labels(output_dir)
        elif args.manual_only:
            manual.generate_sample(output_dir)
        else:
            # --all: always (re)generate the sample, then opportunistically evaluate
            if not os.path.exists(sample_path):
                manual.generate_sample(output_dir)
            else:
                print("=" * 65)
                print("Phase 2c-i: skipped (manual_sample_FOR_LABELING.csv already "
                      "exists; delete it to regenerate)")
                print("=" * 65)
            if os.path.exists(labeled_path):
                manual_metrics = manual.evaluate_labels(output_dir)

    summary_csv = _write_summary(
        output_dir, phrasebank_metrics, correlation_metrics, manual_metrics,
    )
    _print_console_summary(
        output_dir, phrasebank_metrics, correlation_metrics, manual_metrics,
    )
    print(f"\n→ summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

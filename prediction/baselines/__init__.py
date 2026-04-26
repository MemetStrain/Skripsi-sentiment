"""
Naive baseline integration for CPO price prediction pipeline.

Implements Hypothesis H4 control experiment: compare combined
sentiment + HMM + lagged price model against naive random walk.

Public API
----------
evaluate_all_naive_baselines : build naive-baseline rows for
    horizon_summary CSVs without re-running the parametric pipeline.
compare_best_vs_naive        : Diebold-Mariano pairwise test between the
    best parametric model and naive_rw across all horizons.
"""

from .naive_evaluator import evaluate_all_naive_baselines
from .dm_comparison import compare_best_vs_naive

__all__ = ["evaluate_all_naive_baselines", "compare_best_vs_naive"]

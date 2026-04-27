"""
hmm_validation_suite.py — Comprehensive HMM diagnostic report for thesis.

Produces validation outputs for Bab 4 (Results & Discussion):
  1. BIC comparison table (N=2, 3, 4)         → hmm_bic_comparison.csv + .png
  2. Multi-restart stability report (30 fits) → hmm_restart_stability.csv + .png
  3. Per-state emission diagnostics            → hmm_emission_diagnostics.csv + .png
  4. State labeling validation                 → hmm_state_labels_validation.csv
  5. Transition matrix audit                   → hmm_transition_audit.csv + sanity
  6. Comprehensive validation summary          → hmm_validation_summary.csv

The suite is purely additive: it imports utilities from markov/cpo_hmm_states.py
without modifying that module, and reads daily CPO data from the local CSV.

Usage:
    python markov/hmm_validation_suite.py --output-dir markov/validation_output/
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 which cannot render the unicode glyphs
# we use in progress prints (∈, →, ≈, ✓, ✗ etc). Force UTF-8 if available.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make markov/ importable when running as a script from project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cpo_hmm_states import (  # noqa: E402  reuse production HMM helpers, do not modify
    COVARIANCE_TYPE,
    N_ITER,
    RANDOM_SEED,
    TOL,
    _fit_single,
    compute_model_scores,
    count_free_params,
    fit_hmm_with_restarts,
    label_states,
    prepare_features,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_INPUT = os.path.join(
    os.path.dirname(_THIS_DIR), "cpo", "output", "cpo_variables_Daily.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(_THIS_DIR, "validation_output")

N_RESTARTS_BIC = 20            # restarts per N in BIC comparison
N_RESTARTS_STABILITY = 30      # restarts at N=3 for stability report
N_STATES_GRID = (2, 3, 4)      # BIC sweep range
N_STATES_THESIS = 3            # thesis design choice
NEUTRAL_TOLERANCE = 5e-4       # |mean log-return| threshold for "≈ 0"
PERSISTENCE_ABSORBING = 0.99   # diagonal threshold for absorbing-state warning
ROW_STOCHASTIC_TOL = 1e-3      # transition matrix row sum tolerance

PLOT_DPI = 300


# =============================================================================
# I/O helpers
# =============================================================================

def _load_cpo_variables(path: str) -> pd.DataFrame:
    """Load the daily CPO variables CSV. Mirrors `cpo_hmm_states.load_cpo_variables`
    but trims the print noise and operates without a frequency tag."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run cpo/preprocess_cpo_variables.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    required = ["Date", "Close", "Log_Return", "RSI", "MACD", "Bollinger_Band_Width"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    return df


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =============================================================================
# Phase 1b — BIC comparison (N ∈ {2, 3, 4})
# =============================================================================

def run_bic_comparison(
    X: np.ndarray, output_dir: str
) -> Tuple[pd.DataFrame, int, Dict[int, object]]:
    """Fit HMMs for N ∈ {2,3,4}, 20 restarts each, write CSV + PNG.

    Returns (bic_df, bic_optimal_N, fitted_models_by_N).
    """
    print("\n" + "=" * 65)
    print("Phase 1b: BIC comparison (N ∈ {2, 3, 4}, 20 restarts each)")
    print("=" * 65)

    rows: List[Dict] = []
    fitted: Dict[int, object] = {}

    for n in N_STATES_GRID:
        model, log_L = fit_hmm_with_restarts(
            X, n_states=n, cov_type=COVARIANCE_TYPE,
            n_iter=N_ITER, tol=TOL, n_restarts=N_RESTARTS_BIC,
            base_seed=RANDOM_SEED,
        )
        if model is None:
            print(f"  N={n}: all {N_RESTARTS_BIC} restarts failed")
            rows.append({
                "n_states": n, "log_likelihood": np.nan,
                "n_free_params": count_free_params(n, X.shape[1], COVARIANCE_TYPE),
                "AIC": np.nan, "BIC": np.nan,
                "converged": False,
            })
            continue

        scores = compute_model_scores(model, X, COVARIANCE_TYPE)
        converged = bool(getattr(model.monitor_, "converged", False))
        rows.append({
            "n_states": n,
            "log_likelihood": round(scores["log_L"], 4),
            "n_free_params": int(scores["k"]),
            "AIC": round(scores["AIC"], 4),
            "BIC": round(scores["BIC"], 4),
            "converged": converged,
        })
        fitted[n] = model
        print(
            f"  N={n}: log_L={scores['log_L']:11.2f}  k={scores['k']:>3}  "
            f"BIC={scores['BIC']:11.2f}  AIC={scores['AIC']:11.2f}  "
            f"converged={converged}"
        )

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["BIC"])
    bic_optimal_n = int(valid.loc[valid["BIC"].idxmin(), "n_states"]) if len(valid) else N_STATES_THESIS

    df["recommendation"] = ""
    df.loc[df["n_states"] == bic_optimal_n, "recommendation"] = "BIC_OPTIMAL"
    df["notes"] = ""
    if bic_optimal_n != N_STATES_THESIS:
        df.loc[df["n_states"] == N_STATES_THESIS, "notes"] = (
            f"Thesis design chose N={N_STATES_THESIS} for economic interpretability "
            f"(Bullish/Neutral/Bearish); BIC suggests N={bic_optimal_n}"
        )

    csv_path = os.path.join(output_dir, "hmm_bic_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    print(f"  BIC-optimal N = {bic_optimal_n} (thesis design = {N_STATES_THESIS})")

    # Plot: BIC bars (left axis) + log-likelihood line (right axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_bic = "#2E86AB"
    color_ll = "#A23B72"

    ax1.bar(
        df["n_states"], df["BIC"],
        width=0.5, color=color_bic, alpha=0.7, label="BIC", edgecolor="white",
    )
    ax1.set_xlabel("Number of states (N)", fontsize=11)
    ax1.set_ylabel("BIC (lower = better)", color=color_bic, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_bic)
    ax1.set_xticks(list(N_STATES_GRID))
    ax1.axvline(bic_optimal_n, color="red", linestyle="--", lw=1.2,
                label=f"BIC optimum (N={bic_optimal_n})")

    ax2 = ax1.twinx()
    ax2.plot(df["n_states"], df["log_likelihood"], color=color_ll,
             linestyle="--", marker="o", linewidth=1.5, label="log-likelihood")
    ax2.set_ylabel("log-likelihood (higher = better)", color=color_ll, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_ll)

    fig.suptitle(
        f"HMM model selection — BIC and log-likelihood by N states\n"
        f"(thesis design N={N_STATES_THESIS}; BIC optimum N={bic_optimal_n})",
        fontsize=12, fontweight="bold",
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    png_path = os.path.join(output_dir, "hmm_bic_comparison.png")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")

    return df, bic_optimal_n, fitted


# =============================================================================
# Phase 1c — Multi-restart stability (N=3, 30 restarts)
# =============================================================================

def run_restart_stability(
    X: np.ndarray, output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run 30 random restarts at N=3 and report log-likelihood distribution."""
    print("\n" + "=" * 65)
    print(f"Phase 1c: Restart stability (N={N_STATES_THESIS}, "
          f"{N_RESTARTS_STABILITY} restarts)")
    print("=" * 65)

    rows: List[Dict] = []
    for i in range(N_RESTARTS_STABILITY):
        seed = RANDOM_SEED + i
        model, log_L = _fit_single(
            X, n_states=N_STATES_THESIS, cov_type=COVARIANCE_TYPE,
            n_iter=N_ITER, tol=TOL, seed=seed,
        )
        if model is None:
            rows.append({
                "seed": seed, "log_likelihood": np.nan,
                "converged": False, "n_iter_to_converge": -1,
            })
            print(f"  seed={seed:>4}  FAILED")
            continue

        converged = bool(getattr(model.monitor_, "converged", False))
        n_iter = int(getattr(model.monitor_, "iter", -1))
        rows.append({
            "seed": seed,
            "log_likelihood": round(float(log_L), 4),
            "converged": converged,
            "n_iter_to_converge": n_iter,
        })
        if i < 5 or i == N_RESTARTS_STABILITY - 1 or (i + 1) % 10 == 0:
            print(f"  seed={seed:>4}  log_L={log_L:11.2f}  "
                  f"converged={converged}  iters={n_iter}")

    df = pd.DataFrame(rows)
    log_L_values = df["log_likelihood"].dropna().to_numpy()

    if len(log_L_values) == 0:
        summary = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                   "ci95_lo": np.nan, "ci95_hi": np.nan, "cv": np.nan}
    else:
        mean = float(np.mean(log_L_values))
        std = float(np.std(log_L_values, ddof=1)) if len(log_L_values) > 1 else 0.0
        sem = std / np.sqrt(len(log_L_values)) if len(log_L_values) > 1 else 0.0
        ci_half = 1.96 * sem
        summary = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(float(np.min(log_L_values)), 4),
            "max": round(float(np.max(log_L_values)), 4),
            "ci95_lo": round(mean - ci_half, 4),
            "ci95_hi": round(mean + ci_half, 4),
            "cv": round(std / abs(mean), 6) if abs(mean) > 0 else float("inf"),
        }

    summary_row = {
        "seed": "SUMMARY",
        "log_likelihood": summary["mean"],
        "converged": "",
        "n_iter_to_converge": "",
        "mean": summary["mean"],
        "std": summary["std"],
        "min": summary["min"],
        "max": summary["max"],
        "ci95_lo": summary["ci95_lo"],
        "ci95_hi": summary["ci95_hi"],
        "cv": summary["cv"],
    }
    df_out = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    csv_path = os.path.join(output_dir, "hmm_restart_stability.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    print(f"  log_L mean={summary['mean']:.2f}  std={summary['std']:.4f}  "
          f"CV={summary['cv']:.6f}  range=[{summary['min']:.2f}, {summary['max']:.2f}]")

    # Histogram with vertical line at best restart
    if len(log_L_values) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(log_L_values, bins=12, color="#2E86AB", alpha=0.75, edgecolor="white")
        ax.axvline(summary["max"], color="red", linestyle="--", lw=1.2,
                   label=f"best restart (log_L = {summary['max']:.2f})")
        ax.axvline(summary["mean"], color="black", linestyle=":", lw=1.0,
                   label=f"mean = {summary['mean']:.2f}")
        ax.set_xlabel("log-likelihood", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            f"HMM restart stability (N={N_STATES_THESIS}, "
            f"{N_RESTARTS_STABILITY} restarts)\n"
            f"mean = {summary['mean']:.2f} ± {summary['std']:.4f}  "
            f"(CV = {summary['cv']:.6f})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        png_path = os.path.join(output_dir, "hmm_restart_stability.png")
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {png_path}")

    return df_out, summary


# =============================================================================
# Phase 1d — Per-state emission diagnostics
# =============================================================================

def run_emission_diagnostics(
    X: np.ndarray, df_clean: pd.DataFrame, model, output_dir: str
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Decode states with Viterbi, compute per-state log-return diagnostics, save CSV+PNG."""
    print("\n" + "=" * 65)
    print(f"Phase 1d: Emission distribution diagnostics (N={N_STATES_THESIS})")
    print("=" * 65)

    states_raw = model.predict(X)

    # Sort states by mean log-return (descending = bullish first), then map old→new ids
    state_means = []
    for s in range(N_STATES_THESIS):
        mask = states_raw == s
        state_means.append(
            float(df_clean.loc[mask, "Log_Return"].mean()) if mask.any() else 0.0
        )
    sort_idx = np.argsort(state_means)[::-1]                   # descending order
    old_to_new = {int(old): int(new) for new, old in enumerate(sort_idx)}
    states_relabeled = np.array([old_to_new[int(s)] for s in states_raw])

    labels = label_states(N_STATES_THESIS)                     # ["Bullish","Neutral","Bearish"]
    transmat_raw = model.transmat_                             # in original ordering
    persistence_old = np.diag(transmat_raw)                    # P(stay) per old state

    n_total = len(df_clean)
    rows: List[Dict] = []
    log_returns_per_state: List[np.ndarray] = []

    for new_id in range(N_STATES_THESIS):
        mask = states_relabeled == new_id
        n_obs = int(mask.sum())
        log_returns = df_clean.loc[mask, "Log_Return"].to_numpy()
        log_returns_per_state.append(log_returns)

        if n_obs >= 3:
            shapiro_W, shapiro_p = stats.shapiro(log_returns)
            jb_stat, jb_p = stats.jarque_bera(log_returns)
            mean_lr = float(np.mean(log_returns))
            std_lr = float(np.std(log_returns, ddof=1))
            skew = float(stats.skew(log_returns))
            kurt = float(stats.kurtosis(log_returns))
        else:
            shapiro_W = shapiro_p = jb_stat = jb_p = np.nan
            mean_lr = std_lr = skew = kurt = np.nan

        # Map persistence: new_id → original old state
        original_old = int(sort_idx[new_id])
        persistence = float(persistence_old[original_old])

        rows.append({
            "state_id": new_id,
            "state_label": labels[new_id],
            "n_observations": n_obs,
            "frequency_pct": round(100.0 * n_obs / n_total, 4) if n_total else 0.0,
            "mean_logret": round(mean_lr, 6) if not np.isnan(mean_lr) else np.nan,
            "std_logret": round(std_lr, 6) if not np.isnan(std_lr) else np.nan,
            "skewness": round(skew, 4) if not np.isnan(skew) else np.nan,
            "kurtosis": round(kurt, 4) if not np.isnan(kurt) else np.nan,
            "shapiro_W": round(float(shapiro_W), 4) if not np.isnan(shapiro_W) else np.nan,
            "shapiro_pvalue": round(float(shapiro_p), 6) if not np.isnan(shapiro_p) else np.nan,
            "shapiro_normal": bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else False,
            "jarque_bera_stat": round(float(jb_stat), 4) if not np.isnan(jb_stat) else np.nan,
            "jb_pvalue": round(float(jb_p), 6) if not np.isnan(jb_p) else np.nan,
            "jb_normal": bool(jb_p > 0.05) if not np.isnan(jb_p) else False,
            "persistence": round(persistence, 6),
        })

    df = pd.DataFrame(rows)

    # Footnote row appended for thesis context (CSV-readable)
    footnote_row = {col: "" for col in df.columns}
    footnote_row["state_label"] = (
        "FOOTNOTE: Non-normality is expected for financial returns (fat tails); "
        "Gaussian HMM remains a reasonable approximation per literature "
        "(Hamilton 1989; Rabiner 1989). Future work could explore Student-t "
        "or mixture emission distributions."
    )
    df_out = pd.concat([df, pd.DataFrame([footnote_row])], ignore_index=True)
    csv_path = os.path.join(output_dir, "hmm_emission_diagnostics.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    for r in rows:
        print(f"  [{r['state_label']:>8s}] n={r['n_observations']:>4} "
              f"({r['frequency_pct']:>5.2f}%)  mean={r['mean_logret']:+.6f}  "
              f"std={r['std_logret']:.6f}  Shapiro p={r['shapiro_pvalue']:.4f}  "
              f"persistence={r['persistence']:.4f}")

    # Per-state plot: hist + Gaussian PDF + Q-Q. 3 cols × 2 rows (hist on top, QQ below)
    fig, axes = plt.subplots(2, N_STATES_THESIS, figsize=(5 * N_STATES_THESIS, 8))
    for new_id in range(N_STATES_THESIS):
        lr = log_returns_per_state[new_id]
        ax_top = axes[0, new_id]
        ax_bot = axes[1, new_id]
        if len(lr) < 3:
            ax_top.set_title(f"{labels[new_id]} — too few obs"); continue

        # Histogram + fitted Gaussian
        mu = float(np.mean(lr))
        sd = float(np.std(lr, ddof=1))
        ax_top.hist(lr, bins=40, density=True, color="#2E86AB", alpha=0.6,
                    edgecolor="white", label="empirical")
        x = np.linspace(lr.min(), lr.max(), 200)
        ax_top.plot(x, stats.norm.pdf(x, mu, sd), color="red", lw=1.4,
                    label=f"N({mu:.4f}, {sd:.4f})")
        sw_p = rows[new_id]["shapiro_pvalue"]
        ax_top.set_title(
            f"{labels[new_id]} — log-return distribution\n"
            f"n={rows[new_id]['n_observations']}  Shapiro p={sw_p:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax_top.set_xlabel("log-return"); ax_top.set_ylabel("density")
        ax_top.legend(fontsize=8); ax_top.grid(True, alpha=0.3)

        # Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(lr, dist="norm")
        ax_bot.scatter(osm, osr, s=6, alpha=0.6, color="#2E86AB")
        ax_bot.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.2)
        ax_bot.set_title(f"Q-Q plot vs Normal (r={r:.4f})", fontsize=11)
        ax_bot.set_xlabel("Theoretical quantiles"); ax_bot.set_ylabel("Sample quantiles")
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle(
        f"Per-state emission diagnostics (N={N_STATES_THESIS})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    png_path = os.path.join(output_dir, "hmm_emission_diagnostics.png")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")

    return df, states_relabeled, labels


# =============================================================================
# Phase 1e — State labeling validation
# =============================================================================

def run_state_label_validation(
    df_clean: pd.DataFrame, states_relabeled: np.ndarray,
    labels: List[str], output_dir: str
) -> pd.DataFrame:
    """Verify Bullish > 0, Bearish < 0, Neutral ≈ 0 in mean log-return."""
    print("\n" + "=" * 65)
    print("Phase 1e: State labeling validation")
    print("=" * 65)

    rows: List[Dict] = []
    for new_id, label in enumerate(labels):
        mask = states_relabeled == new_id
        mean_lr = float(df_clean.loc[mask, "Log_Return"].mean()) if mask.any() else 0.0

        if label == "Bullish":
            expected = "+"
            consistent = mean_lr > 0
        elif label == "Bearish":
            expected = "-"
            consistent = mean_lr < 0
        else:
            expected = "≈0"
            consistent = abs(mean_lr) < NEUTRAL_TOLERANCE

        notes = "" if consistent else (
            f"INCONSISTENT: {label} state has mean_logret={mean_lr:.6f}, "
            f"expected sign='{expected}'"
        )
        rows.append({
            "state_id": new_id,
            "assigned_label": label,
            "mean_logret": round(mean_lr, 6),
            "expected_sign": expected,
            "label_consistent": consistent,
            "notes": notes,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "hmm_state_labels_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    for r in rows:
        flag = "✓" if r["label_consistent"] else "✗"
        print(f"  {flag} {r['assigned_label']:>8s}: mean={r['mean_logret']:+.6f}  "
              f"expected '{r['expected_sign']}'")

    return df


# =============================================================================
# Phase 1f — Transition matrix audit
# =============================================================================

def run_transition_audit(
    model, labels: List[str], output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-order transition matrix to match relabeled states and emit two CSVs."""
    print("\n" + "=" * 65)
    print("Phase 1f: Transition matrix audit")
    print("=" * 65)

    transmat_raw = model.transmat_

    # Reconstruct old→new id mapping by sorting on emission means component 0
    means_logret_z = model.means_[:, 0]               # feature 0 = Log_Return_Z
    sort_idx = np.argsort(means_logret_z)[::-1]       # descending → bullish first
    n = transmat_raw.shape[0]
    transmat = np.zeros_like(transmat_raw)
    for i in range(n):
        for j in range(n):
            transmat[i, j] = transmat_raw[sort_idx[i], sort_idx[j]]

    # Long-format audit
    long_rows: List[Dict] = []
    for i in range(n):
        for j in range(n):
            p = float(transmat[i, j])
            note_bits: List[str] = []
            if p >= PERSISTENCE_ABSORBING:
                note_bits.append(f"absorbing-state warning (p>={PERSISTENCE_ABSORBING})")
            long_rows.append({
                "from_state": i,
                "to_state": j,
                "from_label": labels[i],
                "to_label": labels[j],
                "transition_prob": round(p, 6),
                "is_diagonal": i == j,
                "notes": "; ".join(note_bits),
            })

    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(output_dir, "hmm_transition_audit.csv")
    long_df.to_csv(long_path, index=False)
    print(f"  → {long_path}")

    # Sanity check (per-row stochasticity)
    sanity_rows: List[Dict] = []
    for i in range(n):
        row_sum = float(transmat[i].sum())
        diagonal = float(transmat[i, i])
        sanity_rows.append({
            "from_state": i,
            "from_label": labels[i],
            "row_sum": round(row_sum, 6),
            "is_stochastic": abs(row_sum - 1.0) < ROW_STOCHASTIC_TOL,
            "max_persistence": round(diagonal, 6),
            "has_absorbing": diagonal >= PERSISTENCE_ABSORBING,
        })
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_path = os.path.join(output_dir, "hmm_transition_sanity.csv")
    sanity_df.to_csv(sanity_path, index=False)
    print(f"  → {sanity_path}")

    print("  transition matrix (relabeled, rows = from):")
    for i in range(n):
        row_str = "  ".join(f"{transmat[i,j]:.4f}" for j in range(n))
        print(f"    {labels[i]:>8s}:  {row_str}")

    return long_df, sanity_df


# =============================================================================
# Phase 1g — Comprehensive summary
# =============================================================================

def build_summary(
    df_clean: pd.DataFrame,
    bic_df: pd.DataFrame, bic_optimal_n: int, chosen_model_scores: dict,
    stability_summary: Dict[str, float],
    state_label_df: pd.DataFrame,
    emission_df: pd.DataFrame,
    transition_sanity_df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """Aggregate validation findings into the thesis-ready single-row summary CSV."""
    print("\n" + "=" * 65)
    print("Phase 1g: Comprehensive validation summary")
    print("=" * 65)

    log_L_chosen = chosen_model_scores["log_L"]
    state_labels_consistent = bool(state_label_df["label_consistent"].all())
    emission_normality_passed = bool(emission_df["shapiro_normal"].all())
    transition_row_stochastic = bool(transition_sanity_df["is_stochastic"].all())
    absorbing_state_detected = bool(transition_sanity_df["has_absorbing"].any())
    n_obs = int(len(df_clean))
    date_min = str(df_clean["Date"].min().date())
    date_max = str(df_clean["Date"].max().date())

    rows = [
        {"metric": "n_states_chosen", "value": N_STATES_THESIS,
         "interpretation": "Bullish/Neutral/Bearish per thesis design"},
        {"metric": "n_states_bic_optimal", "value": bic_optimal_n,
         "interpretation": f"BIC suggests {bic_optimal_n} state(s)"},
        {"metric": "log_likelihood_chosen", "value": round(log_L_chosen, 4),
         "interpretation": f"Log-likelihood of chosen N={N_STATES_THESIS} model"},
        {"metric": "log_likelihood_mean_30restarts",
         "value": stability_summary["mean"],
         "interpretation": f"Mean across {N_RESTARTS_STABILITY} restarts"},
        {"metric": "log_likelihood_cv", "value": stability_summary["cv"],
         "interpretation": "Coefficient of variation (lower = more stable)"},
        {"metric": "state_labels_consistent", "value": state_labels_consistent,
         "interpretation": "All labels match economic expectation"},
        {"metric": "emission_normality_passed", "value": emission_normality_passed,
         "interpretation": "Are log-returns normally distributed per state? "
                           "(Expected: False for financial data)"},
        {"metric": "transition_row_stochastic", "value": transition_row_stochastic,
         "interpretation": "Sanity check: every transition row sums to 1"},
        {"metric": "absorbing_state_detected", "value": absorbing_state_detected,
         "interpretation": f"Warns of degenerate model (any diagonal "
                           f">= {PERSISTENCE_ABSORBING})"},
        {"metric": "data_observations", "value": n_obs,
         "interpretation": "Total daily observations used"},
        {"metric": "data_period", "value": f"{date_min} to {date_max}",
         "interpretation": "Date range"},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "hmm_validation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    return df


# =============================================================================
# Phase 1h — End-to-end orchestration
# =============================================================================

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HMM Validation Suite — comprehensive diagnostic outputs for thesis Bab 4"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Path to cpo_variables_Daily.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    output_dir = os.path.abspath(args.output_dir)
    _ensure_dir(output_dir)

    print("=" * 65)
    print("HMM Validation Suite — Thesis Diagnostic Report")
    print("=" * 65)
    print(f"Input  : {args.input}")
    print(f"Output : {output_dir}")

    # --- Load data and prepare features (mirrors production pipeline) ---
    df_raw = _load_cpo_variables(args.input)
    df_clean, feat_cols = prepare_features(df_raw, frequency="Daily")
    X = df_clean[feat_cols].to_numpy()

    # --- Phase 1b: BIC sweep ---
    bic_df, bic_optimal_n, fitted_models = run_bic_comparison(X, output_dir)

    # --- Pick the chosen N=3 model from the BIC sweep (already fit with 20 restarts) ---
    chosen_model = fitted_models.get(N_STATES_THESIS)
    if chosen_model is None:
        print("\n[ERROR] N=3 model failed all restarts; aborting downstream diagnostics.")
        return 1
    chosen_scores = compute_model_scores(chosen_model, X, COVARIANCE_TYPE)

    # --- Phase 1c: Stability ---
    _, stability_summary = run_restart_stability(X, output_dir)

    # --- Phase 1d: Emission diagnostics ---
    emission_df, states_relabeled, labels = run_emission_diagnostics(
        X, df_clean, chosen_model, output_dir,
    )

    # --- Phase 1e: Label validation ---
    label_df = run_state_label_validation(df_clean, states_relabeled, labels, output_dir)

    # --- Phase 1f: Transition audit ---
    _, sanity_df = run_transition_audit(chosen_model, labels, output_dir)

    # --- Phase 1g: Summary ---
    summary_df = build_summary(
        df_clean, bic_df, bic_optimal_n, chosen_scores,
        stability_summary, label_df, emission_df, sanity_df,
        output_dir,
    )

    # --- Phase 1h: Console report ---
    persistence_str = ", ".join(
        f"{lbl}={emission_df.iloc[i]['persistence']:.2f}"
        for i, lbl in enumerate(labels)
    )
    print()
    print("=" * 65)
    print("HMM Validation Summary")
    print("=" * 65)
    print(f"N states chosen           : {N_STATES_THESIS} (Bullish/Neutral/Bearish)")
    print(f"BIC-optimal N             : {bic_optimal_n}")
    print(f"Log-likelihood (chosen)   : {chosen_scores['log_L']:.2f}")
    print(f"Stability (CV across {N_RESTARTS_STABILITY:>2}) : "
          f"{stability_summary['cv']:.6f}")
    print(f"State labels consistent   : "
          f"{bool(label_df['label_consistent'].all())}")
    print(f"Persistence (diagonal)    : {persistence_str}")
    abs_flag = bool(sanity_df["has_absorbing"].any())
    print(f"Absorbing states          : {'DETECTED' if abs_flag else 'None detected'}")
    print(f"Output files              : {output_dir}/")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())

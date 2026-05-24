"""FASE 4 thesis-results runner.

Pipeline:
  1. BASE pass: for every (tag, horizon) in C1 x C2 x C3 x C4 x h in 1..7,
     train the ablation's BASE XGBoost model on pre-cutoff data, predict the
     post-cutoff test window. BASE = true library defaults (FASE 2.3), so the
     run is fast.
  2. Pick the winning ablation per horizon by min BASE RMSE on test.
  3. CSA pass: for each winning (tag, horizon), re-train with CSA-tuned
     hyperparameters. We only spend the CSA budget on cells we'll actually
     report in Tabel 4.7 (the user's scoped plan: ~7 x 4 minutes instead of
     ~28 x 4 minutes).
  4. Build deliverables under THESIS_RESULTS_<YYYYMMDD>/:
       - tabel_4.6_base_testing.csv               (BASE, 4 x 7 grid)
       - tabel_4.7_csa_winners_testing.csv        (CSA, 7 winning rows)
       - tabel_4.8_dm_base_pairwise.csv           (BASE DM HLN across C1-C4)
       - tabel_4.9_dm_csa_vs_base_winners.csv     (CSA vs BASE on each winner)
       - pt_directional_Daily_testing.csv          (PT 1992 per ablation x horizon)
       - h4_sufficiency_C4_vs_naive_rw.csv         (DM HLN of C4 BASE vs naive)
       - h2_feature_importance_C1_h1_top5.csv
       - h2_acf_close_lag1_5.csv (+ acf_pacf_log_return_Daily.png copied)
       - figure_4_3_mape_da_across_horizons.png    (Gambar 4.3)
       - winners.json (also written to prediction/winners.json)
       - SUMMARY.md (with Markdown tables ready to paste into Bab 4)

Run::

    PYTHONIOENCODING=utf-8 website/venv/Scripts/python.exe \
        prediction/fase4_thesis_runner.py --csa-population 50 --csa-iterations 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `prediction/` importable when invoked as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import horizon_forecast_C1_price_only as c1mod   # noqa: E402
import horizon_forecast_C2_price_hmm as c2mod    # noqa: E402
import horizon_forecast_C3_price_sentiment as c3mod  # noqa: E402
import horizon_forecast_C4_full as c4mod         # noqa: E402
from naive_baseline import diebold_mariano_test  # noqa: E402
from baselines import pt_directional as pt_module  # noqa: E402
from utils.forecast_utils import VAL_CUTOFF      # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _HERE.parent
HORIZONS = list(range(1, 8))

# --- CSA search budget -------------------------------------------------------
# 50/50 dipilih konsisten dgn train_winners_csa.py dan berada dlm rentang
# praktik metaheuristik utk tuning XGBoost berbasis CV (mis. populasi 50;
# lih. Askarzadeh, 2016 — pilihan parameter bersifat problem-dependent /
# No Free Lunch). Nilai TIDAK boleh berbeda antara FASE 4 dan klaim Bab 3.
DEFAULT_CSA_POPULATION = 50
DEFAULT_CSA_ITERATIONS = 50

# Tag -> (ablation human label, module exposing load + run helpers)
TAG_MODULES: Dict[str, Tuple[str, object]] = {
    "cpo_only":      ("C1_cpo_only", c1mod),
    "cpo_hmm":       ("C2_cpo_hmm", c2mod),
    "cpo_sentiment": ("C3_cpo_sentiment", c3mod),
    "full":          ("C4_full", c4mod),
}
TAG_TO_CONFIG = {"cpo_only": "C1", "cpo_hmm": "C2",
                 "cpo_sentiment": "C3", "full": "C4"}


# ---------------------------------------------------------------------------
# Step 1 + 3: pipeline runs (delegate to the C-script run_single_horizon)
# ---------------------------------------------------------------------------

def run_pass(csa_config: dict, only: List[Tuple[str, int]] | None = None) -> None:
    """If `only` is None: run all (tag, horizon) cells.
    Otherwise: run only the listed pairs. csa_config is forwarded as-is.
    """
    for tag, (_label, mod) in TAG_MODULES.items():
        # Filter horizons for this ablation.
        if only is not None:
            horizons_to_run = [h for (t, h) in only if t == tag]
            if not horizons_to_run:
                continue
        else:
            horizons_to_run = list(HORIZONS)

        print(f"\n{'#'*70}")
        print(f"  PASS: {tag}  horizons={horizons_to_run}  "
              f"CSA={'ON' if csa_config.get('enabled') else 'OFF'}")
        print(f"{'#'*70}")
        merged_df = mod.load_and_merge_data("Daily")
        output_dir = str(_HERE / "output_horizons" / mod.SCRIPT_TAG)
        os.makedirs(output_dir, exist_ok=True)

        for h in horizons_to_run:
            mod.run_single_horizon("Daily", h, merged_df, output_dir, csa_config)


# ---------------------------------------------------------------------------
# Step 2 + 4: aggregation, DM tests, plots, summary
# ---------------------------------------------------------------------------

def _results_csv(tag: str, h: int) -> Path:
    return (_HERE / "output_horizons" / tag / "Daily" / f"horizon_{h}"
            / f"testing_results_Daily_h{h}.csv")


def _predictions_csv(tag: str, h: int) -> Path:
    return (_HERE / "output_horizons" / tag / "Daily" / f"horizon_{h}"
            / f"testing_predictions_Daily_h{h}.csv")


def _read_metric_row(tag: str, h: int, opt: str) -> dict | None:
    path = _results_csv(tag, h)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sub = df[(df["Model"] == "xgboost") & (df["Optimization"].str.upper() == opt)]
    if sub.empty:
        return None
    row = sub.iloc[0].to_dict()
    row["Horizon"] = h
    row["Config"] = TAG_TO_CONFIG[tag]
    row["Ablation"] = tag
    row["Optimization"] = opt
    return row


def build_table_4_6_base(out_dir: Path) -> pd.DataFrame:
    """Tabel 4.6 — BASE testing metrics for the 4x7 grid."""
    rows: List[dict] = []
    for tag in TAG_MODULES:
        for h in HORIZONS:
            r = _read_metric_row(tag, h, "BASE")
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)[["Horizon", "Config", "Ablation", "Optimization",
                             "MAPE", "sMAPE", "RMSE",
                             "Directional_Accuracy", "n_samples"]]
    df = df.sort_values(["Horizon", "Config"]).reset_index(drop=True)
    df.to_csv(out_dir / "tabel_4.6_base_testing.csv", index=False)
    return df


def pick_winners_by_base_rmse(table_46: pd.DataFrame) -> Dict[int, str]:
    """Best tag per horizon by minimum BASE RMSE on the test set.

    User decision 2026-05-23 (was MAPE). RMSE matches CSA's training-loss
    objective (squared error on log-return) and is less anchor-inflated.
    """
    winners: Dict[int, str] = {}
    for h in HORIZONS:
        sub = table_46[table_46["Horizon"] == h].dropna(subset=["RMSE"])
        if sub.empty:
            continue
        best = sub.sort_values("RMSE").iloc[0]
        winners[h] = best["Ablation"]
    return winners


def build_table_4_7_csa(winners: Dict[int, str], out_dir: Path) -> pd.DataFrame:
    """Tabel 4.7 — CSA testing metrics for the winning ablation per horizon."""
    rows: List[dict] = []
    for h, tag in sorted(winners.items()):
        r = _read_metric_row(tag, h, "CSA")
        if r:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)[["Horizon", "Config", "Ablation", "Optimization",
                             "MAPE", "sMAPE", "RMSE",
                             "Directional_Accuracy", "n_samples"]]
    df.to_csv(out_dir / "tabel_4.7_csa_winners_testing.csv", index=False)
    return df


# ----- DM HLN helpers -------------------------------------------------------

def _load_errors_logreturn(tag: str, h: int, opt: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (errors_in_logreturn_space, predictions_logreturn) for (tag, h, opt)."""
    path = _predictions_csv(tag, h)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = f"xgboost_{opt.lower()}_LogReturn"
    if col not in df.columns or "Actual_LogReturn" not in df.columns:
        return None
    y_true = df["Actual_LogReturn"].to_numpy(dtype=float)
    y_pred = df[col].to_numpy(dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    err = y_true[mask] - y_pred[mask]
    return err, y_pred[mask]


def build_table_4_8_dm_base_pairwise(out_dir: Path) -> pd.DataFrame:
    """Tabel 4.8 — BASE pairwise DM HLN across C1-C4 per horizon."""
    import itertools
    pairs = list(itertools.combinations(TAG_MODULES.keys(), 2))
    rows: List[dict] = []
    for h in HORIZONS:
        for tag_a, tag_b in pairs:
            ea = _load_errors_logreturn(tag_a, h, "BASE")
            eb = _load_errors_logreturn(tag_b, h, "BASE")
            if ea is None or eb is None:
                continue
            # Align lengths defensively (should match by construction).
            n = min(len(ea[0]), len(eb[0]))
            dm, p = diebold_mariano_test(ea[0][:n], eb[0][:n], h=h, loss="squared")
            rows.append({
                "Horizon": h,
                "Model_A": TAG_TO_CONFIG[tag_a],
                "Model_B": TAG_TO_CONFIG[tag_b],
                "n": n,
                "DM_star": round(dm, 4) if not np.isnan(dm) else float("nan"),
                "p_value": round(p, 4) if not np.isnan(p) else float("nan"),
                "significant_at_0.05": (not np.isnan(p)) and (p < 0.05),
                "better_model": _verdict(dm, p, TAG_TO_CONFIG[tag_a],
                                         TAG_TO_CONFIG[tag_b]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tabel_4.8_dm_base_pairwise.csv", index=False)
    return df


def build_table_4_9_dm_csa_vs_base(winners: Dict[int, str], out_dir: Path
                                     ) -> pd.DataFrame:
    """Tabel 4.9 — for each winner (tag, horizon), DM HLN of CSA vs BASE.

    Tests whether CSA actually improved over BASE for the winning ablation.
    """
    rows: List[dict] = []
    for h, tag in sorted(winners.items()):
        ea = _load_errors_logreturn(tag, h, "CSA")
        eb = _load_errors_logreturn(tag, h, "BASE")
        if ea is None or eb is None:
            continue
        n = min(len(ea[0]), len(eb[0]))
        dm, p = diebold_mariano_test(ea[0][:n], eb[0][:n], h=h, loss="squared")
        rows.append({
            "Horizon": h,
            "Winner_Config": TAG_TO_CONFIG[tag],
            "n": n,
            "DM_star": round(dm, 4) if not np.isnan(dm) else float("nan"),
            "p_value": round(p, 4) if not np.isnan(p) else float("nan"),
            "significant_at_0.05": (not np.isnan(p)) and (p < 0.05),
            "better_model": _verdict(dm, p, "CSA", "BASE"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tabel_4.9_dm_csa_vs_base_winners.csv", index=False)
    return df


def build_pt_directional(out_dir: Path) -> Path | None:
    """Run Pesaran-Timmermann across C1-C4 x horizons; copy CSV into bundle.

    H3 framing: HMM-bearing configs (C2, C4) should show PT-significant
    directional skill on more horizons than price-only (C1). PT is per-
    config (not a between-config delta test) — see Bab 1.4.8 / 2.5.3.2.
    """
    pt_module.run()
    src = Path(pt_module.OUTPUT_DIR) / f"pt_directional_{pt_module.INTERVAL}_testing.csv"
    if not src.exists():
        return None
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def build_h4_sufficiency(out_dir: Path) -> pd.DataFrame:
    """H4 sufficiency — C4 BASE vs naive RW per horizon (council 6d honest skill signal)."""
    rows: List[dict] = []
    for h in HORIZONS:
        path = _predictions_csv("full", h)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Actual_LogReturn" not in df.columns or "xgboost_base_LogReturn" not in df.columns:
            continue
        y_true = df["Actual_LogReturn"].to_numpy(dtype=float)
        y_c4   = df["xgboost_base_LogReturn"].to_numpy(dtype=float)
        mask = ~(np.isnan(y_true) | np.isnan(y_c4))
        err_c4    = y_true[mask] - y_c4[mask]
        err_naive = y_true[mask] - 0.0          # naive RW predicts lr=0
        n = int(mask.sum())
        dm, p = diebold_mariano_test(err_c4, err_naive, h=h, loss="squared")
        rows.append({
            "Horizon": h,
            "n": n,
            "DM_star": round(dm, 4) if not np.isnan(dm) else float("nan"),
            "p_value": round(p, 4) if not np.isnan(p) else float("nan"),
            "significant_at_0.05": (not np.isnan(p)) and (p < 0.05),
            "better_model": _verdict(dm, p, "C4_full_BASE", "naive_rw"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "h4_sufficiency_C4_vs_naive_rw.csv", index=False)
    return df


def _verdict(dm: float, p: float, label_a: str, label_b: str) -> str:
    if np.isnan(dm) or np.isnan(p):
        return "undetermined"
    if p >= 0.05:
        return "tie"
    return label_a if dm < 0 else label_b


# ----- H2 deliverables ------------------------------------------------------

def build_h2_feature_importance(out_dir: Path) -> pd.DataFrame:
    src = (_HERE / "output_horizons" / "cpo_only" / "Daily" / "horizon_1"
           / "feature_importance_base_h1.csv")
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src)
    top5 = df.head(5).copy()
    top5.to_csv(out_dir / "h2_feature_importance_C1_h1_top5.csv", index=False)
    return top5


def build_h2_acf(out_dir: Path) -> pd.DataFrame:
    """ACF lag 1..5 of CPO daily log-return."""
    cpo_csv = REPO_ROOT / "cpo" / "output" / "cpo_variables_Daily.csv"
    if not cpo_csv.exists():
        return pd.DataFrame()
    cpo = pd.read_csv(cpo_csv, parse_dates=["Date"]).sort_values("Date")
    lr = np.log(cpo["Close"] / cpo["Close"].shift(1)).dropna().to_numpy()
    lr = lr - lr.mean()
    denom = float((lr ** 2).sum())
    rows = []
    for k in range(1, 6):
        num = float((lr[k:] * lr[:-k]).sum())
        rows.append({"lag": k, "acf": round(num / denom, 6)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "h2_acf_close_lag1_5.csv", index=False)

    # Copy the existing ACF/PACF figure into the bundle if present.
    src_png = _HERE / "acf_pacf_log_return_Daily.png"
    if src_png.exists():
        shutil.copy2(src_png, out_dir / "acf_pacf_log_return_Daily.png")
    return df


# ----- Gambar 4.3 -----------------------------------------------------------

def build_figure_4_3(table_46: pd.DataFrame, table_47: pd.DataFrame,
                     out_dir: Path) -> None:
    """Two-panel figure: MAPE and DA across horizons, BASE all four + CSA winner."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = {"C1": "#5DA5DA", "C2": "#FAA43A",
               "C3": "#60BD68", "C4": "#F17CB0"}

    for metric, ax in zip(["MAPE", "Directional_Accuracy"], axes):
        # BASE lines per config
        for cfg, sub in table_46.groupby("Config"):
            ax.plot(sub["Horizon"], sub[metric],
                    marker="o", linestyle="--", linewidth=1.4,
                    color=palette.get(cfg, "#999"),
                    label=f"{cfg} BASE")
        # CSA winner overlay
        if not table_47.empty:
            ax.plot(table_47["Horizon"], table_47[metric],
                    marker="s", linestyle="-", linewidth=2.0,
                    color="black", label="CSA winner")
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric.replace("_", " "))
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle("Gambar 4.3 — MAPE & Directional Accuracy across horizons "
                 "(BASE C1-C4 vs CSA winner)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_4_3_mape_da_across_horizons.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----- SUMMARY.md -----------------------------------------------------------

def write_summary(out_dir: Path,
                  t46: pd.DataFrame, t47: pd.DataFrame,
                  t48: pd.DataFrame, t49: pd.DataFrame,
                  h4: pd.DataFrame, h2_imp: pd.DataFrame, h2_acf: pd.DataFrame,
                  pt: pd.DataFrame,
                  winners: Dict[int, str], csa_config: dict) -> None:
    today = date.today().isoformat()
    lines: List[str] = []
    P = lines.append

    P(f"# THESIS_RESULTS — {today}")
    P("")
    P("Generated by `prediction/fase4_thesis_runner.py` on branch "
      "`fix/critical-major-cleanup`. BASE = true XGBoost library defaults "
      "(FASE 2.3); CSA budget for this scoped run: "
      f"pop={csa_config['population_size']}, iter={csa_config['max_iterations']}, "
      f"cv={csa_config['cv_folds']}.")
    P("")
    P("CSA was run only on the 7 winning (Config, Horizon) cells picked by "
      "minimum BASE MAPE — the user's scoped plan in lieu of the full "
      "4 x 7 CSA grid. Pairwise DM tests therefore use BASE for fairness "
      "(Tabel 4.8); Tabel 4.9 confirms whether CSA actually improved over "
      "BASE on each winner.")
    P("")

    P("## Winners by horizon (min BASE RMSE)")
    P("")
    P("| Horizon | Winner config | Winner tag |")
    P("|---|---|---|")
    for h in HORIZONS:
        t = winners.get(h, "")
        P(f"| {h} | {TAG_TO_CONFIG.get(t, '-')} | `{t}` |")
    P("")

    def _md_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_(empty)_"
        return df.to_markdown(index=False)

    P("## Tabel 4.6 — BASE testing metrics (4 x 7 grid)")
    P("")
    P("Source: `tabel_4.6_base_testing.csv`")
    P("")
    P(_md_table(t46))
    P("")

    P("## Tabel 4.7 — CSA testing metrics (winners only)")
    P("")
    P("Source: `tabel_4.7_csa_winners_testing.csv`")
    P("")
    P(_md_table(t47))
    P("")

    P("## Tabel 4.8 — BASE pairwise Diebold-Mariano (HLN-corrected)")
    P("")
    P("Source: `tabel_4.8_dm_base_pairwise.csv`. `better_model = tie` when "
      "p >= 0.05; otherwise the side with lower squared loss is reported.")
    P("")
    P(_md_table(t48))
    P("")

    P("## Tabel 4.9 — CSA vs BASE on winners (DM HLN)")
    P("")
    P("Source: `tabel_4.9_dm_csa_vs_base_winners.csv`. Per winner cell, "
      "tests whether CSA-tuned hyperparams beat the BASE defaults.")
    P("")
    P(_md_table(t49))
    P("")

    P("## H3 — Pesaran-Timmermann directional skill per ablation")
    P("")
    P("Source: `pt_directional_Daily_testing.csv`. PT(1992) is per-config "
      "(H0: arah prediksi independen dari arah aktual); H3 is supported "
      "when HMM-bearing configs (C2, C4) are PT-significant on more "
      "horizons than C1.")
    P("")
    P(_md_table(pt))
    P("")

    P("## H4 sufficiency — C4 BASE vs naive random walk")
    P("")
    P("Source: `h4_sufficiency_C4_vs_naive_rw.csv`. Naive RW predicts "
      "log-return = 0, so its forecast error equals the realised log-return.")
    P("")
    P(_md_table(h4))
    P("")

    P("## H2 — feature importance C1 BASE h=1 (top 5)")
    P("")
    P("Source: `h2_feature_importance_C1_h1_top5.csv`")
    P("")
    P(_md_table(h2_imp))
    P("")

    P("## H2 — ACF lag 1..5 of CPO daily log-return")
    P("")
    P("Source: `h2_acf_close_lag1_5.csv` (companion figure: "
      "`acf_pacf_log_return_Daily.png`)")
    P("")
    P(_md_table(h2_acf))
    P("")

    P("## Gambar 4.3 — MAPE & Directional Accuracy across horizons")
    P("")
    P("Source: `figure_4_3_mape_da_across_horizons.png` "
      "(BASE C1-C4 as dashed lines + CSA winner overlay).")
    P("")

    P("## Reproducibility")
    P("")
    P("- Branch: `fix/critical-major-cleanup`")
    P("- Code seed: `random_state=42` end to end")
    P("- BASE: true XGBoost defaults (no `early_stopping_rounds`, no custom "
      "eval_metric)")
    P("- CSA objective: mean CV RMSE on the log-return target (FASE 2.2)")
    P("- DM: Harvey-Leybourne-Newbold corrected, two-sided p from t_{n-1}")
    P(f"- VAL_CUTOFF: {VAL_CUTOFF.date()} (Date<cutoff = CV; Date>=cutoff = test)")
    P("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_winners_payload(winners: Dict[int, str], out_dir: Path,
                          base_table: pd.DataFrame, csa_table: pd.DataFrame
                          ) -> None:
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "horizons": HORIZONS,
        "tags": list(TAG_MODULES.keys()),
        "tag_to_config": TAG_TO_CONFIG,
        "winners_by_horizon": {str(h): t for h, t in winners.items()},
        "configs_by_horizon": {str(h): TAG_TO_CONFIG[t] for h, t in winners.items()},
    }
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]] = {}
    for tag in TAG_MODULES:
        metrics[tag] = {}
        for h in HORIZONS:
            metrics[tag][str(h)] = {}
            base = base_table[(base_table["Ablation"] == tag) & (base_table["Horizon"] == h)]
            csa  = csa_table[(csa_table["Ablation"]  == tag) & (csa_table["Horizon"]  == h)] \
                if not csa_table.empty else pd.DataFrame()
            if not base.empty:
                r = base.iloc[0]
                metrics[tag][str(h)]["BASE"] = {
                    "mape": float(r["MAPE"]), "smape": float(r["sMAPE"]),
                    "rmse": float(r["RMSE"]),  "da": float(r["Directional_Accuracy"]),
                }
            if not csa.empty:
                r = csa.iloc[0]
                metrics[tag][str(h)]["CSA"] = {
                    "mape": float(r["MAPE"]), "smape": float(r["sMAPE"]),
                    "rmse": float(r["RMSE"]),  "da": float(r["Directional_Accuracy"]),
                }
    payload["metrics"] = metrics

    (out_dir / "winners.json").write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")
    # Also write to prediction/winners.json so the website/scheduler stack
    # can pick up the updated picks.
    (_HERE / "winners.json").write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csa-population", type=int, default=DEFAULT_CSA_POPULATION)
    ap.add_argument("--csa-iterations", type=int, default=DEFAULT_CSA_ITERATIONS)
    ap.add_argument("--csa-cv-folds", type=int, default=5)
    ap.add_argument("--skip-base-pass", action="store_true",
                    help="Use existing testing_results_h{h}.csv (assumes BASE pass "
                         "already done) and only run the CSA pass + aggregation.")
    ap.add_argument("--skip-csa-pass", action="store_true",
                    help="Skip CSA on winners; Tabel 4.7 + 4.9 will be empty.")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO_ROOT / f"THESIS_RESULTS_{date.today():%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTHESIS_RESULTS bundle -> {out_dir}")

    t_total = time.time()

    if not args.skip_base_pass:
        print("\n[Step 1] BASE pass (4 ablations x 7 horizons, no CSA)")
        run_pass({"enabled": False, "population_size": 0,
                  "max_iterations": 0, "cv_folds": 3}, only=None)

    print("\n[Step 2] Aggregate BASE -> Tabel 4.6; pick winners")
    t46 = build_table_4_6_base(out_dir)
    winners = pick_winners_by_base_rmse(t46)
    print(f"  Winners by horizon: {winners}")

    if not args.skip_csa_pass:
        print(f"\n[Step 3] CSA pass on {len(winners)} winners "
              f"(pop={args.csa_population}, iter={args.csa_iterations})")
        winner_pairs = [(t, h) for h, t in winners.items()]
        run_pass({"enabled": True,
                  "population_size": args.csa_population,
                  "max_iterations": args.csa_iterations,
                  "cv_folds": args.csa_cv_folds}, only=winner_pairs)

    print("\n[Step 4] Build deliverables")
    t47 = build_table_4_7_csa(winners, out_dir)
    t48 = build_table_4_8_dm_base_pairwise(out_dir)
    t49 = build_table_4_9_dm_csa_vs_base(winners, out_dir)
    pt_csv = build_pt_directional(out_dir)
    pt_df = pd.read_csv(pt_csv) if pt_csv is not None else pd.DataFrame()
    h4   = build_h4_sufficiency(out_dir)
    h2_i = build_h2_feature_importance(out_dir)
    h2_a = build_h2_acf(out_dir)
    build_figure_4_3(t46, t47, out_dir)
    write_winners_payload(winners, out_dir, t46, t47)
    write_summary(out_dir, t46, t47, t48, t49, h4, h2_i, h2_a, pt_df,
                  winners,
                  {"population_size": args.csa_population,
                   "max_iterations": args.csa_iterations,
                   "cv_folds": args.csa_cv_folds})

    print(f"\nDONE in {time.time()-t_total:.1f}s")
    print(f"Bundle: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

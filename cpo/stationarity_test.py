"""
CPO Log Return Stationarity Tests
==================================
Tests whether Log_Return in each frequency CSV is stationary using:
  - ADF  (Augmented Dickey-Fuller)  — H0: unit root (non-stationary)
  - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) — H0: stationary
  - PP   (Phillips-Perron)          — H0: unit root (non-stationary)

A series is considered stationary when:
  ADF  → p < 0.05  (reject H0 → no unit root → stationary)
  KPSS → p > 0.05  (fail to reject H0 → stationary)
  PP   → p < 0.05  (reject H0 → no unit root → stationary)
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")

ALPHA = 0.05

FILES = {
    "Daily":   os.path.join(os.path.dirname(__file__), "output", "cpo_variables_Daily.csv"),
    "Weekly":  os.path.join(os.path.dirname(__file__), "output", "cpo_variables_Weekly.csv"),
    "Monthly": os.path.join(os.path.dirname(__file__), "output", "cpo_variables_Monthly.csv"),
}


def run_adf(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test (autolag='AIC')."""
    stat, p, lags, nobs, crit, _ = adfuller(series, autolag="AIC")
    return {
        "stat":      round(stat, 4),
        "p_value":   round(p, 4),
        "lags":      lags,
        "nobs":      nobs,
        "crit_1pct": round(crit["1%"], 4),
        "crit_5pct": round(crit["5%"], 4),
        "crit_10pct":round(crit["10%"], 4),
        "stationary": p < ALPHA,
    }


def run_kpss(series: pd.Series) -> dict:
    """KPSS test (regression='c' = level stationarity)."""
    stat, p, lags, crit = kpss(series, regression="c", nlags="auto")
    return {
        "stat":      round(stat, 4),
        "p_value":   round(p, 4),
        "lags":      lags,
        "crit_1pct": round(crit["1%"], 4),
        "crit_5pct": round(crit["5%"], 4),
        "crit_10pct":round(crit["10%"], 4),
        # KPSS H0 = stationary, so we FAIL to reject H0 when p > alpha
        "stationary": p > ALPHA,
    }


def run_pp(series: pd.Series) -> dict:
    """Phillips-Perron test (arch library)."""
    pp = PhillipsPerron(series)
    return {
        "stat":      round(float(pp.stat), 4),
        "p_value":   round(float(pp.pvalue), 4),
        "lags":      pp.lags,
        "crit_1pct": round(float(pp.critical_values["1%"]), 4),
        "crit_5pct": round(float(pp.critical_values["5%"]), 4),
        "crit_10pct":round(float(pp.critical_values["10%"]), 4),
        "stationary": pp.pvalue < ALPHA,
    }


def verdict(adf_stat: bool, kpss_stat: bool, pp_stat: bool) -> str:
    votes = sum([adf_stat, kpss_stat, pp_stat])
    if votes == 3:
        return "STATIONARY  (all 3 tests agree)"
    elif votes == 2:
        return "LIKELY STATIONARY  (2/3 tests)"
    elif votes == 1:
        return "LIKELY NON-STATIONARY  (1/3 tests)"
    else:
        return "NON-STATIONARY  (all 3 tests agree)"


def print_result(name: str, result: dict, h0_label: str, stat_if: str):
    flag = "STATIONARY" if result["stationary"] else "NON-STATIONARY"
    print(f"    Stat    : {result['stat']:>10.4f}   "
          f"(crit 1%={result['crit_1pct']}, 5%={result['crit_5pct']}, 10%={result['crit_10pct']})")
    print(f"    p-value : {result['p_value']:>10.4f}   → {flag}  [{stat_if}]")
    print(f"    Lags    : {result['lags']}")


def test_frequency(freq: str, filepath: str):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  FREQUENCY: {freq.upper()}")
    print(sep)

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    series = df["Log_Return"].dropna()
    n_raw  = len(df)
    n_used = len(series)
    print(f"  Records: {n_raw} total, {n_used} usable (NaN dropped)")
    print(f"  Log Return — mean={series.mean():.6f}, std={series.std():.6f}, "
          f"min={series.min():.4f}, max={series.max():.4f}\n")

    # ── ADF ──────────────────────────────────────────────────────────────────
    print("  [1] Augmented Dickey-Fuller (ADF)")
    print("      H0: unit root exists (non-stationary)   → reject if p < 0.05")
    adf = run_adf(series)
    print_result("ADF", adf, "H0: unit root", "p < 0.05 → stationary")

    # ── KPSS ─────────────────────────────────────────────────────────────────
    print("\n  [2] KPSS (level stationarity)")
    print("      H0: series is stationary                → reject if p < 0.05")
    kpss_r = run_kpss(series)
    print_result("KPSS", kpss_r, "H0: stationary", "p > 0.05 → stationary")

    # ── Phillips-Perron ───────────────────────────────────────────────────────
    print("\n  [3] Phillips-Perron (PP)")
    print("      H0: unit root exists (non-stationary)   → reject if p < 0.05")
    pp = run_pp(series)
    print_result("PP", pp, "H0: unit root", "p < 0.05 → stationary")

    # ── Verdict ───────────────────────────────────────────────────────────────
    v = verdict(adf["stationary"], kpss_r["stationary"], pp["stationary"])
    print(f"\n  ─── VERDICT: {v}")

    return {
        "Frequency": freq,
        "N":         n_used,
        "ADF_stat":  adf["stat"],   "ADF_p":  adf["p_value"],  "ADF_ok":  adf["stationary"],
        "KPSS_stat": kpss_r["stat"],"KPSS_p": kpss_r["p_value"],"KPSS_ok": kpss_r["stationary"],
        "PP_stat":   pp["stat"],    "PP_p":   pp["p_value"],   "PP_ok":   pp["stationary"],
        "Verdict":   v,
    }


def main():
    print("\n" + "=" * 65)
    print("  CPO LOG RETURN STATIONARITY TESTS")
    print("  Tests: ADF | KPSS | Phillips-Perron   α = 0.05")
    print("=" * 65)

    rows = []
    for freq, path in FILES.items():
        if not os.path.exists(path):
            print(f"\n  [SKIP] {freq}: file not found → {path}")
            continue
        try:
            row = test_frequency(freq, path)
            rows.append(row)
        except Exception as exc:
            print(f"\n  [ERROR] {freq}: {exc}")

    if rows:
        summary = pd.DataFrame(rows)
        print("\n\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        cols = ["Frequency", "N", "ADF_p", "ADF_ok", "KPSS_p", "KPSS_ok", "PP_p", "PP_ok", "Verdict"]
        print(summary[cols].to_string(index=False))

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()

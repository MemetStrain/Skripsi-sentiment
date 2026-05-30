"""Inference-only deliverable refresh — predict the test split with the
ALREADY-TRAINED saved models, without re-running BASE or CSA training.

Why this exists
---------------
The CSA models in ``saved_models/{tag}/Daily/h{h}/xgboost_csa`` were trained on
the pre-cutoff window. When the run that trained them happened, the HMM states
CSV was stale (ended 2024-12-31), so the C2/C4 inner-join dropped the entire
2025-2026 test window and ``testing_results_*.csv`` was never written with the
test metrics. The HMM is decoded *causally* (forward filter), so regenerating
it to cover the test window does NOT change any pre-cutoff state label — the
saved models stay valid. We only need to run inference on the now-available
test set and rewrite the testing CSVs.

For each (tag, horizon) and each variant present on disk (BASE / CSA) this:
  1. rebuilds the exact training feature matrix (load_and_merge_data ->
     engineer_features_for_horizon -> prepare_cv_test_split),
  2. loads the saved model.pkl,
  3. predicts the test split (data['X_test'] is scaled by the same RobustScaler
     recipe the model was trained with — identical because X_pre is identical),
  4. rewrites testing_predictions_*.csv + testing_results_*.csv via the module's
     own _save_split_outputs (byte-for-byte the same format as training).

After this, build the bundle without retraining:
    fase4_thesis_runner.py --skip-base-pass --skip-csa-pass --force-deliverables
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import horizon_forecast_C1_price_only as c1mod      # noqa: E402
import horizon_forecast_C2_price_hmm as c2mod       # noqa: E402
import horizon_forecast_C3_price_sentiment as c3mod  # noqa: E402
import horizon_forecast_C4_full as c4mod            # noqa: E402
from utils.forecast_utils import (                  # noqa: E402
    prepare_cv_test_split, calculate_metrics, VAL_CUTOFF,
)

HORIZONS = list(range(1, 8))
MODELS_DIR = _HERE / "saved_models"

TAG_MODULES = {
    "cpo_only":      c1mod,
    "cpo_hmm":       c2mod,
    "cpo_sentiment": c3mod,
    "full":          c4mod,
}


def _saved_dir(tag: str, h: int, variant: str) -> Path:
    return MODELS_DIR / tag / "Daily" / f"h{h}" / f"xgboost_{variant}"


def _saved_feature_cols(saved_dir: Path) -> list | None:
    """The feature schema the saved model was trained on (None if no meta)."""
    meta = saved_dir / "meta.json"
    if not meta.exists():
        return None
    return list(json.loads(meta.read_text(encoding="utf-8")).get("feature_cols", []))


def run_tag(tag: str, mod) -> int:
    cfg = mod.INTERVAL_CONFIGS["Daily"]
    merged = mod.load_and_merge_data("Daily")
    output_dir = _HERE / "output_horizons" / mod.SCRIPT_TAG
    written = 0

    for h in HORIZONS:
        df, feature_cols = mod.engineer_features_for_horizon(merged, h)
        data = prepare_cv_test_split(df, feature_cols, cfg["cv_folds"], VAL_CUTOFF)

        if not len(data["dates_test"]):
            print(f"  [skip] {tag} h{h}: no test rows (Date >= {VAL_CUTOFF.date()})")
            continue

        test_preds: dict = {}
        test_results: dict = {}
        for variant in ("base", "csa"):
            sdir = _saved_dir(tag, h, variant)
            mp = sdir / "model.pkl"
            if not mp.exists():
                continue
            # Guard: only run a saved model whose feature schema matches the
            # CURRENT feature engineering. Stale models (e.g. the 05-23 CSA
            # set trained with Volume_lag1, now dropped from master_features)
            # are skipped instead of raising a shape mismatch.
            saved_cols = _saved_feature_cols(sdir)
            if saved_cols is not None and saved_cols != feature_cols:
                extra = [c for c in saved_cols if c not in feature_cols]
                missing = [c for c in feature_cols if c not in saved_cols]
                print(f"  [stale] {tag} h{h} {variant.upper()}: schema mismatch "
                      f"(saved n={len(saved_cols)} vs current n={len(feature_cols)}; "
                      f"saved-only={extra or '-'} current-only={missing or '-'}) — skipped")
                continue
            model = joblib.load(mp)
            y_pred = model.predict(data["X_test"])
            key = f"xgboost_{variant}"
            test_preds[key] = y_pred
            test_results[key] = calculate_metrics(
                data["y_test"], y_pred, data["close_test"])

        if not test_results:
            print(f"  [skip] {tag} h{h}: no saved models found")
            continue

        horizon_dir = os.path.join(str(output_dir), "Daily", f"horizon_{h}")
        os.makedirs(horizon_dir, exist_ok=True)
        mod._save_split_outputs(
            "testing", data["dates_test"], data["y_test"], data["close_test"],
            test_preds, test_results, horizon_dir, "Daily", h)

        tags_done = ", ".join(sorted(k.split("_")[1].upper() for k in test_results))
        n = test_results[next(iter(test_results))]["n_samples"]
        print(f"  [ok]   {tag} h{h}: wrote testing CSV ({tags_done}; n_test={n})")
        written += 1

    return written


def main() -> int:
    print(f"Inference-only test refresh  (test split: Date >= {VAL_CUTOFF.date()})")
    total = 0
    for tag, mod in TAG_MODULES.items():
        print(f"\n{'='*60}\n  {tag}\n{'='*60}")
        total += run_tag(tag, mod)
    print(f"\nDONE — rewrote {total} horizon testing CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

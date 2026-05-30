"""
Canonical date-window constants for the CPO price-prediction pipeline.

Single source of truth. Every script must import from here instead of
hardcoding date literals.

Modeling window (applies to price, sentiment, hmm_state used as model inputs/targets):
  FULL_START / FULL_END  -- full usable window after HMM warmup data loss
  TRAIN_START / TRAIN_END -- training subset (feature selection, model fit)
  TEST_START  / TEST_END  -- held-out evaluation; NO fitting, no selection here

HMM-specific window:
  HMM_FIT_RAW_START / HMM_FIT_RAW_END -- the HMM is FIT on this raw CPO series.
  HMM states are available from FULL_START (2015-08-03) because the rolling
  Z-score normalisation window (252 days) + volatility window (20 days) consumes
  the head of the raw series -- this data loss is expected and documented.

All boundaries are inclusive on both ends (>= start, <= end).
"""

import pandas as pd

# ---------------------------------------------------------------------------
# String constants  (use these for pd.read_csv parse / .query / display)
# ---------------------------------------------------------------------------

FULL_START   = "2015-08-03"   # first usable date after HMM warmup
FULL_END     = "2026-02-28"   # 2026 is not a leap year; Feb has 28 days

TRAIN_START  = "2015-08-03"
TRAIN_END    = "2024-12-31"

TEST_START   = "2025-01-01"   # = VAL_CUTOFF used throughout prediction/
TEST_END     = "2026-02-28"

HMM_FIT_RAW_START = "2015-01-01"   # raw CPO series fed to the HMM fitter
HMM_FIT_RAW_END   = "2024-12-31"   # HMM is fit on train only (no test leakage)

# ---------------------------------------------------------------------------
# Timestamp constants  (use these for DataFrame boolean indexing)
# ---------------------------------------------------------------------------

FULL_START_TS   = pd.Timestamp(FULL_START)
FULL_END_TS     = pd.Timestamp(FULL_END)

TRAIN_START_TS  = pd.Timestamp(TRAIN_START)
TRAIN_END_TS    = pd.Timestamp(TRAIN_END)

TEST_START_TS   = pd.Timestamp(TEST_START)   # also = VAL_CUTOFF
TEST_END_TS     = pd.Timestamp(TEST_END)

HMM_FIT_RAW_START_TS = pd.Timestamp(HMM_FIT_RAW_START)
HMM_FIT_RAW_END_TS   = pd.Timestamp(HMM_FIT_RAW_END)

# Convenience alias so prediction/ can reference the canonical cutoff by name
VAL_CUTOFF = TEST_START_TS

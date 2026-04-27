"""
Hidden Markov Model (HMM) for CPO Price State Analysis
=======================================================
Processes Daily data only (Weekly / Monthly variants dropped from scope
during the 2026-04-26 thesis-scope-reduction sweep).

Statistical Design
------------------
Features (5 per observation):
  1. Log_Return_Z      — rolling Z-score of log returns          (stationary)
  2. Volatility_Z      — rolling Z-score of rolling-std(returns) (stationary)
  3. RSI_norm          — (RSI − 50) / 50                         (bounded ≈ −1..1)
  4. MACD_Z            — rolling Z-score of MACD line            (stationary)
  5. BB_Width_Z        — rolling Z-score of Bollinger Band Width  (stationary)

Normalization:
  All Z-scores use a rolling window (= 1 year of trading periods) so that
  at time t only information up to t is used → eliminates look-ahead bias.

Covariance type:
  daily → "full"  (≥500 obs; enough to estimate full covariance matrix)

Note: PCA is intentionally omitted.  A global PCA fit on the full time series
would introduce look-ahead bias; the GaussianHMM already captures feature
correlations through its per-state covariance matrices.

Model selection:
  Bayesian Information Criterion (BIC) over 2…MAX_STATES.
  BIC = −2 · log L + k · ln(N)
  where k = number of free model parameters, N = number of observations.
  Lower BIC ⟹ better balance of fit and parsimony.

Training:
  N_RESTARTS independent random initialisations; best log-likelihood kept.
  This guards against local optima in the Baum-Welch (EM) algorithm.

State interpretation:
  States are sorted by mean log-return (descending) and labelled:
    2 states  → Bullish | Bearish
    3 states  → Bullish | Neutral | Bearish
    4+ states → Bullish-1, Bullish-2 … Neutral … Bearish-2, Bearish-1
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
except ImportError:
    raise ImportError(
        "hmmlearn not installed. Run:  pip install hmmlearn"
    )

# ─────────────────────────────── CONFIGURATION ────────────────────────────── #

FREQUENCIES = ["Daily"]

INPUT_FILES = {
    "Daily": "../cpo/output/cpo_variables_Daily.csv",
}

# Rolling window for computing intra-period volatility (std of log returns)
VOLATILITY_WINDOW = 20   # ≈ 1 trading month

# Rolling window for Z-score normalisation (≈ 1 trading year)
NORM_WINDOW = 252

# GaussianHMM covariance type
COVARIANCE_TYPE = "full"

# BIC model-selection
AUTO_OPTIMIZE   = False  # Locked to N_STATES_MANUAL; set True to re-enable BIC search
N_STATES_MANUAL = 3      # Exactly 3 states: Bullish | Neutral | Bearish
MAX_STATES      = 10     # Upper bound (only used when AUTO_OPTIMIZE = True)

# HMM training
N_ITER      = 1000   # Max EM iterations per restart (≥ 200 required)
N_RESTARTS  = 50     # Independent K-Means-seeded restarts; best log-L is kept
TOL         = 1e-4   # EM convergence tolerance
RANDOM_SEED = 42

# Output
OUTPUT_DIR = "output"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (18, 12)

# ─────────────────────────────── DATA LOADING ─────────────────────────────── #

def load_cpo_variables(filepath: str, frequency: str) -> pd.DataFrame:
    """
    Load pre-engineered CPO variables CSV produced by the preprocessing
    pipeline.  Sorts by Date and validates required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Input file not found: {filepath}\n"
            "Run the preprocessing script first to generate cpo_variables files."
        )

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Date", "Close", "Log_Return", "RSI", "MACD", "Bollinger_Band_Width"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{frequency.upper()}] Missing columns: {missing}")

    print(
        f"[{frequency.upper()}] Loaded {len(df):,} records  "
        f"({df['Date'].min().date()} → {df['Date'].max().date()})"
    )
    return df


# ─────────────────────────────── FEATURE PREP ─────────────────────────────── #

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling Z-score using an expanding min_periods of window//2.
    At time t uses only t-window..t; no look-ahead.
    """
    min_p = max(window // 2, 2)
    mu    = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std()
    return (series - mu) / (sigma + 1e-8)


def prepare_features(df: pd.DataFrame, frequency: str):
    """
    Build the 5-dimensional stationary feature matrix used for HMM training.

    Returns
    -------
    df_clean : DataFrame with original columns + feature columns, NaN rows dropped
    feat_cols : list of 5 feature column names
    """
    df = df.copy()
    vol_win  = VOLATILITY_WINDOW
    norm_win = NORM_WINDOW

    # 1. Intra-period volatility (rolling std of log returns)
    df["Volatility"] = df["Log_Return"].rolling(vol_win, min_periods=2).std()

    # 2. Rolling Z-scores for return-based features
    df["Log_Return_Z"] = rolling_zscore(df["Log_Return"], norm_win)
    df["Volatility_Z"] = rolling_zscore(df["Volatility"], norm_win)
    df["MACD_Z"]       = rolling_zscore(df["MACD"],       norm_win)
    df["BB_Width_Z"]   = rolling_zscore(df["Bollinger_Band_Width"], norm_win)

    # 3. RSI normalisation: centre at 0, divide by 50 → approx −1..+1
    #    RSI = 50 → neutral; RSI = 100 → overbought (+1); RSI = 0 → oversold (−1)
    df["RSI_norm"] = (df["RSI"] - 50.0) / 50.0

    feat_cols = ["Log_Return_Z", "Volatility_Z", "RSI_norm", "MACD_Z", "BB_Width_Z"]

    # Drop warmup NaNs and any residual infinities
    df_clean = df.dropna(subset=feat_cols).copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=feat_cols).reset_index(drop=True)

    n_dropped = len(df) - len(df_clean)
    print(
        f"[{frequency.upper()}] Features prepared: {len(df_clean):,} usable observations "
        f"({n_dropped} warmup rows dropped)"
    )
    return df_clean, feat_cols


# ─────────────────────────────── HMM TRAINING ─────────────────────────────── #

def _fit_single(X: np.ndarray, n_states: int, cov_type: str,
                n_iter: int, tol: float, seed: int):
    """
    Fit one GaussianHMM with K-Means seeded initialisation.

    Initialisation strategy:
      - Means:       K-Means cluster centres (n_init=10 for stable centroids)
      - Covariances: per-cluster sample covariance, diagonal floored at 1e-3
      - Transition:  uniform + small random noise, renormalised row-wise
      - π:           uniform + small random noise, renormalised

    Numerical stability:
      - init_params="" prevents hmmlearn from overwriting our seeded values
      - min_covar=1e-3 floors every covariance diagonal, preventing log(0)
        in the Baum-Welch E-step (hmmlearn uses log-scale / log-sum-exp internally)

    Returns (model, log_likelihood) or (None, -inf) on failure.
    """
    try:
        rng = np.random.RandomState(seed)
        D   = X.shape[1]

        # ── K-Means seeding ──────────────────────────────────────────────────
        km = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
        km.fit(X)
        labels     = km.labels_
        init_means = km.cluster_centers_.copy()

        init_covars_full = []
        for s in range(n_states):
            cluster_X = X[labels == s]
            cov = np.cov(cluster_X.T) if len(cluster_X) > 1 else np.eye(D)
            cov += np.eye(D) * 1e-3   # diagonal floor
            init_covars_full.append(cov)

        # ── Transition matrix: uniform + small noise ─────────────────────────
        transmat  = np.full((n_states, n_states), 1.0 / n_states)
        transmat += rng.uniform(-0.05, 0.05, transmat.shape)
        transmat  = np.abs(transmat)
        transmat /= transmat.sum(axis=1, keepdims=True)

        # ── Start probabilities: uniform + small noise ───────────────────────
        startprob  = np.full(n_states, 1.0 / n_states)
        startprob += rng.uniform(-0.05, 0.05, n_states)
        startprob  = np.abs(startprob)
        startprob /= startprob.sum()

        # ── Build model with custom init ─────────────────────────────────────
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            verbose=False,
            init_params="",   # use our seeded values; skip hmmlearn's random init
            params="stmc",    # optimise: startprob, transmat, means, covars
            min_covar=1e-3,   # floor on covariance diagonal (prevents log(0))
        )
        model.startprob_ = startprob
        model.transmat_  = transmat
        model.means_     = init_means

        if cov_type == "full":
            model.covars_ = np.array(init_covars_full)
        elif cov_type == "diag":
            model.covars_ = np.maximum(
                np.array([np.diag(c) for c in init_covars_full]), 1e-3
            )
        elif cov_type == "tied":
            model.covars_ = np.mean(init_covars_full, axis=0)
        elif cov_type == "spherical":
            model.covars_ = np.array(
                [np.trace(c) / D for c in init_covars_full]
            )

        model.fit(X)
        return model, model.score(X)
    except Exception:
        return None, -np.inf


def fit_hmm_with_restarts(X: np.ndarray, n_states: int, cov_type: str,
                          n_iter: int = N_ITER, tol: float = TOL,
                          n_restarts: int = N_RESTARTS,
                          base_seed: int = RANDOM_SEED):
    """
    Train GaussianHMM with multiple random restarts.
    Returns the model with the highest log-likelihood.
    Multiple restarts guard against local optima in the Baum-Welch EM algorithm.
    """
    best_model, best_score = None, -np.inf
    for i in range(n_restarts):
        model, score = _fit_single(X, n_states, cov_type, n_iter, tol,
                                   seed=base_seed + i)
        if score > best_score:
            best_score, best_model = score, model

    return best_model, best_score


def count_free_params(n_states: int, n_features: int, cov_type: str) -> int:
    """
    Count free parameters of a GaussianHMM.

    Components:
      Transition matrix A   : n_states × (n_states − 1)      [rows sum to 1]
      Initial distribution π : n_states − 1
      Emission means        : n_states × n_features
      Emission covariances  :
        full      → n_states × n_features(n_features+1)/2
        diag      → n_states × n_features
        tied      → n_features(n_features+1)/2               [shared]
        spherical → n_states                                  [one var per state]
    """
    k_trans  = n_states * (n_states - 1)
    k_init   = n_states - 1
    k_means  = n_states * n_features
    if cov_type == "full":
        k_cov = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        k_cov = n_states * n_features
    elif cov_type == "tied":
        k_cov = n_features * (n_features + 1) // 2
    elif cov_type == "spherical":
        k_cov = n_states
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
    return k_trans + k_init + k_means + k_cov


def compute_model_scores(model, X: np.ndarray, cov_type: str) -> dict:
    """
    Compute log-likelihood, AIC, and BIC for a fitted GaussianHMM.

    AIC = -2·log L + 2·k
    BIC = -2·log L + k·ln(N)

    where k = number of free parameters (see count_free_params).
    """
    N, D  = X.shape
    log_L = model.score(X)
    k     = count_free_params(model.n_components, D, cov_type)
    aic   = -2.0 * log_L + 2.0 * k
    bic   = -2.0 * log_L + k * np.log(N)
    return {"log_L": log_L, "k": k, "N": N, "AIC": aic, "BIC": bic}


def optimize_states_bic(X: np.ndarray, max_states: int, cov_type: str,
                         frequency: str):
    """
    Select the optimal number of hidden states via BIC.

    BIC = −2 · log L + k · ln(N)

    Lower BIC = better model (balances fit against parameter complexity).
    """
    N, n_features = X.shape
    print(f"\n  BIC optimisation: testing 2–{max_states} states "
          f"[N={N}, features={n_features}, cov={cov_type}]")
    print(f"  {'states':>6}  {'log_L':>12}  {'k':>5}  {'BIC':>12}  converged")
    print("  " + "-" * 52)

    bic_scores, models = {}, {}

    for n_s in range(2, max_states + 1):
        model, log_L = fit_hmm_with_restarts(X, n_s, cov_type)
        if model is None:
            bic_scores[n_s] = np.inf
            print(f"  {n_s:>6}  {'FAILED':>12}")
            continue

        k   = count_free_params(n_s, n_features, cov_type)
        bic = -2.0 * log_L + k * np.log(N)
        bic_scores[n_s] = bic
        models[n_s]     = model

        converged = getattr(model.monitor_, "converged", "?")
        print(f"  {n_s:>6}  {log_L:>12.2f}  {k:>5}  {bic:>12.2f}  {converged}")

    optimal = min(bic_scores, key=bic_scores.get)
    print(f"\n  ✓ Optimal states for {frequency.upper()}: {optimal}  "
          f"(BIC = {bic_scores[optimal]:.2f})\n")

    return optimal, bic_scores, models.get(optimal)


# ─────────────────────────────── STATE ANALYSIS ───────────────────────────── #

def label_states(n_states: int) -> list[str]:
    """
    Generate economic regime labels for states sorted by mean log-return
    (index 0 = highest return = most bullish).
    """
    if n_states == 2:
        return ["Bullish", "Bearish"]
    if n_states == 3:
        return ["Bullish", "Neutral", "Bearish"]
    if n_states == 4:
        return ["Strong Bullish", "Mild Bullish", "Mild Bearish", "Strong Bearish"]

    # n ≥ 5: split into thirds
    labels = []
    bull_n  = n_states // 3
    bear_n  = n_states // 3
    neut_n  = n_states - bull_n - bear_n

    for i in range(bull_n):
        labels.append(f"Bullish-{i + 1}")
    for i in range(neut_n):
        labels.append(f"Neutral-{i + 1}")
    for i in range(bear_n, 0, -1):
        labels.append(f"Bearish-{i}")

    return labels


def characterize_states(df: pd.DataFrame, states: np.ndarray,
                         n_states: int, frequency: str) -> pd.DataFrame:
    """
    Compute per-state descriptive statistics and assign economic labels.
    States are sorted by mean log-return (descending → bull-to-bear order).
    """
    df = df.copy()
    df["State"] = states

    rows = []
    for s in range(n_states):
        mask = df["State"] == s
        sd   = df[mask]
        rows.append({
            "State":          s,
            "N":              int(mask.sum()),
            "Pct_Time":       round(mask.mean() * 100, 2),
            "Avg_LogReturn":  round(sd["Log_Return"].mean(), 6),
            "Std_LogReturn":  round(sd["Log_Return"].std(),  6),
            "Avg_Volatility": round(sd["Volatility"].mean(), 6)
                               if "Volatility" in df.columns else np.nan,
            "Avg_Price":      round(sd["Close"].mean(), 2),
            "Avg_RSI":        round(sd["RSI"].mean(), 2)
                               if "RSI" in df.columns else np.nan,
        })

    stats = (
        pd.DataFrame(rows)
        .sort_values("Avg_LogReturn", ascending=False)
        .reset_index(drop=True)
    )
    stats["Label"] = label_states(n_states)

    # Print summary
    print(f"\n  {'─'*65}")
    print(f"  STATE CHARACTERISTICS — {frequency.upper()}")
    print(f"  {'─'*65}")
    for _, row in stats.iterrows():
        print(
            f"  [{row['Label']:20s}] (orig state {int(row['State'])}): "
            f"{row['Pct_Time']:5.1f}% of time | "
            f"avg return = {row['Avg_LogReturn']*100:+.3f}% | "
            f"avg vol = {row['Avg_Volatility']*100:.3f}%"
        )

    return stats


def analyze_transition_matrix(model, state_stats: pd.DataFrame,
                               frequency: str) -> pd.DataFrame:
    """
    Report the transition probability matrix.
    Flags absorbing states (diagonal ≥ 0.99) which suggest over-parameterisation
    or a structural break captured as a permanent regime.
    Verifies row-stochasticity as a sanity check.
    """
    A = model.transmat_

    # Map original state indices → labels in original order
    label_map  = dict(zip(state_stats["State"].astype(int),
                          state_stats["Label"]))
    labels     = [label_map[i] for i in range(len(label_map))]

    trans_df = pd.DataFrame(A, index=labels, columns=labels).round(4)

    print(f"\n  Transition matrix — {frequency.upper()}")
    print("  (rows = from state, columns = to state)\n")
    print(trans_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Sanity check 1: row sums
    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        print(f"\n  ⚠  Row sums deviate from 1.0: {row_sums}")

    # Sanity check 2: absorbing states
    absorbing = [(labels[i], A[i, i]) for i in range(len(labels)) if A[i, i] >= 0.99]
    if absorbing:
        print(f"\n  ⚠  ABSORBING STATE WARNING (diagonal ≥ 0.99):")
        for lbl, p in absorbing:
            print(f"     {lbl}: persistence = {p:.4f}")
        print(
            "  Possible causes: too many states for available data, or a\n"
            "  unique structural break captured as a permanent regime.\n"
            "  Consider reducing MAX_STATES or checking for structural breaks."
        )

    return trans_df


def validate_model(model, X: np.ndarray, states: np.ndarray,
                   feat_cols: list, cov_type: str, frequency: str) -> dict:
    """
    Post-fit diagnostic report:
      a. Per-state means and covariance diagonals
      b. State occupancy (% of timesteps assigned to each state)
      c. Final log-likelihood, AIC, BIC
      d. Warn if any state has < 5% occupancy (collapsed / degenerate)
      e. Warn if any two state means are within 1 pooled-std (near-duplicate states)
    """
    n_states = model.n_components
    N        = X.shape[0]
    scores   = compute_model_scores(model, X, cov_type)

    print(f"\n  {'─'*65}")
    print(f"  MODEL VALIDATION REPORT — {frequency.upper()}")
    print(f"  {'─'*65}")
    print(f"  log-likelihood  : {scores['log_L']:.4f}")
    print(f"  AIC             : {scores['AIC']:.4f}")
    print(f"  BIC             : {scores['BIC']:.4f}")
    print(f"  Free params (k) : {scores['k']}")
    print(f"  Observations    : {N}")
    print(f"  Converged       : {getattr(model.monitor_, 'converged', '?')}")

    # ── a. Per-state means ───────────────────────────────────────────────────
    print(f"\n  Per-state means  [{', '.join(feat_cols)}]:")
    for s in range(n_states):
        print(f"    State {s}: {np.round(model.means_[s], 4).tolist()}")

    # ── a. Per-state covariance diagonals ────────────────────────────────────
    print(f"\n  Per-state covariance diagonals:")
    if cov_type == "full":
        state_stds = np.array([np.sqrt(np.diag(model.covars_[s])) for s in range(n_states)])
        for s in range(n_states):
            print(f"    State {s}: {np.round(np.diag(model.covars_[s]), 4).tolist()}")
    elif cov_type == "diag":
        state_stds = np.sqrt(model.covars_)
        for s in range(n_states):
            print(f"    State {s}: {np.round(model.covars_[s], 4).tolist()}")
    else:
        state_stds = None

    # ── b. State occupancy ───────────────────────────────────────────────────
    counts    = np.bincount(states, minlength=n_states)
    occupancy = counts / N * 100
    print(f"\n  State occupancy:")
    for s in range(n_states):
        warn = "  ⚠  LOW OCCUPANCY (<5%)" if occupancy[s] < 5.0 else ""
        print(f"    State {s}: {counts[s]:5d} obs  ({occupancy[s]:5.1f}%){warn}")

    # ── d. Low-occupancy warning ─────────────────────────────────────────────
    low_occ = [s for s in range(n_states) if occupancy[s] < 5.0]
    if low_occ:
        print(f"\n  ⚠  DEGENERATE STATE(S) {low_occ}: < 5% occupancy.")
        print("     Model may be over-parameterised or data lacks variety.")

    # ── e. Near-duplicate state warning ─────────────────────────────────────
    if state_stds is not None:
        pooled_std = state_stds.mean(axis=0)
        for i in range(n_states):
            for j in range(i + 1, n_states):
                diff = np.abs(model.means_[i] - model.means_[j])
                if np.all(diff < pooled_std):
                    print(
                        f"\n  ⚠  NEAR-DUPLICATE STATES {i} & {j}: means are within "
                        f"1 pooled std on every feature — may represent the same regime."
                    )

    return {
        "log_L":     scores["log_L"],
        "AIC":       scores["AIC"],
        "BIC":       scores["BIC"],
        "occupancy": occupancy.tolist(),
    }


# ───────────────────────────── VISUALISATION ──────────────────────────────── #

def _state_colors(n_states: int, state_stats: pd.DataFrame) -> dict:
    """Map original state int → colour (green = bull, red = bear, grey = neutral)."""
    if n_states == 2:
        palette = ["#2ca02c", "#d62728"]
    elif n_states == 3:
        palette = ["#2ca02c", "#7f7f7f", "#d62728"]
    else:
        cmap    = plt.cm.RdYlGn
        palette = [cmap(1 - i / (n_states - 1)) for i in range(n_states)]

    # state_stats is already sorted bull→bear; palette[0] = greenest
    return {int(row["State"]): palette[idx]
            for idx, (_, row) in enumerate(state_stats.iterrows())}


def create_visualisations(df: pd.DataFrame, states: np.ndarray,
                           state_stats: pd.DataFrame,
                           trans_df: pd.DataFrame,
                           bic_scores: dict, frequency: str,
                           output_path: str):
    """
    7-panel figure:
      1. Price with state background shading
      2. Log returns coloured by state
      3. Time spent per state (bar chart)
      4. Transition matrix heatmap
      5. Volatility coloured by state
      6. Return distribution per state (violin)
      7. BIC curve (model-selection diagnostic)
    """
    df = df.copy()
    df["State"] = states
    n_states    = len(state_stats)
    colors      = _state_colors(n_states, state_stats)
    label_map   = dict(zip(state_stats["State"].astype(int), state_stats["Label"]))

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"CPO HMM State Analysis — {frequency.upper()} "
        f"({df['Date'].min().year}–{df['Date'].max().year})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Price + state shading ──────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(df["Date"], df["Close"], "k-", lw=1.2, alpha=0.8, label="CPO Close")
    for s in sorted(df["State"].unique()):
        idx_list = df.index[df["State"] == s].tolist()
        for idx in idx_list:
            x0 = df.loc[idx, "Date"]
            x1 = df.loc[idx + 1, "Date"] if idx + 1 < len(df) else x0
            ax1.axvspan(x0, x1, alpha=0.25, color=colors[s], linewidth=0)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=colors[int(r["State"])], label=r["Label"])
                  for _, r in state_stats.iterrows()]
    ax1.legend(handles=legend_els, fontsize=8, loc="upper left")
    ax1.set_ylabel("Price (MYR/ton)")
    ax1.set_title("CPO Price with Hidden Market States", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Log returns coloured by state ───────────────────────────────
    ax2 = fig.add_subplot(4, 2, 2)
    for s in sorted(df["State"].unique()):
        sd = df[df["State"] == s]
        ax2.scatter(sd["Date"], sd["Log_Return"] * 100,
                    c=colors[s], s=4, alpha=0.6, label=label_map[s])
    ax2.axhline(0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax2.set_ylabel("Log Return (%)")
    ax2.set_title("Log Returns by State", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Time per state ──────────────────────────────────────────────
    ax3 = fig.add_subplot(4, 2, 3)
    ordered_states = state_stats["State"].astype(int).tolist()
    bar_colors     = [colors[s] for s in ordered_states]
    bar_labels     = state_stats["Label"].tolist()
    bar_vals       = state_stats["N"].tolist()
    bars = ax3.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="white", lw=0.5)
    for bar, n in zip(bars, bar_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{n}\n({n/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Observations")
    ax3.set_title("Time Spent per State", fontweight="bold")
    ax3.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Transition matrix heatmap ──────────────────────────────────
    ax4 = fig.add_subplot(4, 2, 4)
    sns.heatmap(trans_df, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax4,
                cbar_kws={"label": "Probability"},
                annot_kws={"size": 8})
    ax4.set_title("State Transition Matrix", fontweight="bold")
    ax4.set_xlabel("To State", fontsize=9)
    ax4.set_ylabel("From State", fontsize=9)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=7)

    # ── Panel 5: Volatility coloured by state ───────────────────────────────
    ax5 = fig.add_subplot(4, 2, 5)
    for s in sorted(df["State"].unique()):
        sd = df[df["State"] == s]
        ax5.scatter(sd["Date"], sd["Volatility"] * 100,
                    c=colors[s], s=4, alpha=0.6, label=label_map[s])
    ax5.set_ylabel("Volatility (%)")
    ax5.set_title("Rolling Volatility by State", fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Return distribution per state (violin) ─────────────────────
    ax6 = fig.add_subplot(4, 2, 6)
    violin_data   = [df[df["State"] == int(r["State"])]["Log_Return"].values * 100
                     for _, r in state_stats.iterrows()]
    violin_labels = state_stats["Label"].tolist()
    parts = ax6.violinplot(violin_data, positions=range(len(violin_data)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[int(state_stats.iloc[i]["State"])])
        pc.set_alpha(0.65)
    ax6.set_xticks(range(len(violin_labels)))
    ax6.set_xticklabels(violin_labels, rotation=20, ha="right", fontsize=8)
    ax6.axhline(0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax6.set_ylabel("Log Return (%)")
    ax6.set_title("Return Distribution per State", fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="y")

    # ── Panel 7: BIC selection curve ────────────────────────────────────────
    ax7 = fig.add_subplot(4, 2, 7)
    valid_bic = {k: v for k, v in bic_scores.items() if v < np.inf}
    if valid_bic:
        ns  = sorted(valid_bic.keys())
        bv  = [valid_bic[n] for n in ns]
        ax7.plot(ns, bv, "o-", color="#1f77b4", lw=1.5)
        optimal = min(valid_bic, key=valid_bic.get)
        ax7.axvline(optimal, color="red", ls="--", lw=1.2,
                    label=f"Optimal = {optimal}")
        ax7.set_xlabel("Number of Hidden States")
        ax7.set_ylabel("BIC Score")
        ax7.set_title("BIC Model-Selection Curve", fontweight="bold")
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.set_xticks(ns)

    # ── Panel 8: State sequence timeline ────────────────────────────────────
    ax8 = fig.add_subplot(4, 2, 8)
    for s in sorted(df["State"].unique()):
        mask = df["State"] == s
        ax8.scatter(df.loc[mask, "Date"], [s] * mask.sum(),
                    c=colors[s], s=6, alpha=0.7, label=label_map[s])
    ax8.set_yticks(sorted(df["State"].unique()))
    ax8.set_yticklabels([label_map[s] for s in sorted(df["State"].unique())],
                         fontsize=8)
    ax8.set_title("State Sequence Over Time", fontweight="bold")
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# ─────────────────────────────── MAIN RUNNER ──────────────────────────────── #

def run_frequency(frequency: str) -> dict:
    """
    Full HMM pipeline for one data frequency.
    Returns a dict with the key results for the summary table.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  PROCESSING: {frequency.upper()}")
    print(sep)

    # 1. Load
    df_raw = load_cpo_variables(INPUT_FILES[frequency], frequency)

    # 2. Features
    df, feat_cols = prepare_features(df_raw, frequency)
    X = df[feat_cols].values.astype(np.float64)
    cov_type = COVARIANCE_TYPE

    # 3. Model selection or manual
    if AUTO_OPTIMIZE:
        print(f"\n  Searching for optimal states via BIC…")
        n_opt, bic_scores, best_model = optimize_states_bic(
            X,
            max_states=MAX_STATES,
            cov_type=cov_type,
            frequency=frequency,
        )
        if best_model is None:
            raise RuntimeError(
                f"All HMM fits failed for {frequency}. "
                "Try reducing MAX_STATES or checking your data."
            )
    else:
        n_opt      = N_STATES_MANUAL
        bic_scores = {}
        print(f"\n  Using fixed state count: {n_opt}  ({N_RESTARTS} K-Means-seeded restarts)")
        best_model, log_L = fit_hmm_with_restarts(X, n_opt, cov_type)
        if best_model is None:
            raise RuntimeError(
                f"All HMM fits failed for {frequency}. "
                "Check your data or reduce N_STATES_MANUAL."
            )
        print(f"  Best restart log-likelihood = {log_L:.2f}  "
              f"converged = {getattr(best_model.monitor_, 'converged', '?')}")

    # 4. Decode states
    states = best_model.predict(X)

    # 5. Analyse
    state_stats = characterize_states(df, states, n_opt, frequency)
    trans_df    = analyze_transition_matrix(best_model, state_stats, frequency)

    # 5b. Post-fit validation (means, covariances, occupancy, AIC/BIC, warnings)
    val = validate_model(best_model, X, states, feat_cols, cov_type, frequency)

    # 6. Merge labels back into df
    state_to_label = dict(zip(state_stats["State"].astype(int),
                               state_stats["Label"]))
    df["State"]       = states
    df["State_Label"] = df["State"].map(state_to_label)

    # 7. Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_states = os.path.join(OUTPUT_DIR, f"hmm_states_results_{frequency}.csv")
    out_stats  = os.path.join(OUTPUT_DIR, f"hmm_states_stats_{frequency}.csv")
    out_trans  = os.path.join(OUTPUT_DIR, f"hmm_transition_matrix_{frequency}.csv")
    out_bic    = os.path.join(OUTPUT_DIR, f"hmm_bic_scores_{frequency}.csv")
    out_plot   = os.path.join(OUTPUT_DIR, f"hmm_states_analysis_{frequency}.png")

    save_cols = ["Date", "Close", "Log_Return", "Volatility",
                 "RSI", "MACD", "State", "State_Label"]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(out_states, index=False)
    state_stats.to_csv(out_stats, index=False)
    trans_df.to_csv(out_trans)
    if bic_scores:
        pd.DataFrame(list(bic_scores.items()),
                     columns=["n_states", "BIC"]).to_csv(out_bic, index=False)

    print(f"\n  Outputs:")
    print(f"    State assignments  → {out_states}")
    print(f"    State statistics   → {out_stats}")
    print(f"    Transition matrix  → {out_trans}")
    if bic_scores:
        print(f"    BIC scores         → {out_bic}")

    # 8. Visualise
    create_visualisations(df, states, state_stats,
                           trans_df, bic_scores, frequency, out_plot)

    return {
        "Frequency":      frequency.capitalize(),
        "Records":        len(df),
        "N_States":       n_opt,
        "Covariance":     cov_type,
        "Converged":      getattr(best_model.monitor_, "converged", "?"),
        "Log_Likelihood": round(val["log_L"], 2),
        "AIC":            round(val["AIC"], 2),
        "BIC":            round(val["BIC"], 2),
        "Occupancy":      " | ".join(
            f"S{s}:{v:.1f}%" for s, v in enumerate(val["occupancy"])
        ),
    }


def main():
    print("\n" + "=" * 65)
    print("  CPO HIDDEN MARKOV MODEL — ALL FREQUENCIES")
    print("  Theoretical basis: Gaussian HMM (Rabiner 1989)")
    print("  Model selection:   BIC (Schwarz 1978)")
    print("  Features:          5 stationary / rolling-normalised")
    print("=" * 65)

    summary_rows = []
    for freq in FREQUENCIES:
        try:
            row = run_frequency(freq)
            summary_rows.append(row)
        except Exception as exc:
            print(f"\n  ✗ FAILED for {freq.upper()}: {exc}")

    # Cross-frequency summary
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_summary = os.path.join(OUTPUT_DIR, "hmm_all_frequencies_summary.csv")
        summary.to_csv(out_summary, index=False)

        print("\n" + "=" * 65)
        print("  SUMMARY ACROSS ALL FREQUENCIES")
        print("=" * 65)
        print(summary.to_string(index=False))
        print(f"\n  Summary saved → {out_summary}")

    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)
    print("\n  Statistical properties ensured:")
    print("  ✓ Look-ahead-free normalisation  (rolling Z-score, window = 1 year)")
    print("  ✓ Stationarity of inputs         (log returns + derived ratios)")
    print("  ✓ Stable initialisation           (K-Means seeding, 50 restarts)")
    print("  ✓ Numerical stability             (min_covar floor, log-scale E-step)")
    print("  ✓ Fixed 3-state model             (Bullish | Neutral | Bearish)")
    print("  ✓ Post-fit validation             (AIC/BIC, occupancy, duplicate warnings)")
    print("  ✓ Absorbing-state detection        (flags degenerate solutions)")
    print("=" * 65)


if __name__ == "__main__":
    main()

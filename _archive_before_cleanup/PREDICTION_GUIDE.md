# CPO Price Prediction — Complete Pipeline Guide

This guide walks through every step required to produce forecasted CPO (Crude Palm Oil) prices, from raw news data all the way to model predictions. Each step is explained in detail including configuration options, expected outputs, and troubleshooting tips.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Scrape News Data](#3-step-1--scrape-news-data)
4. [Step 2 — Preprocess News Text](#4-step-2--preprocess-news-text)
5. [Step 3 — Sentiment Analysis (FinBERT)](#5-step-3--sentiment-analysis-finbert)
6. [Step 4 — HMM Market State Analysis](#6-step-4--hmm-market-state-analysis)
7. [Step 5 — Create Prediction Dataset (Optional)](#7-step-5--create-prediction-dataset-optional)
8. [Step 6 — Run Price Prediction Models](#8-step-6--run-price-prediction-models)
9. [Understanding the Outputs](#9-understanding-the-outputs)
10. [Choosing a Data Frequency](#10-choosing-a-data-frequency)
11. [Full Pipeline Quick Reference](#11-full-pipeline-quick-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture Overview

The prediction system works through a sequence of dependent stages. Each stage produces outputs consumed by the next:

```
[MPOB Website]
      │
      ▼
 scrap_fast.py           →  mpob_news_fast.csv
      │
      ▼
 news_preprocessing.py   →  mpob_news_preprocessed.csv
      │
      ▼
 finbert_sentiment_       →  news/output/sentiment_aggregate_*.csv
 analysis_flexible.py        news/output/monthly_sentiment_aggregate.csv
      │
      ├──────────────────────────────────────────────────────────────┐
      ▼                                                              ▼
 cpo_hmm_states.py       →  markov/output/hmm_states_results_*.csv  │
      │                                                              │
      └────────────────────────────────────────────────────────────┐ │
                                                                   ▼ ▼
                                              price_prediction_models_csa.py
                                              price_prediction_models_improved.py
                                              price_prediction_models.py
                                                       │
                                                       ▼
                                              prediction/output/
                                              prediction_results_*.csv
                                              predictions_*month.png
```

**Data flow summary:**
- News is scraped → cleaned → scored for sentiment by FinBERT
- CPO price history is fed through a Hidden Markov Model to identify hidden market states (bullish / neutral / bearish)
- The prediction models take **CPO price history + sentiment scores + HMM states** and forecast prices 1–6 months ahead

---

## 2. Prerequisites

### Python Dependencies

Install all required packages. From the project root:

```bash
cd d:\Skripsi1
pip install pandas numpy matplotlib seaborn scikit-learn xgboost hmmlearn
pip install transformers torch tqdm requests beautifulsoup4
```

If you have an NVIDIA GPU (strongly recommended for Step 3):
```bash
# Install PyTorch with CUDA support — visit https://pytorch.org/get-started/locally/
# for the correct command for your CUDA version. Example for CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:
```bash
cd d:\Skripsi1\news
python check_cuda.py
```

### Required Raw Data Files

These files must already be present in the project — they are raw inputs and are **not generated** by the pipeline:

| File | Location | Description |
|------|----------|-------------|
| `Data_CPO_Daily.csv` | `cpo/` | Daily CPO futures prices (OHLCV, Indonesian format) |
| `Data_CPO_Weekly.csv` | `cpo/` | Weekly CPO futures prices |
| `Data_CPO_Monthly.csv` | `cpo/` | Monthly CPO futures prices |

The CSV date format is `DD/MM/YYYY` and prices use Indonesian number formatting (`.` for thousands separator, `,` for decimals — e.g., `10.234,56` means 10234.56).

---

## 3. Step 1 — Scrape News Data

**Script:** `news/scrap_fast.py`
**Working directory:** `d:\Skripsi1\news\`

### What it does

Scrapes news articles about CPO from the MPOB (Malaysian Palm Oil Board) website. It searches for the keywords *"CPO"* and *"crude palm oil"*, downloading articles with their dates, headlines, and full content. It uses multi-threaded concurrent requests and saves every 50 articles to prevent data loss on interruption.

The script is **auto-resumable** — if `mpob_news_fast.csv` already exists, it reads the latest date from that file and continues from where it left off, avoiding duplicate downloads.

### Key configuration (top of file)

```python
SCRAPE_LIMIT = 240000       # Maximum articles to collect (stop condition)
MAX_WORKERS = 8             # Parallel download threads
SAVE_INTERVAL = 50          # Flush to disk every N articles
RETRY_ATTEMPTS = 3          # Retry on network failure
REQUEST_DELAY = 0.1         # Seconds between requests (be polite to server)
SCRAPE_END_DATE = "01-01-2001"  # How far back to scrape
```

> **Tip:** If you only need to update the data (not scrape from scratch), just run the script — it will automatically detect the latest date in the existing CSV and resume from there.

### How to run

```bash
cd d:\Skripsi1\news
python scrap_fast.py
```

### Output

| File | Location | Description |
|------|----------|-------------|
| `mpob_news_fast.csv` | `news/` | Raw scraped news (Date, headline, content, source) |

> **Expected size:** ~46 MB for several thousand articles spanning 2001–2025.

---

## 4. Step 2 — Preprocess News Text

**Script:** `news/news_preprocessing.py`
**Working directory:** `d:\Skripsi1\news\`

### What it does

Cleans the raw news text before it is fed into the FinBERT sentiment model. Preprocessing improves classification accuracy by removing noise:

- Strips HTML tags and encoded entities
- Removes URLs and hyperlinks
- Removes date patterns embedded in text (e.g., "January 15, 2023")
- Removes special characters while preserving sentence punctuation
- Normalizes unicode characters and whitespace
- Removes duplicate articles

### Key configuration

```python
INPUT_CSV = 'mpob_news_fast.csv'
OUTPUT_CSV = 'mpob_news_preprocessed.csv'
```

### How to run

```bash
cd d:\Skripsi1\news
python news_preprocessing.py
```

### Output

| File | Location | Description |
|------|----------|-------------|
| `mpob_news_preprocessed.csv` | `news/` | Cleaned news articles ready for FinBERT |

> **Expected size:** ~43 MB (slightly smaller than raw due to removed noise).

---

## 5. Step 3 — Sentiment Analysis (FinBERT)

**Script:** `news/finbert_sentiment_analysis_flexible.py`
**Working directory:** `d:\Skripsi1\news\`

### What it does

Runs each news article through **FinBERT**, a BERT-based transformer model fine-tuned specifically on financial text. FinBERT outputs three probability scores per article:

- `positive` — probability the article conveys positive market sentiment
- `negative` — probability the article conveys negative market sentiment
- `neutral` — probability the article is neutral/factual

After scoring all articles, it **aggregates** sentiment to the chosen time frequency (Daily, Weekly, or Monthly) by averaging probabilities across all articles published on each date. This produces a time-series of sentiment scores that can be matched against price data.

### Key configuration (top of file)

```python
INPUT_CSV = 'mpob_news_preprocessed.csv'
OUTPUT_CSV = 'mpob_news_with_sentiment.csv'   # Per-article scores (stays in news/)
BATCH_SIZE = 16              # Articles per GPU batch (auto-adjusted)
MAX_LENGTH = 512             # Token limit per article
USE_HALF_PRECISION = True    # float16 inference (faster on GPU)
FORCE_CPU = False            # Set True if you have no GPU

# IMPORTANT: Set this to match the frequency you will use for prediction
AGGREGATION_MODE = 'Daily'   # Options: 'Daily', 'Weekly', 'Monthly'
```

> **Critical:** `AGGREGATION_MODE` must match the `DATA_FREQUENCY` you plan to use in later steps. See [Choosing a Data Frequency](#10-choosing-a-data-frequency) for guidance.

### GPU memory and batch size

The script auto-adjusts `BATCH_SIZE` based on available GPU VRAM:

| GPU VRAM | Auto Batch Size |
|----------|----------------|
| < 4 GB | 4 |
| 4–8 GB | 8 |
| 8–16 GB | 16 |
| > 16 GB | 32 |

If you encounter CUDA out-of-memory errors, set `BATCH_SIZE = 4` or `FORCE_CPU = True`.

### First-run model download

FinBERT is automatically downloaded from HuggingFace on the first run (~500 MB). **Internet access is required for the first run.** Subsequent runs use the cached model.

### How to run

```bash
cd d:\Skripsi1\news
python finbert_sentiment_analysis_flexible.py
```

### Output

| File | Location | Description |
|------|----------|-------------|
| `mpob_news_with_sentiment.csv` | `news/` | Per-article sentiment scores (intermediate) |
| `sentiment_aggregate_Daily.csv` | `news/output/` | Daily-aggregated sentiment time series |
| `sentiment_aggregate_Weekly.csv` | `news/output/` | (if mode is Weekly) |
| `sentiment_aggregate_Monthly.csv` | `news/output/` | (if mode is Monthly) |
| `monthly_sentiment_aggregate.csv` | `news/output/` | Monthly aggregation (from `finbert_sentiment_analysis.py`) |

> **Note:** Run the script once per frequency mode if you need multiple frequencies (e.g., run with `'Daily'` then change to `'Monthly'` and run again).

---

## 6. Step 4 — HMM Market State Analysis

**Script:** `markov/cpo_hmm_states.py`
**Working directory:** `d:\Skripsi1\markov\`

### What it does

Applies a **Hidden Markov Model (HMM)** to CPO price history to identify hidden market *states* — recurring regimes in price behavior that are not directly observable but can be inferred from price patterns.

**Feature engineering for HMM:**
- Log returns (price changes between periods)
- Rolling volatility (standard deviation of returns)
- Price momentum (short-term vs long-term trends)
- Seasonality features (sine/cosine encoding of month)
- All features are normalized using rolling Z-scores to prevent look-ahead bias

**Model selection:**
With `AUTO_OPTIMIZE_STATES = True`, the script tests 2–12 states and selects the optimal number using the **Bayesian Information Criterion (BIC)** — balancing model fit against complexity. Typically 3–5 states are identified, corresponding to distinct market regimes (e.g., bull, bear, sideways).

**State labeling:**
After training, each state is automatically labeled based on its average return:
- State with highest average return → **Bullish** (positive)
- State with lowest average return → **Bearish** (negative)
- Remaining states → **Neutral** variants

### Key configuration

```python
DATA_FREQUENCY = 'daily'     # 'daily', 'weekly', or 'monthly'
START_YEAR = 2015
END_YEAR = 2026
AUTO_OPTIMIZE_STATES = True  # Find optimal number of states via BIC
N_STATES = 3                 # Used only if AUTO_OPTIMIZE_STATES = False
MAX_STATES = 12              # Upper bound for state search
N_ITERATIONS = 1000          # HMM EM algorithm iterations
```

> **Important:** Set `DATA_FREQUENCY` to match your chosen prediction frequency.

### How to run

```bash
cd d:\Skripsi1\markov
python cpo_hmm_states.py
```

### Output

| File | Location | Description |
|------|----------|-------------|
| `hmm_states_results_daily.csv` | `markov/output/` | Per-period state assignments (Date, State, Price, Return, Volatility, Label) |
| `hmm_transition_matrix_daily.csv` | `markov/output/` | State-to-state transition probabilities |
| `hmm_states_results_daily_stats.csv` | `markov/output/` | Per-state statistics (avg return, volatility, frequency) |
| `hmm_states_analysis_daily.png` | `markov/output/` | Visualization: price timeline with state colors + transition matrix heatmap |

---

## 7. Step 5 — Create Prediction Dataset (Optional)

**Script:** `create_prediction_dataset.py`
**Working directory:** `d:\Skripsi1\`

### What it does

This is an **optional** step that merges all three data sources (CPO prices, news sentiment, HMM states) into a single comprehensive feature-rich dataset suitable for exploratory analysis or custom model training.

It engineers 60+ features including:
- Price returns at multiple lag windows (1, 3, 5, 10 periods)
- Simple and exponential moving averages (5, 10, 20, 60 periods)
- RSI, MACD, Bollinger Bands
- Rolling volatility and Parkinson volatility
- Sentiment lag features (sentiment from 1–10 periods ago)
- HMM state one-hot encodings
- Seasonal features (quarter, month, week-of-year)
- Target variables for multiple prediction horizons (1, 3, 5, 10 periods ahead)

> **Note:** The prediction scripts (`prediction/`) load data independently and do their own feature engineering. This step creates the `cpo_prediction_dataset_daily.csv` file which can be used for **custom model development or exploration** — it is not required to run the prediction models in Step 6.

### Key configuration

```python
DATA_FREQUENCY = 'daily'     # 'daily', 'weekly', or 'monthly'
START_YEAR = 2015
END_YEAR = 2025
MAX_PRICE_LAGS = 5
MAX_SENTIMENT_LAGS = 10
```

### How to run

```bash
cd d:\Skripsi1
python create_prediction_dataset.py
```

### Output

| File | Location | Description |
|------|----------|-------------|
| `cpo_prediction_dataset_daily.csv` | `markov/` | Full feature-engineered dataset (~3.6 GB for daily) |

---

## 8. Step 6 — Run Price Prediction Models

**Working directory for all prediction scripts:** `d:\Skripsi1\prediction\`

The prediction scripts load data directly from the CPO price files, sentiment outputs, and HMM state outputs. There are three model scripts, each representing a different approach.

---

### Option A: Baseline Model — `price_prediction_models.py`

The simplest model. Uses XGBoost and Random Forest with standard hyperparameters. Good for a quick baseline and for understanding the data.

**What it does:**
1. Loads monthly CPO prices, sentiment, and HMM states
2. Creates lagged features (6-month lag window for all inputs)
3. Trains XGBoost and Random Forest separately for each prediction horizon (1–6 months ahead)
4. Uses time-series cross-validation to evaluate
5. Produces an ensemble (average of both models)

**Inputs required:**
- `cpo/Data_CPO_Monthly.csv`
- `news/output/monthly_sentiment_aggregate.csv`
- `markov/output/hmm_states_results_daily.csv`

```bash
cd d:\Skripsi1\prediction
python price_prediction_models.py
```

**Outputs:**
| File | Description |
|------|-------------|
| `output/prediction_results.csv` | RMSE, MAE, R², MAPE per model per horizon |
| `output/predictions_with_features.csv` | Actual vs predicted prices with all features |
| `output/predictions_1month.png` … `predictions_6month.png` | Prediction plots for each horizon |
| `output/feature_importance.csv` | Feature importance rankings |
| `output/feature_importance.png` | Feature importance bar chart |
| `output/model_comparison.png` | Model performance comparison across horizons |

---

### Option B: Improved Model — `price_prediction_models_improved.py`

An enhanced version with more sophisticated feature engineering, outlier handling, feature scaling (RobustScaler), feature selection (SelectKBest), and optionally CSA-optimized ensemble weights.

**What it does:**
1. Loads daily CPO prices, daily sentiment, and HMM states
2. Calculates technical indicators: RSI, MACD, moving averages (3, 6, 12 period SMA and EMA), momentum
3. Handles outliers using IQR winsorization
4. Scales features with RobustScaler (less sensitive to outliers than StandardScaler)
5. Selects top-K features using F-statistic
6. Trains XGBoost, Random Forest, and Gradient Boosting with RandomizedSearchCV
7. If `csa_optimized_params.json` exists in `output/`, it loads those hyperparameters instead of running search
8. Produces a weighted ensemble

**Inputs required:**
- `cpo/Data_CPO_Daily.csv`
- `news/output/sentiment_aggregate_Daily.csv` *(capital D)*
- `markov/output/hmm_states_results_daily.csv`

```bash
cd d:\Skripsi1\prediction
python price_prediction_models_improved.py
```

**Outputs:**
| File | Description |
|------|-------------|
| `output/prediction_results_improved.csv` | Evaluation metrics per horizon |
| `output/csa_optimized_params.json` | Saved optimal hyperparameters (if CSA was run) |

---

### Option C: CSA-Optimized Model — `price_prediction_models_csa.py` *(Recommended)*

The most advanced and accurate model. Uses the **Crow Search Algorithm (CSA)** — a nature-inspired metaheuristic optimization — to find the best hyperparameters for XGBoost and Random Forest, replacing grid search with an intelligent search strategy.

**What CSA does:**
- Maintains a population of 20–25 "crows", each representing a set of hyperparameters
- Each crow remembers its personal best position (hyperparameter configuration)
- Crows share information: a crow may follow another toward its best-known position
- A flight length (`flight_length = 2.0`) controls exploration vs exploitation
- An awareness probability (`awareness_prob = 0.1`) determines when a crow explores randomly instead of following
- After 50 iterations, the algorithm converges to near-optimal hyperparameters

**Inputs required:**
- `cpo/Data_CPO_Monthly.csv`
- `news/output/monthly_sentiment_aggregate.csv`
- `markov/output/hmm_states_results_daily.csv`

```bash
cd d:\Skripsi1\prediction
python price_prediction_models_csa.py
```

**Outputs:**
| File | Description |
|------|-------------|
| `output/prediction_results_csa.csv` | Evaluation metrics per model per horizon |
| `output/csa_convergence_1m.png` … `csa_convergence_6m.png` | CSA optimization convergence curves |

---

### Option D: Standalone CSA Optimizer — `run_csa_optimization.py`

Runs CSA optimization as a standalone step with full control over parameters, then saves the optimal hyperparameters. These saved parameters can be loaded by `price_prediction_models_improved.py` to skip re-optimization on subsequent runs.

**Command-line arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | `1` | Prediction horizon in months (1–6) |
| `--model` | `both` | `xgboost`, `random_forest`, or `both` |
| `--population` | `25` | Number of crows (larger = better but slower) |
| `--iterations` | `50` | CSA iterations (more = better convergence) |
| `--metric` | `weighted` | Optimization metric: `rmse`, `mae`, `r2`, `weighted` |
| `--cv-folds` | `3` | Cross-validation folds |
| `--output-dir` | `output/csa_results` | Where to save results |
| `--train-end-date` | *(all data)* | Last training date, format `YYYY-MM-DD` |
| `--random-seed` | `42` | Reproducibility seed |

**Example — optimize for 3-month horizon:**
```bash
cd d:\Skripsi1\prediction
python run_csa_optimization.py --horizon 3 --model both --iterations 100 --population 30 --train-end-date 2024-06-30
```

**Example — optimize all horizons sequentially:**
```bash
cd d:\Skripsi1\prediction
for horizon in 1 2 3 4 5 6; do
    python run_csa_optimization.py --horizon $horizon --model both --iterations 50
done
```

**Outputs in `output/csa_results/`:**
| File | Description |
|------|-------------|
| `csa_best_params_horizon_Xm.json` | Best hyperparameters found for horizon X |
| `csa_optimization_results_horizon_Xm.csv` | Full optimization history |
| `csa_convergence_horizon_Xm.png` | Convergence plot showing fitness improvement |

---

### Comparing Ensemble Methods — `compare_ensemble_methods.py`

After running both `price_prediction_models_improved.py` and `run_csa_optimization.py`, run this to compare simple 50/50 ensemble vs CSA-optimized ensemble weights:

```bash
cd d:\Skripsi1\prediction
python compare_ensemble_methods.py
```

Prints a side-by-side comparison of RMSE, MAE, and directional accuracy for both approaches. No file outputs — results are printed to console.

---

## 9. Understanding the Outputs

### `prediction_results_csa.csv` (main results file)

Each row represents one model × one prediction horizon combination:

| Column | Description |
|--------|-------------|
| `horizon` | Months ahead (1 = next month, 6 = six months from now) |
| `model` | `XGBoost`, `RandomForest`, or `Ensemble` |
| `test_rmse` | Root Mean Squared Error on test set (lower = better) |
| `test_mae` | Mean Absolute Error (lower = better) |
| `test_r2` | R-squared (higher = better, max 1.0) |
| `test_mape` | Mean Absolute Percentage Error (lower = better) |
| `directional_accuracy` | % of times the model correctly predicted price direction (up/down) |

### `predictions_with_features.csv`

Row-by-row actual vs predicted prices with all features. Useful for:
- Plotting custom charts
- Analyzing which periods the model struggled with
- Feature correlation analysis

### Convergence plots (`csa_convergence_*.png`)

Shows how the CSA fitness (error) decreased over 50 iterations. A steep early drop followed by a flat plateau indicates good convergence. If the curve is still declining at iteration 50, increase `--iterations`.

### HMM state visualization (`hmm_states_analysis_daily.png`)

A multi-panel chart showing:
- CPO price timeline colored by market state
- State distribution (how often each state occurs)
- Transition probability matrix heatmap
- Return distributions per state

---

## 10. Choosing a Data Frequency

The pipeline supports three frequencies. The key trade-off is **granularity vs noise**:

| Frequency | Pros | Cons | Best for |
|-----------|------|------|----------|
| **Daily** | More training samples, captures short-term dynamics | Higher noise, more missing sentiment data | Research, short-term signals |
| **Weekly** | Balance between noise and signal | Moderate | Exploratory work |
| **Monthly** | Smoothest signal, matches CPO trading cycles | Fewer training samples | Price forecasting (recommended) |

> **Recommendation:** Use **Monthly** for the prediction models (`price_prediction_models_csa.py` and `price_prediction_models.py`). This matches the natural trading and reporting cycle for CPO futures and gives the most stable predictions.

### How to switch frequency

Change these settings across scripts **consistently**:

**For Monthly predictions:**

| Script | Setting to change |
|--------|------------------|
| `news/finbert_sentiment_analysis_flexible.py` | `AGGREGATION_MODE = 'Monthly'` |
| `markov/cpo_hmm_states.py` | `DATA_FREQUENCY = 'monthly'` |
| `create_prediction_dataset.py` | `DATA_FREQUENCY = 'monthly'` *(if using Step 5)* |

> Note: `price_prediction_models_csa.py` and `price_prediction_models.py` are already hardcoded to monthly data.
> `price_prediction_models_improved.py` uses daily data and daily sentiment.

---

## 11. Full Pipeline Quick Reference

Run these commands in order to go from raw CPO data to predicted prices:

```bash
# ── Step 1: Scrape news ──────────────────────────────────────────────────
cd d:\Skripsi1\news
python scrap_fast.py

# ── Step 2: Preprocess news text ────────────────────────────────────────
python news_preprocessing.py

# ── Step 3: Run FinBERT sentiment analysis ───────────────────────────────
# Edit finbert_sentiment_analysis_flexible.py first:
#   Set AGGREGATION_MODE = 'Monthly'   (for monthly predictions)
python finbert_sentiment_analysis_flexible.py

# ── Step 4: HMM market state analysis ───────────────────────────────────
cd d:\Skripsi1\markov
# Edit cpo_hmm_states.py first:
#   Set DATA_FREQUENCY = 'daily'   (always use daily for HMM)
python cpo_hmm_states.py

# ── Step 5 (optional): Build feature dataset ────────────────────────────
# Only needed for custom model exploration
cd d:\Skripsi1
python create_prediction_dataset.py

# ── Step 6: Predict prices ───────────────────────────────────────────────
cd d:\Skripsi1\prediction

# Option A — Quick baseline:
python price_prediction_models.py

# Option B — Improved model:
python price_prediction_models_improved.py

# Option C — CSA-optimized (recommended):
python price_prediction_models_csa.py

# Option D — Optimize first, then predict:
python run_csa_optimization.py --horizon 1 --model both --iterations 50
python run_csa_optimization.py --horizon 3 --model both --iterations 50
python price_prediction_models_improved.py   # Loads saved params automatically

# ── Compare ensemble methods (optional) ─────────────────────────────────
python compare_ensemble_methods.py
```

---

## 12. Troubleshooting

### "File not found" errors

Check that the previous pipeline step completed successfully and its output is in the correct `output/` subfolder. The most common cause is a mismatch in the `DATA_FREQUENCY` / `AGGREGATION_MODE` settings — e.g., running HMM with `'daily'` but sentiment with `'Monthly'`.

**Quick check — verify all required files exist before running predictions:**
```bash
# For price_prediction_models_csa.py
ls d:/Skripsi1/cpo/Data_CPO_Monthly.csv
ls d:/Skripsi1/news/output/monthly_sentiment_aggregate.csv
ls d:/Skripsi1/markov/output/hmm_states_results_daily.csv
```

### CUDA / GPU errors

- Set `FORCE_CPU = True` in the finbert script to run on CPU (much slower)
- Reduce `BATCH_SIZE` to 4 or 2 if you get out-of-memory errors
- Run `python check_cuda.py` in the `news/` folder to diagnose

### FinBERT model download fails

- Ensure internet access during first run
- The model is cached after the first download in `~/.cache/huggingface/`
- If behind a proxy, set `HTTPS_PROXY` environment variable

### Poor prediction accuracy

1. **Check sentiment quality:** Open `news/output/monthly_sentiment_aggregate.csv` and verify there are sentiment scores for most months in the date range
2. **Check HMM states:** Open `markov/output/hmm_states_results_daily.csv` and verify state labels are present — if all rows show the same state, the HMM may not have converged (try increasing `N_ITERATIONS`)
3. **Increase CSA iterations:** Run `run_csa_optimization.py` with `--iterations 100` or `--population 50` for better hyperparameter search
4. **Check date range coverage:** The CPO price data, sentiment data, and HMM states must all overlap in date. If sentiment data only covers recent years but CPO data goes back to 2010, the model will have very few aligned rows

### HMM produces only 1 state

Increase `MAX_STATES` in `cpo_hmm_states.py` or lower `START_YEAR` to provide more training data. The BIC optimization needs sufficient historical variation to identify multiple distinct regimes.

### CSA convergence curve is flat from iteration 1

The search space may be too narrow. This is normal if you have already run CSA before and the saved parameters are loaded — the model is already near-optimal. Check whether `output/csa_optimized_params.json` exists; if so, the script is using those parameters rather than re-optimizing.

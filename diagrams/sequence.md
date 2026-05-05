# Sequence Diagrams — CPO Price Prediction System

> ⚠️ **OUTDATED — pre-2026-05-05**
>
> The flows below show the pre-cleanup multi-model dispatch
> (RF / ARIMAX / SARIMAX / XGBoost × base / csa / Bayesian). Current
> scope is XGBoost only × {base, csa}. See
> [CLEANUP_INVENTORY.md](../CLEANUP_INVENTORY.md) and
> [ARCHITECTURE.md](../ARCHITECTURE.md). Sequence diagrams will be
> redrawn in a separate task.
>
> The earlier "User Registration" and "User Login and Dashboard Load"
> sections were removed on 2026-05-05 as part of the auth-removal
> cleanup — the site is now public-facing read-only with no login.

Two interaction flows are documented here:
1. [Prediction API Call](#1-prediction-api-call)
2. [Daily Scheduler Pipeline Run](#2-daily-scheduler-pipeline-run)

---

## 1. Prediction API Call

### Diagram

```mermaid
sequenceDiagram
    actor U as User (Browser)
    participant JS as Dashboard JavaScript<br/>(dashboard.html)
    participant DV as Django View<br/>(views.py - prediction_api)
    participant FS as Firestore<br/>(predictions collection)

    U->>JS: Selects Model, Variant, Horizon<br/>Clicks "Get Prediction"

    JS->>JS: Validate: all dropdowns selected

    alt Missing selection
        JS-->>U: Show inline warning:<br/>"Please select all options"
    else All selected
        JS->>JS: Show loading spinner

        JS->>DV: GET /api/prediction/<br/>?model=xgboost&variant=csa<br/>&frequency=Daily&horizon=3

        DV->>DV: Extract query params:<br/>model, variant, frequency, horizon

        DV->>DV: Build document ID:<br/>"xgboost_csa_Daily_h3"

        DV->>FS: predictions.document("xgboost_csa_Daily_h3").get()

        alt Document not found
            FS-->>DV: DocumentSnapshot (exists=False)
            DV-->>JS: JSON {success: false,<br/>error: "No prediction available"}
            JS-->>U: Show error card
        else Document found
            FS-->>DV: DocumentSnapshot with all fields

            DV->>DV: Extract:<br/>predicted_price, predicted_date,<br/>last_actual_price, last_actual_date,<br/>mape, r2, directional_accuracy, computed_at

            DV-->>JS: JSON {<br/>  success: true,<br/>  model: "xgboost",<br/>  variant: "csa",<br/>  horizon: 3,<br/>  predicted_price: 3842.50,<br/>  predicted_date: "2025-04-18",<br/>  last_actual_price: 3820.00,<br/>  mape: 2.14,<br/>  r2: 0.91,<br/>  directional_accuracy: 68.5,<br/>  computed_at: "2025-04-15T06:00:00Z"<br/>}

            JS->>JS: Hide loading spinner

            JS->>U: Update DOM:<br/>Prediction result card<br/>Model info card<br/>Metrics display
        end
    end
```

### Message Descriptions

| # | From | To | Message | Description |
|---|---|---|---|---|
| 1 | User | JS | `selects + clicks` | User interaction with dropdowns and button in prediction control panel |
| 2 | JS | JS | `validate dropdowns` | Checks `select.value !== ""` for all three dropdowns |
| 3 | JS | JS | `show spinner` | CSS class toggled to display loading indicator |
| 4 | JS | Django View | `GET /api/prediction/` | `fetch("/api/prediction/?" + new URLSearchParams({...}))` |
| 5 | Django View | Django View | `extract params` | `request.GET.get("model", "")` for each parameter |
| 6 | Django View | Django View | `build doc ID` | `f"{model}_{variant}_{frequency}_h{horizon}"` |
| 7 | Django View | Firestore | `.document(id).get()` | Single-document read — O(1) latency |
| 8 | Firestore | Django View | `DocumentSnapshot` | `.exists` is `False` if no such document |
| 9 | Django View | JS | `{success: false}` | `JsonResponse({"success": False, "error": "..."})`; HTTP 200 |
| 10 | Django View | Django View | `extract fields` | `doc.to_dict()` then pick required keys |
| 11 | Django View | JS | `{success: true, ...}` | `JsonResponse({...})` with all prediction fields |
| 12 | JS | JS | `hide spinner` | CSS class removed |
| 13 | JS | User | `update DOM` | `document.getElementById("predicted-price").textContent = ...` |

**Key Files:** `website/web/views.py` → `prediction_api()`, `website/web/templates/dashboard.html` (JavaScript section)

---

## 2. Daily Scheduler Pipeline Run

### Diagram

```mermaid
sequenceDiagram
    participant GCS as Google Cloud Scheduler<br/>(cron trigger)
    participant CR as Cloud Run Job<br/>(scheduler/main.py)
    participant API as External APIs<br/>(Investing.com / MPOB)
    participant ML as ML Models<br/>(prediction/*)
    participant FS as Firestore<br/>(all collections)

    GCS->>CR: HTTP POST trigger<br/>(daily cron)

    CR->>CR: Parse --mode daily<br/>initialise Firestore client

    Note over CR,FS: Step 1 — Prices

    CR->>FS: daily_prices — get latest date
    FS-->>CR: Last stored date

    CR->>API: Fetch CPO prices since last date<br/>(Investing.com API)
    API-->>CR: New OHLCV rows

    CR->>FS: Batch write new daily_prices<br/>documents (doc ID = date)
    FS-->>CR: Batch commit ACK

    Note over CR,FS: Step 2 — News

    CR->>API: Scrape MPOB news website<br/>(BeautifulSoup / requests)
    API-->>CR: HTML article list

    CR->>CR: Parse articles<br/>deduplicate by MD5(url)

    CR->>FS: Batch write new news_articles<br/>documents (doc ID = MD5)
    FS-->>CR: Batch commit ACK

    Note over CR,ML: Step 3 — Sentiment

    CR->>FS: Query news_articles<br/>where sentiment_label == null
    FS-->>CR: Unlabelled articles

    CR->>ML: Run FinBERT inference<br/>(batched, GPU)
    ML-->>CR: {sentiment_label, sentiment_score} per article

    CR->>FS: Update news_articles with labels
    CR->>FS: Batch write sentiment_aggregates<br/>(grouped by date)
    FS-->>CR: Batch commit ACK

    Note over CR,ML: Step 4 — HMM States

    CR->>FS: Load all daily_prices<br/>(full history)
    FS-->>CR: All price documents

    CR->>ML: Compute log-returns<br/>Fit GaussianHMM (BIC selection)<br/>Label states
    ML-->>CR: {date → state} mapping

    CR->>FS: Batch write all hmm_states<br/>(doc ID = Daily_YYYY-MM-DD)
    FS-->>CR: Batch commit ACK

    Note over CR,ML: Step 5 — Predictions (56 combinations)

    CR->>FS: Load prices + HMM + sentiment<br/>for feature engineering
    FS-->>CR: All required documents

    CR->>CR: Build prediction feature dataset<br/>(create_prediction_dataset.py)

    loop For each of 56 model combinations
        CR->>ML: Train / load model<br/>(XGBoost / RF / ARIMAX / SARIMAX)
        CR->>ML: Run hyperparameter optimisation<br/>(CSA or Bayesian if not base variant)
        ML-->>CR: Optimised model

        CR->>ML: Generate h-step ahead prediction
        ML-->>CR: predicted_price, predicted_date

        CR->>ML: Evaluate on test split
        ML-->>CR: mape, r2, rmse, directional_accuracy

        CR->>FS: Write predictions document<br/>(doc ID = model_variant_freq_hN)
        FS-->>CR: Write ACK
    end

    CR-->>GCS: Pipeline complete<br/>HTTP 200 response
```

### Message Descriptions

| # | From | To | Message | Description |
|---|---|---|---|---|
| 1 | Cloud Scheduler | Cloud Run | `HTTP POST trigger` | GCP Cloud Scheduler fires the daily job |
| 2 | Cloud Run | Cloud Run | `parse args + init` | `argparse` reads `--mode daily`; Firestore client created with service account |
| 3 | Cloud Run | Firestore | `get latest price date` | Finds the most recent `daily_prices` document date to determine what's missing |
| 4 | Cloud Run | Investing.com API | `fetch prices` | HTTP request for CPO futures OHLCV data since last stored date |
| 5 | Cloud Run | Firestore | `batch write daily_prices` | Uses `scheduler/firestore_writer.py`; 500-doc batch limit enforced |
| 6 | Cloud Run | MPOB website | `scrape news` | `requests.get()` + BeautifulSoup; multi-threaded via `ThreadPoolExecutor` |
| 7 | Cloud Run | Cloud Run | `deduplicate` | MD5 hash of URL compared against already-stored document IDs |
| 8 | Cloud Run | Firestore | `batch write news_articles` | Document ID = `md5(url)` ensures idempotency |
| 9 | Cloud Run | Firestore | `query unlabelled articles` | `news_articles.where("sentiment_label", "==", None)` |
| 10 | Cloud Run | FinBERT (ML) | `run inference` | `AutoModelForSequenceClassification` from HuggingFace; GPU-batched |
| 11 | Cloud Run | Firestore | `update articles + aggregates` | Article-level labels written; then per-date aggregates computed and written |
| 12 | Cloud Run | Firestore | `load all prices` | Full history needed for GaussianHMM fitting (stateful model) |
| 13 | Cloud Run | HMM (ML) | `fit GaussianHMM` | `hmmlearn.GaussianHMM`; BIC criterion tests 2–5 states |
| 14 | Cloud Run | Firestore | `batch write hmm_states` | All states rewritten (entire history) due to HMM re-labelling |
| 15 | Cloud Run | Firestore | `load feature data` | Prices + HMM + sentiment all loaded for `create_prediction_dataset.py` |
| 16 | Cloud Run | Cloud Run | `build feature dataset` | Merges three sources; engineers 60+ lag, return, cyclical, and indicator features |
| 17 | Cloud Run | ML Models | `train/load model` | XGBoost/RF: load from GCS cache or retrain; ARIMAX/SARIMAX: always refit |
| 18 | Cloud Run | ML Models | `hyperparameter opt` | CSA (`prediction/csa_hyperparameter_optimizer.py`) or Bayesian (`prediction/bayesian_optimizer.py`) |
| 19 | Cloud Run | ML Models | `generate prediction` | Single h-step-ahead forecast for the next `horizon` trading days |
| 20 | Cloud Run | ML Models | `evaluate on test split` | 15% most-recent data held out; MAPE, R², RMSE, directional accuracy computed |
| 21 | Cloud Run | Firestore | `write predictions doc` | `predictions.document(f"{model}_{variant}_{freq}_h{h}").set({...})` |
| 22 | Cloud Run | Cloud Scheduler | `200 OK` | Job completion signal; Cloud Scheduler records success |

**Key Files:** `scheduler/main.py`, `scheduler/price_fetcher.py`, `scheduler/news_extractor.py`, `scheduler/sentiment_runner.py`, `scheduler/hmm_updater.py`, `scheduler/prediction_updater.py`, `scheduler/firestore_writer.py`, `prediction/horizon_forecast.py`, `prediction/bayesian_optimizer.py`, `prediction/csa_hyperparameter_optimizer.py`, `markov/cpo_hmm_states.py`, `news/finbert_sentiment_analysis_flexible.py`

# Sequence Diagrams — CPO Price Prediction System

Four interaction flows are documented here:
1. [User Registration](#1-user-registration)
2. [User Login and Dashboard Load](#2-user-login-and-dashboard-load)
3. [Prediction API Call](#3-prediction-api-call)
4. [Daily Scheduler Pipeline Run](#4-daily-scheduler-pipeline-run)

---

## 1. User Registration

### Diagram

```mermaid
sequenceDiagram
    actor U as User (Browser)
    participant DV as Django View<br/>(views.py)
    participant FS as Firestore<br/>(users collection)

    U->>DV: GET /register/
    DV-->>U: Render register.html (empty form)

    U->>DV: POST /register/<br/>username, email, password

    DV->>DV: Validate: password length ≥ 8

    alt Password too short
        DV-->>U: Re-render form with error:<br/>"Password must be at least 8 characters"
    end

    DV->>FS: Query where email == submitted_email

    alt Email already registered
        FS-->>DV: Returns existing document
        DV-->>U: Re-render form with error:<br/>"Email already registered"
    else Email available
        FS-->>DV: Returns empty result

        DV->>DV: make_password(password)<br/>→ PBKDF2-SHA256 hash

        DV->>FS: .collection("users").add({<br/>  uid, username, email,<br/>  password_hash, created_at,<br/>  is_active: true<br/>})

        FS-->>DV: Document reference (auto UID)

        DV-->>U: HTTP 302 redirect to /login/
    end
```

### Message Descriptions

| # | From | To | Message | Description |
|---|---|---|---|---|
| 1 | User | Django View | `GET /register/` | User navigates to registration page |
| 2 | Django View | User | `render register.html` | Empty form rendered from `website/web/templates/register.html` |
| 3 | User | Django View | `POST /register/` | Form submission with `username`, `email`, `password` fields |
| 4 | Django View | Django View | `validate password length` | Local check: `len(password) < 8` → error |
| 5 | Django View | Firestore | `query email` | `users.where("email", "==", email.lower()).limit(1)` |
| 6 | Firestore | Django View | `existing doc` | If a document is returned, email is taken |
| 7 | Django View | User | `re-render with error` | Template re-rendered with `{"error": "Email already registered"}` context |
| 8 | Django View | Django View | `make_password()` | Django's `hashers.make_password()` → PBKDF2-SHA256 |
| 9 | Django View | Firestore | `.add({...})` | New `users` document with all required fields |
| 10 | Firestore | Django View | `document reference` | Returns the auto-generated UID |
| 11 | Django View | User | `302 /login/` | Redirect to login page |

**Key Files:** `website/web/views.py` → `register_view()`, `website/web/auth_backend.py`

---

## 2. User Login and Dashboard Load

### Diagram

```mermaid
sequenceDiagram
    actor U as User (Browser)
    participant MW as Auth Middleware<br/>(auth_backend.py)
    participant DV as Django View<br/>(views.py)
    participant FS as Firestore<br/>(multiple collections)
    participant TM as Template Engine<br/>(Django + Jinja2)

    U->>DV: POST /login/<br/>email, password

    DV->>FS: Query users where email == submitted_email

    alt User not found
        FS-->>DV: Empty result
        DV-->>U: Re-render login.html with error:<br/>"Invalid credentials"
    else User found
        FS-->>DV: User document {uid, username, password_hash, is_active}

        DV->>DV: check_password(submitted, hash)

        alt Password mismatch
            DV-->>U: Re-render login.html with error
        else Password correct
            DV->>DV: Write _uid, _username, _email<br/>to signed session cookie

            DV->>FS: Update users/{uid}<br/>last_login = now()

            DV-->>U: HTTP 302 redirect to /dashboard/
        end
    end

    U->>MW: GET /dashboard/<br/>(with session cookie)

    MW->>MW: Decode signed cookie<br/>extract _uid, _username, _email

    MW->>MW: Check in-memory UID cache

    alt UID not in cache
        MW->>FS: users/{uid}.get()
        FS-->>MW: User document
        MW->>MW: Add UID to in-memory cache
    end

    MW->>MW: Build FirestoreUser object<br/>set request.user

    MW->>DV: Pass request to dashboard()

    DV->>FS: daily_prices<br/>order_by("date", desc).limit(90)
    FS-->>DV: 90 price documents

    DV->>FS: hmm_states<br/>where frequency==Daily<br/>where date in [date_list]
    FS-->>DV: HMM state documents

    DV->>DV: Compute statistics:<br/>current, max, min, avg price

    DV->>DV: Serialize prices + HMM states<br/>to JSON arrays

    DV->>TM: render("dashboard.html", context)
    TM-->>U: Full HTML page with inline JSON

    U->>U: Chart.js parses inline JSON<br/>draws price line chart

    U->>U: Annotation plugin draws<br/>HMM state colored bands
```

### Message Descriptions

| # | From | To | Message | Description |
|---|---|---|---|---|
| 1 | User | Django View | `POST /login/` | Email + password submitted |
| 2 | Django View | Firestore | `query users by email` | Exact-match query on lowercased email |
| 3 | Firestore | Django View | `user document` | Returns dict with `password_hash`, `uid`, `is_active` |
| 4 | Django View | Django View | `check_password()` | Django's `hashers.check_password()` |
| 5 | Django View | Django View | `write session cookie` | `request.session["_uid"] = uid` etc; Django signs the cookie |
| 6 | Django View | Firestore | `update last_login` | `users.document(uid).update({"last_login": now})` |
| 7 | Django View | User | `302 /dashboard/` | Redirect with `Set-Cookie` header |
| 8 | User | Auth Middleware | `GET /dashboard/` | Browser sends the session cookie |
| 9 | Auth Middleware | Auth Middleware | `decode cookie` | `request.session.get("_uid")` |
| 10 | Auth Middleware | Auth Middleware | `check UID cache` | Module-level dict `{uid: FirestoreUser}` |
| 11 | Auth Middleware | Firestore | `users/{uid}.get()` | Only on cache miss |
| 12 | Auth Middleware | Auth Middleware | `set request.user` | Attaches `FirestoreUser` instance to request |
| 13 | Auth Middleware | Django View | `pass request` | Middleware chain continues to view |
| 14 | Django View | Firestore | `daily_prices query` | `.order_by("date", direction=DESCENDING).limit(90)` |
| 15 | Django View | Firestore | `hmm_states query` | `.where("date", "in", date_list)` |
| 16 | Django View | Django View | `compute stats` | Simple Python `max()`, `min()`, `sum()/len()` |
| 17 | Django View | Django View | `serialize to JSON` | `json.dumps({"dates": [...], "prices": [...], "states": {...}})` |
| 18 | Django View | Template Engine | `render dashboard.html` | Django `render(request, "dashboard.html", context)` |
| 19 | Template Engine | User | `HTML with inline JSON` | `{{ chart_data\|safe }}` outputs the JSON directly into a `<script>` tag |
| 20 | User (browser) | User (browser) | `Chart.js init` | JavaScript parses the JSON and instantiates a Chart.js line chart |
| 21 | User (browser) | User (browser) | `draw HMM bands` | Annotation plugin groups consecutive same-state dates into background rectangles |

**Key Files:** `website/web/views.py`, `website/web/auth_backend.py`, `website/web/templates/dashboard.html`

---

## 3. Prediction API Call

### Diagram

```mermaid
sequenceDiagram
    actor U as User (Browser)
    participant JS as Dashboard JavaScript<br/>(dashboard.html)
    participant MW as Auth Middleware<br/>(auth_backend.py)
    participant DV as Django View<br/>(views.py - prediction_api)
    participant FS as Firestore<br/>(predictions collection)

    U->>JS: Selects Model, Variant, Horizon<br/>Clicks "Get Prediction"

    JS->>JS: Validate: all dropdowns selected

    alt Missing selection
        JS-->>U: Show inline warning:<br/>"Please select all options"
    else All selected
        JS->>JS: Show loading spinner

        JS->>DV: GET /api/prediction/<br/>?model=xgboost&variant=csa<br/>&frequency=Daily&horizon=3

        MW->>MW: Verify session cookie<br/>set request.user

        alt Unauthenticated
            DV-->>JS: HTTP 302 redirect to /login/
        else Authenticated
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
    end
```

### Message Descriptions

| # | From | To | Message | Description |
|---|---|---|---|---|
| 1 | User | JS | `selects + clicks` | User interaction with dropdowns and button in prediction control panel |
| 2 | JS | JS | `validate dropdowns` | Checks `select.value !== ""` for all three dropdowns |
| 3 | JS | JS | `show spinner` | CSS class toggled to display loading indicator |
| 4 | JS | Django View | `GET /api/prediction/` | `fetch("/api/prediction/?" + new URLSearchParams({...}))` |
| 5 | Auth Middleware | Auth Middleware | `verify cookie` | `FirestoreAuthMiddleware` populates `request.user` before view runs |
| 6 | Auth Middleware | JS | `302 redirect` | If cookie invalid/missing; JS `fetch()` follows redirect to login page |
| 7 | Django View | Django View | `extract params` | `request.GET.get("model", "")` for each parameter |
| 8 | Django View | Django View | `build doc ID` | `f"{model}_{variant}_{frequency}_h{horizon}"` |
| 9 | Django View | Firestore | `.document(id).get()` | Single-document read — O(1) latency |
| 10 | Firestore | Django View | `DocumentSnapshot` | `.exists` is `False` if no such document |
| 11 | Django View | JS | `{success: false}` | `JsonResponse({"success": False, "error": "..."})`; HTTP 200 |
| 12 | Django View | Django View | `extract fields` | `doc.to_dict()` then pick required keys |
| 13 | Django View | JS | `{success: true, ...}` | `JsonResponse({...})` with all prediction fields |
| 14 | JS | JS | `hide spinner` | CSS class removed |
| 15 | JS | User | `update DOM` | `document.getElementById("predicted-price").textContent = ...` |

**Key Files:** `website/web/views.py` → `prediction_api()`, `website/web/templates/dashboard.html` (JavaScript section)

---

## 4. Daily Scheduler Pipeline Run

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

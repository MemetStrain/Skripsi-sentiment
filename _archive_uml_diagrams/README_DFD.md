# Data Flow Diagrams (DFD)
## CPO Price Prediction System — Web Application

These diagrams describe the **runtime data flow** of the deployed website
(Django on Vercel + Firestore + bundled XGBoost / HMM artefacts) plus the
**local scheduler** the maintainer runs on their workstation to refresh
Firestore. They reflect the architecture as of the 2026-04-26
thesis-scope-reduction sweep and the 2026-05-05 auth removal:

- XGBoost only (no Random Forest / ARIMAX / SARIMAX in production)
- Daily horizon only (no Weekly / Monthly)
- C1–C4 ablation (price-only / +HMM / +sentiment / full)
- Public read-only frontend (no users, no admin role)
- Local scheduler (no Cloud Run / Docker)

## Files in this folder

| File | What it is |
|---|---|
| [dfd_level_0_context.html](dfd_level_0_context.html) | Level 0 — Context Diagram. The whole system as a single process; four external entities. |
| [dfd_level_1_detailed.html](dfd_level_1_detailed.html) | Level 1 — Decomposed into three internal processes (Web Frontend, Inference Engine, Local Scheduler) with all eight data stores. |
| [dfd_level_2_inference.html](dfd_level_2_inference.html) | Level 2 — Decomposition of Process 2.0 (the live inference engine behind `/api/forecasts/`). |
| README_DFD.md | This file. |

Open the HTML files directly in any modern browser — Mermaid 10 and Tailwind
are loaded via CDN, no build step is needed.

## Diagram hierarchy

```
DFD Level 0 (Context)
└── 0.0 CPO Prediction System
    ├── ↔ Web Visitor          (HTTP requests / HTML + JSON)
    ├── ↔ Maintainer           (CLI invocation / log output)
    ├── ←  MPOB Website        (article HTML)
    └── ←  Investing.com       (OHLCV)

DFD Level 1 (Detailed)
├── Processes
│   ├── 1.0 Web Frontend         (Django on Vercel)
│   ├── 2.0 Live Inference       (predictor.py)
│   └── 3.0 Local Scheduler      (scheduler/main.py)
└── Data stores
    ├── Firestore
    │   ├── D1 daily_prices
    │   ├── D2 hmm_states
    │   ├── D3 sentiment_aggregates
    │   ├── D4 news_articles
    │   └── D5 hmm_models (frozen params)
    ├── Bundled with deploy
    │   ├── D6 prediction/winners.json
    │   └── D7 prediction/saved_models/{tag}/Daily/h{1..7}/xgboost_csa/
    └── Workstation only
        └── D8 cpo + news CSVs (source of truth)

DFD Level 2 (Inference — Process 2.0 expanded)
├── 2.1 Load Winners              (cached read of D6)
├── 2.2 Build Inference Frame     (D1 + D2 + D3 → engineered DataFrame)
├── 2.3 Load Models               (cached read of D7)
├── 2.4 Compute Rolling Trails    (per-horizon prediction loop)
└── 2.5 Format JSON Response
```

## External entities

| Entity | Role | Direction | How it's reached |
|---|---|---|---|
| **Web Visitor** | Anonymous user of the public dashboard | Two-way HTTP | Browser → Vercel-hosted Django app |
| **Maintainer** | Operator who refreshes data ad-hoc | Two-way CLI | Local Python invocation of `scheduler/main.py` |
| **MPOB Website** | News article source | Inbound (scraped) | `requests` + `beautifulsoup4` from the scheduler |
| **Investing.com** | Daily OHLCV source | Inbound (API) | `investiny` library from the scheduler |

There is **no admin role** inside the web app — Django auth was removed
on 2026-05-05. The maintainer interacts with the system only via the
scheduler script, which runs outside the deployed runtime.

## Processes

### 1.0 Web Frontend — `website/web/`
Django 6 application served by `vercel_wsgi.py`. Three pages and one JSON
endpoint:

| Route | View | Reads | Writes |
|---|---|---|---|
| `GET /` | `dashboard` | D1, D2, D3, D6 | — |
| `GET /news/` | `news` | D4 | — |
| `GET /about/` | `about` | (none) | — |
| `GET /api/forecasts/` | `forecasts_api` | delegates to 2.0 | — |

The dashboard view fetches a 90-day window from Firestore and embeds the
data as JSON in the rendered template; Chart.js draws everything client-side.
The browser then makes a follow-up call to `/api/forecasts/` for the
rolling-trail overlay.

### 2.0 Live Inference Engine — `website/web/predictor.py`
Re-engineers the full feature frame from Firestore on every request and
runs the winning CSA XGBoost model per horizon. Module-level `lru_cache`
keeps `winners.json` and the seven model artefacts in memory across
requests in a warm Vercel function. Feature engineering is delegated to
`prediction/feature_engineering.py` — the same module the offline C1–C4
training scripts use, so production inference and offline training share
exactly one feature pipeline.

See [dfd_level_2_inference.html](dfd_level_2_inference.html) for the
sub-process breakdown.

### 3.0 Local Scheduler — `scheduler/main.py`
Three modes:

- **`--mode initial`** — bulk-mirror local CSVs (D8) into Firestore. Each
  step is checkpointed in `scheduler/initial_load_progress.json`.
- **`--mode daily`** — incremental: fetch the latest price, scrape new
  MPOB articles, score them with FinBERT-Tone, recompute aggregates,
  decode HMM states with frozen parameters. Writes D1, D2, D3, D4.
- **`--mode rebuild-hmm`** — re-publish the offline-trained HMM
  parameters (D5) and rewrite D2 from scratch. Used after re-running
  `markov/cpo_hmm_states.py`.

Predictions are **not** written by the scheduler. They are produced live
by 2.0 on demand.

## Data stores

| ID | Name | Backend | Doc / row key | Written by | Read by |
|---|---|---|---|---|---|
| D1 | `daily_prices` | Firestore | `YYYY-MM-DD` | 3.0 | 1.0, 2.0 |
| D2 | `hmm_states` | Firestore | `Daily_YYYY-MM-DD` | 3.0 | 1.0, 2.0 |
| D3 | `sentiment_aggregates` | Firestore | `Daily_YYYY-MM-DD` | 3.0 | 1.0, 2.0 |
| D4 | `news_articles` | Firestore | `md5(url)` | 3.0 | 1.0 |
| D5 | `hmm_models/Daily` | Firestore document | Frozen GaussianHMM params + 252-day z-score normaliser | 3.0 (`rebuild-hmm` only) | 3.0 (every daily run) |
| D6 | `prediction/winners.json` | Local file (bundled in deploy) | Single JSON object | Offline tool: `compute_winners.py` | 1.0, 2.0 |
| D7 | `prediction/saved_models/{tag}/Daily/h{1..7}/xgboost_csa/` | Local folder (bundled in deploy) | `model.pkl` · `scaler.pkl` · `meta.json` | Offline training: `horizon_forecast_C{1..4}_*.py` | 2.0 |
| D8 | Local CSVs in `cpo/` and `news/` | Workstation files | One row per date / article | 3.0 (append-only) | 3.0 |

D6 and D7 are bundled into the Vercel deploy at build time so that the
inference engine can read them without a network round-trip. D8 lives on
the maintainer's workstation only and is never deployed.

## Notable data flows

### A. Dashboard page load
1. Visitor `GET /`
2. Frontend reads D1 (90-day OHLCV), D2 (state labels), D3 (aggregates), D6 (metrics table data)
3. Renders `dashboard.html` with everything embedded as JSON
4. Browser then fires `GET /api/forecasts/?max_horizon=7&window_days=90`

### B. Forecast API call (handled by Process 2.0)
1. Visitor `GET /api/forecasts/?max_horizon=7&window_days=90`
2. Frontend invokes `compute_forecast_trails(db, 7, 90)`
3. Inference engine reads D1, D2, D3, D6, D7 (cached after first call), produces trail JSON
4. Frontend wraps it in `JsonResponse` and returns

### C. News page
1. Visitor `GET /news/?sentiment=Positive&page=2`
2. Frontend streams **all** D4 documents, sorts and filters in Python (avoids a composite index)
3. Paginates 9 cards per page

### D. Daily scheduler run
1. Maintainer `python scheduler/main.py --mode daily`
2. Scheduler reads D8 (latest local date)
3. If stale → fetch from Investing.com, scrape MPOB → append to D8 → mirror to D1, D4
4. Recompute aggregates from D8 → write D3
5. Read D5 (frozen params) → forward-filter HMM decoding → write D2

## DFD notation

| Symbol | Represents | Examples |
|---|---|---|
| Yellow rounded rectangle | External entity | Web Visitor, MPOB |
| Teal circle | Process | 1.0 Web Frontend, 2.4 Compute Rolling Trails |
| Blue cylinder | Data store | D1 daily_prices, D7 saved_models |
| Solid arrow | Command / HTTP message / write | `GET /` request, scheduler upsert |
| Dashed arrow | Read (often cached) | `lru_cache`-backed `joblib.load` |

## Technology stack (reference)

- **Frontend:** Django 6, Chart.js 4, chartjs-plugin-annotation, Tailwind CSS (CDN)
- **Inference:** numpy, pandas, scikit-learn (RobustScaler), xgboost ≥ 2.0, joblib
- **Database:** Google Cloud Firestore via `firebase-admin`
- **Scheduler:** `investiny` (price), `requests` + `beautifulsoup4` (news), `transformers` + `torch` + `nltk` (FinBERT-Tone), `hmmlearn` (HMM)
- **Deployment:** Vercel (Python 3.11 serverless, `maxLambdaSize: 50mb`); scheduler is local-only

## Out of scope

These DFDs describe the **production / serving** path. They deliberately
omit:

- Offline training pipelines (`prediction/horizon_forecast_C{1..4}_*.py`)
- Offline HMM fitting (`markov/cpo_hmm_states.py`)
- Sentiment-weight grid-search experiments (archived)
- The Diebold-Mariano comparison framework (offline tooling)

Those produce the artefacts that land in D5, D6, and D7 — but they run on
the maintainer's machine, never as part of a user request.

## Metadata

| Field | Value |
|---|---|
| Diagram type | Data Flow Diagram (DFD), three levels |
| System | CPO Price Prediction Web Application |
| Methodology | Structured Systems Analysis & Design (Yourdon/DeMarco) |
| Tooling | Mermaid 10 (flowchart syntax) + Tailwind CSS, both via CDN |
| Author | Matthew / ExMatter, Universitas Bina Nusantara |
| Project | Skripsi (thesis) |

# Data Flow Diagram (DFD)
## Sistem Web Prediksi Harga CPO dengan FinBERT & HMM

> ⚠️ **OUTDATED — pre-2026-04-26**
>
> Mentions Random Forest as a production model. After the
> thesis-scope-reduction sweep the production scope is XGBoost only
> with ablation study (C1-C4). See
> [CLEANUP_INVENTORY.md](../CLEANUP_INVENTORY.md) and
> [ARCHITECTURE.md](../ARCHITECTURE.md).

### 📁 File Diagram

1. **dfd_level_0_context.html** - Context Diagram (DFD Level 0)
2. **dfd_level_1_detailed.html** - Detailed Process View (DFD Level 1)

### 🚀 Cara Menggunakan

Buka file HTML di browser:
```bash
# Windows
start dfd_level_0_context.html
start dfd_level_1_detailed.html

# Atau double-click file HTML di Windows Explorer
```

### 📊 Hierarki Diagram

```
DFD Level 0 (Context Diagram)
└── Menampilkan sistem sebagai single process
    └── Interaksi dengan 3 entitas eksternal:
        ├── User (Pengguna Umum)
        ├── Admin
        └── MPOB (Malaysian Palm Oil Board)

DFD Level 1 (Detailed Process)
└── Memecah sistem menjadi 4 sub-proses:
    ├── P1: Authentication
    ├── P2: News Automation Pipeline
    ├── P3: Price & Market State Management
    └── P4: Prediction Inference
└── 3 Data Stores:
    ├── D1: Users DB
    ├── D2: News_Sentiment DB
    └── D3: Price_MarketState DB
```

### 🎯 Komponen Sistem

#### **Entitas Eksternal**
| Entitas | Peran | Interaksi |
|---------|-------|-----------|
| **User** | Pengguna umum | Login + request prediksi → menerima hasil prediksi |
| **Admin** | Administrator | Login + input data harga → kelola data price |
| **MPOB** | Sumber berita | Menyediakan raw news data (otomatis di-scrape) |

#### **Proses Utama (Level 1)**

**P1: Authentication**
- Input: Credentials (username, password)
- Proses: Validasi dengan Users DB
- Output: Session token / access granted
- Teknologi: JWT, session management

**P2: News Automation Pipeline**
- Input: Raw news dari MPOB (scraping otomatis)
- Proses: FinBERT Sentiment Analysis
- Output: Sentiment scores (positive/negative/neutral)
- Jadwal: Background scheduler (harian)

**P3: Price & Market State Management**
- Input: Data OHLC (Open/High/Low/Close) dari Admin
- Proses: Hidden Markov Model (HMM) untuk hidden states
- Output: Update Price_MarketState DB dengan states (Bullish/Bearish)
- Akses: Admin only

**P4: Prediction Inference**
- Input: Request dari User + data dari D2 & D3
- Proses: Pre-trained ML Model (Random Forest/XGBoost)
- Output: Prediksi harga t+1, metrik (MAPE, R²), visualisasi
- Mode: Real-time inference

#### **Data Stores**

**D1: Users DB**
- Kredensial login (user_id, username, password_hash, role)
- Diakses oleh: P1 (Authentication)

**D2: News_Sentiment DB**
- Berita MPOB + hasil analisis sentimen FinBERT
- Struktur: date, news_text, sentiment_score, sentiment_label
- Update: Otomatis via P2 (harian)
- Digunakan: P4 untuk fitur prediksi

**D3: Price_MarketState DB**
- Harga historis CPO + HMM hidden states
- Struktur: date, open, high, low, close, volume, state, lagged_features
- Update: Manual via Admin (P3)
- Digunakan: P4 untuk fitur prediksi

### 🔄 Aliran Data Kritis

1. **User Authentication Flow**
   ```
   User → [Credentials] → P1 ↔ D1 → [Token] → User
   ```

2. **News Processing Flow (Background)**
   ```
   MPOB → [Raw News] → P2 → [Sentiment Scores] → D2
   ```

3. **Price Update Flow (Admin)**
   ```
   Admin → [OHLC Data] → P3 ↔ D3 (HMM Calculation)
   ```

4. **Prediction Flow (User Request)**
   ```
   User → [Request] → P4
                       ↑
                       ├── D2 (Sentiment Data)
                       └── D3 (Price & States)
                       ↓
   User ← [Prediction + Charts]
   ```

### 📝 Notasi DFD

| Simbol | Representasi | Contoh |
|--------|-------------|--------|
| 🟦 Persegi Panjang | Entitas Eksternal | User, Admin, MPOB |
| ⭕ Lingkaran | Proses | 1.0 Authentication |
| 💾 Silinder | Data Store | D1: Users DB |
| → Panah | Aliran Data | Credentials, Token |
| - - → Panah Putus | Read Historical Data | D3 ke P3 |

### 🛠️ Teknologi Stack (Referensi)

- **Frontend**: HTML/CSS/JavaScript + Chart.js
- **Backend**: Python Flask/FastAPI
- **ML Models**: FinBERT (Hugging Face), Scikit-learn HMM, Random Forest
- **Database**: PostgreSQL / MySQL
- **Scheduler**: APScheduler / Celery
- **Authentication**: JWT / OAuth2

### 📌 Catatan Penting

1. **Fase Inference**: DFD ini fokus pada fase production/inference, bukan training model
2. **Pre-trained Models**: Model FinBERT dan model prediksi sudah dilatih sebelumnya
3. **Automated Pipeline**: P2 berjalan otomatis di background (tidak memerlukan interaksi user)
4. **Role-Based Access**: P3 hanya bisa diakses oleh Admin
5. **Real-time Prediction**: P4 memberikan hasil prediksi secara real-time saat user request

### 📧 Metadata

- **Diagram Type**: Data Flow Diagram (DFD)
- **System**: Web Application - CPO Price Prediction
- **Methodology**: Structured System Analysis & Design
- **Version**: 1.0
- **Date**: December 2025
- **Role**: Senior System Analyst & Software Architect

---

**Generated by**: System Analysis Documentation Tool  
**Purpose**: Skripsi/Thesis - Sistem Prediksi Harga CPO

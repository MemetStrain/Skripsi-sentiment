# User Flowchart - CPO Price Prediction System

## Mermaid.js Activity Diagram

```mermaid
flowchart TD
    Start([🌐 Buka Website]) --> CheckAccount{Sudah Punya Akun?}
    
    %% Registration Flow
    CheckAccount -->|Tidak| Register[📝 Halaman Register]
    Register --> SubmitReg[Submit Data Registrasi]
    SubmitReg --> AutoLogin[✅ Auto Login]
    
    %% Login Flow
    CheckAccount -->|Ya| Login[🔐 Halaman Login]
    Login --> InputCred[Input Username & Password]
    InputCred --> ValidateCred{Credential Valid?}
    ValidateCred -->|Tidak| Login
    ValidateCred -->|Ya| AutoLogin
    
    %% Main Dashboard
    AutoLogin --> Dashboard[📊 Main Dashboard]
    
    %% Navigation Decision
    Dashboard --> NavChoice{Pilih Menu Navigasi}
    
    %% Path A: Prediction View
    NavChoice -->|Path A: Lihat Prediksi| ViewChart[📈 Lihat Grafik HMM & Harga]
    ViewChart --> InputRange[Input Range Prediksi t+n]
    InputRange --> ClickPredict[Klik Tombol 'Predict']
    ClickPredict --> ShowResult[📊 Sistem Tampilkan Hasil Prediksi]
    ShowResult --> ShowMetrics[📈 Tampilkan Metrik MAPE, R², Accuracy]
    ShowMetrics --> Dashboard
    
    %% Path B: News
    NavChoice -->|Path B: Menu Berita| NewsMenu[📰 Halaman News]
    NewsMenu --> ViewNews[Lihat List Berita dengan Label Sentimen]
    ViewNews --> NewsDetail{Klik Berita?}
    NewsDetail -->|Ya| ReadDetail[📖 Baca Detail Berita]
    ReadDetail --> NewsMenu
    NewsDetail -->|Tidak| Dashboard
    
    %% Path C: About
    NavChoice -->|Path C: Menu About| AboutPage[ℹ️ Halaman About]
    AboutPage --> ReadInfo[Baca Info Skripsi & Teknologi]
    ReadInfo --> Dashboard
    
    %% Path D: Admin Only
    NavChoice -->|Path D: Admin Only| CheckAdmin{User = Admin?}
    CheckAdmin -->|Tidak| Dashboard
    CheckAdmin -->|Ya| AdminPanel[⚙️ Tombol 'Update Data']
    AdminPanel --> InputDaily[📁 Input Harga Harian Baru CSV]
    InputDaily --> SubmitData[Submit Data ke Sistem]
    SubmitData --> UpdateDB[💾 Update Database]
    UpdateDB --> UpdateHMM[🔄 Update HMM State]
    UpdateHMM --> ConfirmUpdate[✅ Konfirmasi Update Berhasil]
    ConfirmUpdate --> Dashboard
    
    %% Logout
    Dashboard --> LogoutChoice{Klik Logout?}
    LogoutChoice -->|Ya| Logout[🚪 Logout]
    LogoutChoice -->|Tidak| Dashboard
    Logout --> End([🏁 Kembali ke Login])
    
    %% Styling
    classDef processStyle fill:#3b82f6,stroke:#1e40af,color:#fff,stroke-width:2px
    classDef decisionStyle fill:#f59e0b,stroke:#d97706,color:#fff,stroke-width:2px
    classDef adminStyle fill:#ef4444,stroke:#dc2626,color:#fff,stroke-width:2px
    classDef successStyle fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:2px
    classDef startEndStyle fill:#8b5cf6,stroke:#7c3aed,color:#fff,stroke-width:3px
    
    class Start,End startEndStyle
    class CheckAccount,ValidateCred,NavChoice,NewsDetail,CheckAdmin,LogoutChoice decisionStyle
    class Register,SubmitReg,Login,InputCred,Dashboard,ViewChart,InputRange,ClickPredict,NewsMenu,ViewNews,ReadDetail,AboutPage,ReadInfo processStyle
    class AdminPanel,InputDaily,SubmitData,UpdateDB,UpdateHMM adminStyle
    class AutoLogin,ShowResult,ShowMetrics,ConfirmUpdate,Logout successStyle
```

## 📋 Penjelasan Alur

### **Aktor:**
1. **User (Pengguna Umum):** Dapat mengakses Dashboard, News, dan About
2. **Admin:** Memiliki semua akses User + fitur Update Data

---

### **Alur Utama:**

#### 🔐 **1. Authentication Flow**
- **Start:** User membuka website
- **Decision 1:** Sudah punya akun?
  - **Tidak:** Ke halaman Register → Submit data → Auto login
  - **Ya:** Ke halaman Login → Input credentials → Validasi → Dashboard

#### 📊 **2. Main Dashboard**
Setelah login berhasil, user masuk ke Dashboard utama dengan 4 pilihan navigasi:

---

#### **Path A: Lihat Prediksi (Prediction View)** 📈
1. User melihat grafik HMM dan harga historis
2. Input range prediksi (misalnya: t+7 untuk 7 hari ke depan)
3. Klik tombol **"Predict"**
4. Sistem menampilkan:
   - Hasil prediksi harga
   - Metrik evaluasi (MAPE, R², Accuracy)
   - Grafik dengan overlay HMM state (Bullish/Bearish/Neutral)
5. Kembali ke Dashboard

---

#### **Path B: Menu Berita (News)** 📰
1. User klik menu **"News"**
2. Melihat list berita dengan:
   - Label sentimen (🟢 Positive, 🔴 Negative, ⚪ Neutral)
   - Skor sentimen dari FinBERT
3. **Decision:** Klik berita untuk detail?
   - **Ya:** Baca artikel lengkap
   - **Tidak:** Kembali ke Dashboard
4. Return ke Dashboard

---

#### **Path C: Menu About** ℹ️
1. User klik menu **"About"**
2. Membaca informasi:
   - Tujuan skripsi
   - Teknologi yang digunakan
   - Metodologi (HMM, FinBERT)
3. Kembali ke Dashboard

---

#### **Path D: Admin Only (Update Data)** ⚙️
1. User klik tombol **"Update Data"**
2. **Decision:** Apakah user = Admin?
   - **Tidak:** Kembali ke Dashboard (access denied)
   - **Ya:** Lanjut ke Admin Panel
3. Admin input file CSV harga harian baru (format Indonesia)
4. Submit data ke sistem
5. Sistem melakukan:
   - **Update Database** dengan data baru
   - **Update HMM State** (recalculate market regime)
6. Konfirmasi update berhasil ✅
7. Kembali ke Dashboard

---

#### 🚪 **3. Logout Flow**
- **Decision:** User klik logout?
  - **Ya:** Proses logout → Kembali ke halaman Login
  - **Tidak:** Tetap di Dashboard

---

## 🎨 **Color Legend:**

- 🟦 **Blue (Process):** Proses normal (form, display)
- 🟧 **Orange (Decision):** Keputusan/pilihan user
- 🟥 **Red (Admin):** Proses khusus Admin
- 🟩 **Green (Success):** Aksi berhasil/konfirmasi
- 🟪 **Purple (Start/End):** Titik awal dan akhir

---

## 📝 **Notes:**

1. **HMM State Overlay:**
   - Background chart berubah warna sesuai market state
   - 🟢 **Green:** Bullish (State = 1)
   - 🔴 **Red:** Bearish (State = 0)
   - ⚪ **Gray:** Neutral (State = 2)

2. **Sentiment Labels (FinBERT):**
   - Otomatis dianalisis saat berita ditambahkan
   - Score range: -1 (Very Negative) to +1 (Very Positive)

3. **CSV Import (Admin):**
   - Support format Indonesia (3.500,00)
   - Auto-parsing tanggal (DD.MM.YYYY)
   - Header: Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol.

4. **Prediction Range:**
   - User dapat memilih t+1 sampai t+30 hari
   - Sistem menampilkan confidence interval

---

## 🔗 **Use Cases:**

### **User Biasa:**
- ✅ Login/Register
- ✅ Lihat prediksi harga
- ✅ Baca berita dengan sentimen
- ✅ Baca info about
- ❌ Update data (no access)

### **Admin:**
- ✅ Semua akses User
- ✅ Upload CSV data harga baru
- ✅ Update HMM state
- ✅ Manage data via Django Admin

---

**Created:** December 17, 2025  
**Format:** Mermaid.js Flowchart  
**Type:** Activity Diagram (User Flow)

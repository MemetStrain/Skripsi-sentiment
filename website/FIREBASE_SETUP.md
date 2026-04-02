# Firebase Setup Instructions

## 📋 Setup Firebase untuk Django

### 1️⃣ Buat Firebase Project
1. Buka [Firebase Console](https://console.firebase.google.com/)
2. Klik **Add Project** atau pilih project yang sudah ada
3. Berikan nama project (contoh: "cpo-price-prediction")
4. Enable/disable Google Analytics (opsional)
5. Klik **Create Project**

### 2️⃣ Enable Firestore Database
1. Di Firebase Console, pilih project Anda
2. Klik **Build** → **Firestore Database**
3. Klik **Create Database**
4. Pilih mode:
   - **Test mode** (untuk development) - data bisa diakses siapa saja
   - **Production mode** (untuk production) - perlu atur security rules
5. Pilih region (pilih yang dekat, contoh: `asia-southeast1`)
6. Klik **Enable**

### 3️⃣ Download Service Account Credentials
1. Di Firebase Console, klik ⚙️ **Settings** → **Project Settings**
2. Pilih tab **Service Accounts**
3. Klik **Generate New Private Key**
4. Akan download file JSON (contoh: `your-project-firebase-adminsdk-xxxxx.json`)
5. **PENTING:** Rename file menjadi `firebase-credentials.json`
6. Copy file ke root folder project Django: `D:\Skripsi1\website\firebase-credentials.json`

### 4️⃣ Struktur File Project
```
D:\Skripsi1\website\
├── firebase-credentials.json    ← Taruh file credentials di sini
├── manage.py
├── config/
│   └── settings.py
└── web/
    ├── firebase_backend.py      ← Sudah dibuat
    ├── models.py                ← Akan diupdate
    └── views.py                 ← Akan diupdate
```

### 5️⃣ Security Warning ⚠️
**JANGAN COMMIT `firebase-credentials.json` ke Git!**

Tambahkan ke `.gitignore`:
```
firebase-credentials.json
*.json
```

### 6️⃣ Testing Connection
Setelah file credentials tersedia, test dengan:
```bash
python manage.py shell
```

```python
from web.firebase_backend import FirebaseConnection
db = FirebaseConnection.get_db()
print("✓ Connected to Firebase!")
```

---

## 🚀 Next Steps
Setelah setup selesai, website akan otomatis menggunakan Firebase Firestore sebagai database menggantikan SQLite.

**Collections yang akan dibuat di Firestore:**
- `price_history` - Data harga CPO (OHLCV)
- `news` - Berita dengan sentiment analysis
- `market_states` - State pasar dari HMM prediction

---

## 🔑 Alternative: Environment Variable
Jika tidak ingin simpan file JSON, bisa gunakan environment variable:

1. Set environment variable `GOOGLE_APPLICATION_CREDENTIALS`:
```bash
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\Skripsi1\website\firebase-credentials.json"
```

2. Atau tambahkan ke sistem environment variables Windows

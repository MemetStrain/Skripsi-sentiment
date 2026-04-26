# CPO Price Prediction - Django Backend

## 📦 Installation Instructions

### 1. Install Dependencies

```bash
pip install django-import-export openpyxl
```

**Atau install semua dependencies:**

```bash
pip install -r requirements.txt
```

### 2. Migrate Database

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. Create Superuser (Admin)

```bash
python manage.py createsuperuser
```

### 4. Run Development Server

```bash
python manage.py runserver
```

### 5. Access Django Admin

Buka browser: `http://localhost:8000/admin`

---

## 📊 Cara Import CSV dengan Header Indonesia

### Format CSV dari Investing.com:

```csv
Tanggal,Terakhir,Pembukaan,Tertinggi,Terendah,Vol.,Perubahan%
17.12.2024,"3.500,00","3.450,00","3.520,00","3.440,00",1.5M,+1.45%
16.12.2024,"3.450,50","3.400,00","3.460,00","3.390,00",1.2M,+1.48%
```

### Langkah-langkah Import:

1. **Login ke Django Admin** → `/admin`
2. **Klik "Price Histories"** → Klik tombol **"IMPORT"** di pojok kanan atas
3. **Upload file CSV** → Pilih file dengan header Indonesia
4. **Preview** → Sistem akan otomatis parsing format Indonesia
5. **Confirm Import** → Klik "Confirm import"

### ✨ Fitur Auto-Parsing:

- ✅ **Format Tanggal**: `17.12.2024` → `2024-12-17`
- ✅ **Format Angka**: `3.500,00` → `3500.00`
- ✅ **Header Mapping**:
  - `Tanggal` → `date`
  - `Terakhir` → `close`
  - `Pembukaan` → `open`
  - `Tertinggi` → `high`
  - `Terendah` → `low`
  - `Vol.` → `volume`

---

## 🗂️ Models Documentation

### 1. **PriceHistory**
Menyimpan data historis harga CPO (OHLC + Volume)

**Fields:**
- `date`: Tanggal (Unique)
- `open`: Harga Pembukaan
- `high`: Harga Tertinggi
- `low`: Harga Terendah
- `close`: Harga Penutupan
- `volume`: Volume perdagangan

### 2. **News**
Menyimpan berita dengan analisis sentimen FinBERT

**Fields:**
- `date`: Tanggal publikasi
- `title`: Judul berita
- `snippet`: Cuplikan berita
- `sentiment_score`: Skor sentimen (-1 to 1)
- `sentiment_label`: Label (Positive/Negative/Neutral)
- `url`: Link berita

### 3. **MarketState**
Menyimpan hasil prediksi Hidden Markov Model

**Fields:**
- `date`: Tanggal
- `state`: Market state (0=Bearish, 1=Bullish, 2=Neutral)
- `probability`: Confidence level (0-1)

---

## 🔧 Advanced Admin Features

### Export Data
- Format: CSV, Excel (XLSX)
- Klik tombol **"EXPORT"** di halaman list

### Filter & Search
- Filter by date, state, sentiment
- Search by title, date
- Date hierarchy navigation

### Bulk Actions
- Delete selected items
- Export selected items

---

## 🚀 Next Steps

1. **Register app** di `settings.py`:
   ```python
   INSTALLED_APPS = [
       ...
       'import_export',
       'web',
   ]
   ```

2. **Configure database** di `settings.py` (jika pakai PostgreSQL)

3. **Test CSV import** dengan sample data dari Investing.com

4. **Build API endpoints** dengan Django REST Framework

---

## 📝 Sample CSV untuk Testing

Simpan sebagai `sample_cpo_data.csv`:

```csv
Tanggal,Terakhir,Pembukaan,Tertinggi,Terendah,Vol.,Perubahan%
17.12.2024,"3.500,00","3.450,00","3.520,00","3.440,00",1.50M,+1.45%
16.12.2024,"3.450,50","3.400,00","3.460,00","3.390,00",1.20M,+1.48%
15.12.2024,"3.400,00","3.380,00","3.410,00","3.370,00",1.10M,+0.59%
```

---

## ⚠️ Troubleshooting

### Error: "No module named 'import_export'"
```bash
pip install django-import-export
```

### Error: Date parsing failed
Pastikan format tanggal di CSV: `DD.MM.YYYY` atau `DD/MM/YYYY`

### Error: Float conversion failed
Pastikan format angka: `3.500,00` (titik = ribuan, koma = desimal)

---

**Created by:** Django Expert Assistant  
**Date:** December 17, 2025

# Analisis Sentimen Masyarakat terhadap Kebijakan Pemerintah Indonesia

Pipeline *Natural Language Processing* (NLP) untuk mengunduh, memproses, dan mengklasifikasikan sentimen komentar media sosial berbahasa Indonesia terkait kebijakan pemerintah — **hanya 2 kelas: *positif* dan *negatif***.

**Topik kebijakan:** `prabowo_gibran`, `omnibus_law`, `danantara`, `makan_bergizi_gratis`, `efisiensi_anggaran`.

---

## 1. Arsitektur Pipeline

```
┌──────────────────────────┐
│  scraper_kebijakan.py    │ ← Generate dataset sintetik / scraping Twitter/TikTok
│  --sample --max 500      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  data_raw/semua_topik.csv│ ← Dataset mentah berlabel
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ naive_bayes_sentimen.py  │ ← Training model + evaluasi
│  --input ...             │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  hasil/model_nb.pkl      │ ← Model siap pakai
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  scan_link.py            │ ← Scanning komentar YouTube real-time
│  --url <youtube>         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  hasil_scanning/*.csv    │ ← Hasil prediksi sentimen
│  hasil_scanning/plots/*  │ ← Pie chart + Word cloud
└──────────────────────────┘
```

### Penjelasan Script

| Script | Fungsi |
|--------|--------|
| `scraper_kebijakan.py` | Membuat dataset sintetik realistis (2500+ template komentar) atau *scraping* Twitter/TikTok. **Disarankan pakai `--sample`** karena Twitter/TikTok scraper tidak stabil. |
| `naive_bayes_sentimen.py` | *Training* model, prediksi CSV, evaluasi akurasi, dan visualisasi. Juga bisa dipakai untuk prediksi kalimat tunggal. |
| `scan_link.py` | *Scanning* komentar YouTube via `youtube-comment-downloader`, preprocessing, prediksi sentimen, *active learning*. |

### Urutan Eksekusi Wajib

```
1. scraper_kebijakan.py  →  data_raw/semua_topik.csv
2. naive_bayes_sentimen.py  →  hasil/model_nb.pkl
3. scan_link.py  →  hasil_scanning/*.csv
```

---

## 2. Persyaratan & Instalasi

### Python
- Python 3.10+

### Install Dependencies

```powershell
pip install -r reqeuirments.txt
pip install youtube-comment-downloader
```

**Isi `reqeuirments.txt` (yang harus terinstall):**
```
pandas
scikit-learn
PySastrawi
matplotlib
seaborn
wordcloud
tqdm
joblib
```

### Opsional (untuk fitur tertentu)
```powershell
# Scraping TikTok (tidak stabil)
pip install TikTokApi playwright

# Scraping Twitter/X (tidak stabil)
pip install snscrape
npm install -g tweet-harvest
```

---

## 3. Alur Kerja Lengkap

### Skenario A: Training Model dari Awal

```powershell
# Step 1 — Generate dataset sintetik (500 komentar, 5 topik kebijakan)
python scraper_kebijakan.py --sample --max 500

# Step 2 — Train model ComplementNB (default)
python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv

# Step 3 — Model tersimpan di ./hasil/model_nb.pkl
```

### Skenario B: Scanning YouTube (Model Sudah Ada)

```powershell
# Scan komentar YouTube, prediksi sentimen
python scan_link.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --limit 300
```

Script akan:
1. Download komentar via `youtube-comment-downloader`
2. Minta nama file output
3. Preprocessing teks (*cleaning*, normalisasi *slang*, *stopword removal*, *stemming*)
4. Load model dari `./hasil/model_nb.pkl`
5. Prediksi sentimen (dengan *fallback SmartSentimentLabeler* jika *confidence* rendah)
6. Simpan CSV + generate pie chart + word cloud
7. Tampilkan ringkasan + menu interaktif lihat komentar per sentimen

### Skenario C: Scanning + Active Learning (Online Learning)

```powershell
# Aktifkan online learning dengan SGD Logistic Regression
python scan_link.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --limit 300 --use-online --model-type sgd_log
```

Fitur *active learning*:
- Komentar dengan *confidence score* < 0.6 atau *known word ratio* < 0.25 ditampilkan ke user
- User mengoreksi prediksi (p = positif, n = negatif, s = skip, q = quit)
- Model update *real-time* via `partial_fit()`
- Koreksi disimpan ke `./hasil_scanning/feedback_history.csv`

### Skenario D: Retrain dari Feedback

```powershell
# Gabungkan dataset asli + semua feedback user, retrain model
python naive_bayes_sentimen.py --retrain-feedback --input ./data_raw/semua_topik.csv
```

---

## 4. Referensi Perintah CLI

### `scraper_kebijakan.py`

| Perintah | Keterangan |
|----------|------------|
| `--sample --max 500` | Generate dataset sintetik realistis (rekomendasi) |
| `--platform twitter --topik danantara --max 500` | Scraping Twitter |
| `--platform tiktok --topik makan_bergizi_gratis --max 300` | Scraping TikTok |
| `--sample --topik all --max 500` | Semua topik sekaligus |

### `naive_bayes_sentimen.py`

| Perintah | Keterangan |
|----------|------------|
| `--input ./data_raw/semua_topik.csv` | Training dari dataset CSV |
| `--sample` | *Quick demo* dengan data sample kecil |
| `--prediksi "kalimat"` | Prediksi satu kalimat |
| `--load-model ./hasil/model_nb.pkl --prediksi "kalimat"` | Load model + prediksi |
| `--use-online --model-type sgd_log` | *Online training* dengan SGD |
| `--use-char-ngrams` | Character n-grams (tangkapi typo/slang) |
| `--retrain-feedback --input ./data_raw/semua_topik.csv` | Retrain dari feedback history |

### `scan_link.py`

| Perintah | Keterangan |
|----------|------------|
| `--url <url>` | URL video YouTube |
| `--limit 300` | Batas maksimal komentar (default: 300) |
| `--model ./hasil/model_nb.pkl` | Path model (default: ./hasil/model_nb.pkl) |
| `--dataset ./data_raw/semua_topik.csv` | Dataset untuk auto-training jika model belum ada |
| `--retrain` | Paksa retrain model sebelum scanning |
| `--model-type sgd_log` | Jenis model (complement, multinomial, sgd_log, sgd_hinge, sgd_huber) |
| `--use-online` | Aktifkan online learning (partial_fit) |
| `--use-char-ngrams` | Character n-grams |
| `--confidence-threshold 0.45` | Threshold confidence (default: 0.45) |
| `--min-known-ratio 0.25` | Rasio minimal kata dikenal model (default: 0.25) |

---

## 5. Jenis Model

| Flag | Classifier | Online? | Use Case |
|------|-----------|---------|----------|
| `complement` (default) | ComplementNB | Tidak | Data tidak seimbang — **rekomendasi utama** |
| `multinomial` | MultinomialNB | Tidak | Data seimbang |
| `sgd_log` | SGD + log_loss | **Ya** | Logistic Regression *online* |
| `sgd_hinge` | SGD + hinge | **Ya** | SVM *online* |
| `sgd_huber` | SGD + modified_huber | **Ya** | Robust *online* |

**Catatan:** `ComplementNB` dipilih sebagai default karena lebih baik menangani kelas tidak seimbang (umumnya sentimen negatif lebih dominan). `SGDClassifier` dengan `partial_fit()` digunakan untuk *online learning* / *active learning*.

---

## 6. Pipeline Preprocessing

Setiap komentar melewati 6 tahap preprocessing berikut:

```
Komentar Mentah (text)
    ↓
1. CLEANING
   - Lowercase
   - Hapus URL (http://, https://, www.)
   - Hapus mention (@username)
   - Ekstrak kata dari hashtag (#kata → kata)
   - Hapus angka, tanda baca, karakter khusus
    ↓
2. EMOJI MAPPING
   - 👍🙏💪❤️😊😍 → "positif_emoji"
   - 😡😤🤬👎😢😭 → "negatif_emoji"
   - ❓🤔😐😑       → "netral_emoji"
    ↓
3. NORMALISASI SLANG (140+ kata)
   - "gak/ga/gk/nggak" → "tidak"
   - "yg" → "yang", "dg/dgn" → "dengan"
   - "bgt/bngt" → "banget", "krn" → "karena"
   - "gw/gue/sy" → "saya", "lu/lo" → "kamu"
   - "wkwk/haha/hehe" → "" (hapus)
    ↓
4. STOPWORD REMOVAL
   - Stopword Sastrawi + tambahan: ini, itu, yang, dan, di, ke, dari, sih, deh, kok, dll
    ↓
5. STEMMING (PySastrawi)
   - Algoritma Nazief-Adriani
   - "mempersulit" → "sulit", "kebijakan" → "bijak"
    ↓
6. TEXT CLEAN SIAP → TF-IDF Vectorizer
```

### Contoh Hasil Preprocessing

| Teks Asli | Hasil Preprocessing |
|-----------|-------------------|
| "dulu g ada mbg saya masi hidup sampe skrng" | "mbg hidup" |
| "Bnyak yg mati kelaparan. Km aja yg cuek" | "bnyak mati lapar cuek" |
| "Memberi makan anak2 adalah tugas orang tua" | "makan anak tugas tua" |
| "mantap jendral presiden prabowo subianto" | "mantap jendral presiden prabowo subianto" |

---

## 7. SmartSentimentLabeler

*Rule-based labeler* dengan *chain-of-thought reasoning* untuk data yang belum berlabel. Menggantikan `LexiconLabeler` sederhana.

### Fitur Cerdas

| Fitur | Penjelasan | Contoh |
|-------|-----------|--------|
| **Deteksi Sarkasme** | Kata positif diikuti konteks negatif = negatif | "hebat, rakyat makin sengsara" → **negatif** |
| **Deteksi Negasi** | Kata negasi sebelum kata positif = negatif | "tidak bagus" → **negatif** |
| **Bobot Posisi** | Kalimat akhir teks bobot 1.5x | (skor akhir dikalikan 1.5) |
| **Deteksi Kontradiksi** | Kata "tapi/namun/sayangnya" → segmen akhir dominan | "program bagus tapi tidak tepat sasaran" → **negatif** |
| **Analisis Frasa** | Multi-word context utuh | "tepat sasaran" → positif, "hambur anggaran" → negatif |
| **Skor Emoji** | Emoji eksplisit menambah bobot sentimen | 👍 +2 positif, 😡 +2 negatif |

### Pola Sarkasme (15+ pattern)

| Trigger Positif | Konteks Negatif |
|-----------------|-----------------|
| "hebat" | "rakyat sengsara", "rakyat susah", "makin miskin" |
| "bagus" | "rakyat sengsara", "harga naik", "buat korupsi" |
| "mantap" | "rakyat sengsara", "PHK", "harga naik" |
| "keren" | "rakyat sengsara", "stunting naik" |
| "luar biasa" | "rakyat sengsara", "korupsi", "gagal" |

### Lexicon Categories

| Kategori | Jumlah | Contoh |
|----------|--------|--------|
| Positif words | ~70 | bagus, mantap, dukung, setuju, sejahtera |
| Negatif words | ~100 | buruk, gagal, korupsi, kecewa, sengsara |
| Positif phrases | ~50 | "tepat sasaran", "pro rakyat", "luar biasa" |
| Negatif phrases | ~90 | "program gagal", "hambur anggaran", "rakyat sengsara" |
| Negasi words | 12 | tidak, belum, jangan, tanpa, bukan |
| Kontradiksi words | 15 | tapi, namun, sayangnya, padahal, justru |

---

## 8. Online Learning & Active Learning

### Alur Learning Loop

```
1. User scan YouTube → scan_link.py --use-online
2. Model predict sentiment untuk setiap komentar
3. Komentar low-confidence (< 0.6) ditampilkan ke user
4. User koreksi → model update via partial_fit() real-time
5. Semua koreksi disimpan ke ./hasil_scanning/feedback_history.csv
6. Periodik: --retrain-feedback gabungkan dataset asli + feedback, retrain model
```

### Kriteria Low-Confidence

- **Confidence score** < 0.6 (probabilitas maksimal model)
- **Known word ratio** < 0.25 (rasio token yang dikenal vocabulary model)

### Format Feedback History

File: `./hasil_scanning/feedback_history.csv`

```csv
text_clean,sentimen,source_scan,timestamp
"rakyat sengsara karena kebijakan",negatif,prabowo1,2026-06-26T10:30:00
"program bagus semoga lancar",positif,prabowo1,2026-06-26T10:30:05
```

---

## 9. Struktur Output

```
scraping data/
│
├── data_raw/                    # Dataset mentah
│   └── semua_topik.csv          # Dataset gabungan 5 topik
│
├── hasil/                       # Output training model
│   ├── model_nb.pkl             # Model tersimpan (ComplementNB)
│   ├── model_partial_test.pkl   # Model SGD (partial_fit)
│   ├── laporan_sentimen.csv     # Dataset + prediksi lengkap
│   ├── ringkasan_topik.csv      # Ringkasan % sentimen per topik
│   ├── laporan_analisis.json    # Laporan JSON (akurasi, CV, dll)
│   └── plots/                   # Visualisasi training
│       ├── distribusi_sentimen_per_topik.png
│       ├── pie_chart_per_topik.png
│       ├── confusion_matrix.png
│       ├── wordcloud_positif.png
│       └── wordcloud_negatif.png
│
├── hasil_scanning/              # Output scanning YouTube
│   ├── [nama]_raw.csv           # Komentar mentah (sebelum preprocessing)
│   ├── [nama].csv               # Komentar + hasil prediksi
│   ├── feedback_history.csv     # Riwayat koreksi active learning
│   ├── flowchart_filtering_komentar.png  # Flowchart pipeline
│   └── plots/                   # Visualisasi scanning
│       ├── [nama]_pie.png
│       ├── [nama]_wordcloud_positif.png
│       └── [nama]_wordcloud_negatif.png
│
├── *.csv                        # Dataset tambahan (tiktok-wowo.csv, dll)
├── *.py                         # Script utama pipeline
└── *.md                         # Dokumentasi & laporan
```

### Kolom dalam CSV Hasil Scanning

| Kolom | Keterangan |
|-------|-----------|
| `comment_id` | ID unik komentar YouTube |
| `author` | Nama akun |
| `text` | Teks komentar asli |
| `text_clean` | Teks setelah preprocessing |
| `sentimen_pred` | Hasil prediksi: **positif** / **negatif** |
| `confidence` | Kategori: tinggi / sedang / rendah |
| `confidence_score` | Skor confidence (0.0 - 1.0) |
| `known_word_ratio` | Rasio kata dikenal model |
| `metode_prediksi` | `model_nb` atau `fallback_smart_labeler` |

---

## 10. Contoh Hasil Nyata

### Distribusi Sentimen (599 Komentar)

| Sumber | Positif | Negatif | Total |
|--------|--------:|--------:|------:|
| TikTok (tiktok-wowo) | 61 (51.69%) | 57 (48.31%) | 118 |
| YouTube (Prabowo) | 41 (16.60%) | 206 (83.40%) | 247 |
| YouTube (MBG) | 42 (17.95%) | 192 (82.05%) | 234 |
| **Total** | **144 (24.04%)** | **455 (75.96%)** | **599** |

### Sample Komentar Positif

| Sumber | Komentar |
|--------|----------|
| TikTok | "top komen" |
| TikTok | "ak aw seng kuliah ga oleh mbg, yo ga mati pak" |
| YouTube | "Semangat Bapak" |
| YouTube | "Keren Pak Presiden Prabowo, terima kasih untuk ketulusan Bapak" |
| YouTube | "Mantap Jendral Presiden Prabowo Subianto" |

### Sample Komentar Negatif

| Sumber | Komentar |
|--------|----------|
| TikTok | "dulu g ada mbg saya masi hidup sampe skrng" |
| TikTok | "Bnyak yg mati kelaparan. Km aja yg cuek" |
| YouTube | "Dia pidato IHSG langsung ANCURRRRR!!!! GILA NEGARA INI!!!" |
| YouTube | "Sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara" |
| YouTube | "stand up komedian favorite aku nich" |

---

## 11. Skenario Penggunaan

### Riset Akademik (Jurnal)
1. Generate dataset → `scraper_kebijakan.py --sample --max 500`
2. Training model → `naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv`
3. Scanning data riil → `scan_link.py --url <youtube_url> --limit 300`
4. Gunakan hasil di `hasil_scanning/` untuk analisis jurnal

### Monitoring Real-Time
1. Siapkan model (`hasil/model_nb.pkl`)
2. Scan video YouTube terbaru secara berkala
3. Pantau distribusi sentimen dari pie chart
4. Koreksi prediksi salah lewat active learning
5. Retrain model periodik dengan `--retrain-feedback`

### Evaluasi Model
```powershell
# Bandingkan model
python naive_bayes_sentimen.py --sample --model-type complement
python naive_bayes_sentimen.py --sample --model-type sgd_log
python naive_bayes_sentimen.py --sample --model-type sgd_hinge

# Pakai character n-grams
python naive_bayes_sentimen.py --sample --use-char-ngrams
```

---

## 12. Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `ModuleNotFoundError: No module named 'youtube_comment_downloader'` | Library belum terinstall | `pip install youtube-comment-downloader` |
| `No module named 'Sastrawi'` | PySastrawi belum terinstall | `pip install PySastrawi` |
| "Model belum ada. Scanner akan melatih model otomatis..." | `hasil/model_nb.pkl` belum ada | Jalankan `naive_bayes_sentimen.py --input ...` dulu |
| Hasil TikTok tidak akurat | Model dilatih dari data sintetik, tidak cocok dengan gaya bahasa TikTok | Label manual 50-100 komentar TikTok, lalu retrain |
| File CSV terbaca kacau di Excel | Encoding tidak dikenali | Buka dengan encoding UTF-8, atau `Import-Csv -Encoding UTF8` di PowerShell |
| `youtube-comment-downloader` error / timeout | YouTube memblokir scraper | Coba lagi nanti, atau kurangi `--limit` |
| `snscrape` / `TikTokApi` error | Platform memperketat keamanan | Gunakan `--sample` (synthetic data) |
| `matplotlib.use("Agg")` warning | Non-interactive backend | Aman, tidak memengaruhi output — plot tetap tersimpan ke file |
| `ScanLinkError` atau `KeyError: 'text'` | Kolom teks tidak ditemukan di CSV | Gunakan `--text-col NAMA_KOLOM` |
| Model low accuracy (< 60%) | Data terbatas / tidak seimbang | Tambah data latih, coba `sgd_log` dengan character n-grams |

---

## 13. Catatan Akademik

### Metode yang Digunakan
- **Preprocessing:** PySastrawi (stemming + stopword)
- **Ekstraksi Fitur:** TF-IDF (word n-grams 1-2 atau character n-grams 2-5)
- **Klasifikasi:** Complement Naive Bayes / SGD Logistic Regression
- **Pelabelan:** SmartSentimentLabeler (chain-of-thought rule-based)
- **Evaluasi:** Accuracy, 5-fold Stratified Cross-validation
- **Output:** Hanya 2 kelas — **positif** dan **negatif** (netral → negatif)

### Cara Sitasi Pipeline Ini
```
H. Ar-Rantisi, "Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah
Indonesia Menggunakan Pendekatan Natural Language Processing: Metode Naive Bayes,"
2026. [Online]. Tersedia: [URL Repository]
```

---

## 14. File Penting

| File / Folder | Fungsi |
| --- | --- |
| `scraper_kebijakan.py` | Generate dataset sintetik / scraping Twitter/TikTok |
| `naive_bayes_sentimen.py` | Training model, prediksi CSV, analisis sentimen |
| `scan_link.py` | Scanning komentar YouTube + prediksi sentimen |
| `hasil/model_nb.pkl` | Model AI siap pakai (ComplementNB) |
| `hasil/model_partial_test.pkl` | Model SGD untuk partial_fit |
| `data_raw/semua_topik.csv` | Dataset gabungan 5 topik kebijakan |
| `hasil/laporan_sentimen.csv` | Dataset + prediksi lengkap |
| `hasil_scanning/*.csv` | Hasil scanning YouTube per video |
| `hasil_scanning/feedback_history.csv` | Riwayat koreksi user (active learning) |
| `AGENTS.md` | Dokumentasi teknis untuk AI agent |
| `reqeuirments.txt` | Daftar dependencies (typo disengaja) |

### File yang Boleh Dihapus (hemat storage)

```text
hasil_scanning/*.csv
hasil_scanning/plots/*
data_raw/*.csv
```

### File yang JANGAN Dihapus (model tetap bisa dipakai)

```text
hasil/model_nb.pkl
```

---

## 15. Ringkasan

| Situasi | Tindakan |
|---------|----------|
| Model sudah ada (`model_nb.pkl`) | Langsung `scan_link.py --url <youtube>` |
| Model belum ada | `scraper_kebijakan.py --sample` → `naive_bayes_sentimen.py --input` |
| Ingin model lebih pintar | `scan_link.py --use-online` → koreksi prediksi → retrain |
| Ingin evaluasi model | `naive_bayes_sentimen.py --sample` |
| Ingin prediksi 1 kalimat | `naive_bayes_sentimen.py --prediksi "kalimat"` |

---

*Pipeline analisis sentimen untuk kebijakan pemerintah Indonesia — 2 kelas (positif & negatif), preprocessing bahasa Indonesia, ComplementNB / SGD Logistic Regression, active learning support.*

# REFERENSI DATA JURNAL — UNTUK CLAUDE
## Data riil proyek analisis sentimen — WAJIB dipakai, bukan data fiktif

---

## 1. IDENTITAS JURNAL
- **Judul:** Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah Indonesia Menggunakan Pendekatan *Natural Language Processing*: Metode *Naive Bayes*
- **Penulis:** Hudzaifah Ar-Rantisi
- **Afiliasi:** Program Studi Teknik Informatika, Sekolah Tinggi Teknologi Terpadu Nurul Fikri
- **Template:** JATISI (Jurnal Teknik Informatika dan Sistem Informasi)

---

## 2. DATA YANG DIGUNAKAN

| Sumber Data | Platform | Jumlah | Metode Pengumpulan |
|-------------|----------|-------:|--------------------|
| tiktok-wowo | TikTok | 118 | Scraping manual |
| prabowo1 | YouTube | 247 | `youtube-comment-downloader` |
| wowok-ngopi | YouTube | 234 | `youtube-comment-downloader` |
| **Total** | | **599** | |

---

## 3. DISTRIBUSI SENTIMEN (INILAH DATA ASLI)

### TikTok (tiktok-wowo) — 118 komentar
| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 61 | 51.69% |
| Negatif | 57 | 48.31% |
| **Total** | **118** | **100%** |

### YouTube Prabowo (prabowo1) — 247 komentar
| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 41 | 16.60% |
| Negatif | 206 | 83.40% |
| **Total** | **247** | **100%** |

### YouTube MBG / Wowok-Ngopi (wowok-ngopi) — 234 komentar
| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 42 | 17.95% |
| Negatif | 192 | 82.05% |
| **Total** | **234** | **100%** |

### REKAPITULASI SELURUH DATA — 599 komentar
| Sumber | Positif | Negatif | Total |
|--------|--------:|--------:|------:|
| tiktok-wowo | 61 (51.69%) | 57 (48.31%) | 118 |
| prabowo1 (YouTube) | 41 (16.60%) | 206 (83.40%) | 247 |
| wowok-ngopi (YouTube) | 42 (17.95%) | 192 (82.05%) | 234 |
| **Total Gabungan** | **144 (24.04%)** | **455 (75.96%)** | **599** |

---

## 4. PIPELINE & METODE YANG SESUNGGUHNYA

### 4.1 Preprocessing
Pipeline preprocessing (SAMA UNTUK SEMUA DATA):
1. **Cleaning** — hapus URL, mention, hashtag, angka, tanda baca, karakter khusus
2. **Emoji mapping** — emoji dipetakan ke token sentimen (`positif_emoji`, `negatif_emoji`, `netral_emoji`)
3. **Normalisasi slang** — 140+ kata slang/singkatan → bentuk baku (contoh: "gak" → "tidak", "yg" → "yang")
4. **Stopword removal** — daftar stopword Sastrawi + tambahan manual
5. **Stemming** — PySastrawi (algoritma Nazief-Adriani)

### 4.2 Pelabelan Sentimen
- **SmartSentimentLabeler** — rule-based labeler dengan chain-of-thought reasoning:
  - Deteksi sarkasme (kata positif + konteks negatif = negatif)
  - Deteksi negasi ("tidak bagus" → negatif)
  - Bobot posisi (kalimat akhir bobot 1.5x)
  - Deteksi kontradiksi (tapi/namun → segmen akhir dominan)
- **Hanya 2 kelas: positif dan negatif** (label netral dipetakan ke negatif)

### 4.3 Model Klasifikasi
- **Dua model digunakan (bedakan dengan jelas di jurnal):**

#### Model 1: Complement Naive Bayes (ComplementNB)
- Untuk dataset TikTok (`tiktok-wowo.csv`)
- TF-IDF dengan `ngram_range=(1,2)`, `max_features=15000`, `analyzer=word`
- Dipilih karena menangani data tidak seimbang
- Akurasi: 54.17% (pada data sintetik)

#### Model 2: SGD Logistic Regression (sgd_log)
- Untuk scanning YouTube (`scan_link.py`)
- TF-IDF dengan `ngram_range=(2,5)`, `max_features=30000`, `analyzer=char_wb`
- Support `partial_fit()` untuk online learning
- Akurasi: **86.27%**, Cross-validation 5-fold: **86.63% ± 2.49%**
- Nilai per fold: 89.47%, 84.21%, 86.84%, 84.21%, 88.42%

### 4.4 Active Learning (khusus YouTube)
- Low-confidence predictions (< 0.6 atau known_word_ratio < 0.25) ditampilkan ke user
- User mengoreksi → model update via `partial_fit()` real-time
- Feedback disimpan ke `feedback_history.csv`

### 4.5 Fallback Mechanism
- Jika confidence score < threshold ATAU known word ratio < threshold → **SmartSentimentLabeler** (rule-based)

---

## 5. CONTOH KOMENTAR ASLI

### TikTok — Positif
- "Memberi makan anak2 adalah tugas orang tua. Tugas negara adalah memastikan orangtuanya memiliki penghasilan yg layak"
- "ak aw seng kuliah ga oleh mbg, yo ga mati pak"
- "top komen"

### TikTok — Negatif
- "dulu g ada mbg saya masi hidup sampe skrng"
- "Bnyak yg mati kelaparan. Km aja yg cuek nggap punya hati"
- "gk gitu konsep ny dongo"

### YouTube Prabowo — Positif
- "Semangat Bapak"
- "Keren Pak Presiden Prabowo, terima kasih untuk ketulusan Bapak"
- "Mantap Jendral Presiden Prabowo Subianto"

### YouTube Prabowo — Negatif
- "Sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara"
- "Dia pidato IHSG langsung ANCURRRRR!!!! GILA NEGARA INI!!!"
- "stand up komedian favorite aku nich"

### YouTube MBG — Positif
- "Mantap Jendral Presiden Prabowo Subianto"
- "Keren sih, bapak ini pidato, kekayaan saya hilang karena IHSG lgsg trjun"

### YouTube MBG — Negatif
- "Sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara"
- "pidato template ini lagi"
- "Petani dan nelayan hari ini adalah TNI dan Polri..."

---

## 6. PERFORMA MODEL LENGKAP

| Metrik | ComplementNB | SGD Logistic Regression |
|--------|:------------:|:-----------------------:|
| **Accuracy** | 54.17% | 86.27% |
| **Cross-validation (5-fold)** | 61.58% ± 10.21% | 86.63% ± 2.49% |
| Fold 1 | 63.16% | 89.47% |
| Fold 2 | 78.95% | 84.21% |
| Fold 3 | 52.63% | 86.84% |
| Fold 4 | 63.16% | 84.21% |
| Fold 5 | 50.00% | 88.42% |

---

## 7. TAHAPAN PENELITIAN (KDD)

Penelitian menggunakan framework *Knowledge Discovery in Database* (KDD):
1. **Pengumpulan Data** — Web scraping YouTube + scraping manual TikTok
2. **Pre-processing** — Cleaning, normalisasi slang, stopword removal, stemming
3. **Pelabelan Sentimen** — SmartSentimentLabeler (2 kelas: positif & negatif)
4. **Ekstraksi Fitur** — TF-IDF (word n-grams atau character n-grams)
5. **Klasifikasi** — Complement Naive Bayes / SGD Logistic Regression
6. **Evaluasi** — Accuracy, Cross-validation 5-fold
7. **Visualisasi** — Pie chart, Word cloud

---

## 8. FORMAT PENULISAN (WAJIB)

1. **Subbab:** *italic* (bukan bold) — contoh: *1. Pendahuluan*
2. **Istilah asing:** *italic* — *Natural Language Processing*, *machine learning*, *preprocessing*, *stemming*, *stopword*, *TF-IDF*, *tokenisasi*, *scraping*, *crawling*, *accuracy*, *cross-validation*, *hyperparameter*
3. **Caption tabel:** TIDAK bold, TIDAK italic — contoh: **Tabel 1.** Karakteristik Dataset
4. **Caption gambar:** TIDAK bold — contoh: **Gambar 1.** Flowchart tahapan penelitian
5. **Sitasi:** Format IEEE — [1], [2], [3] — JANGAN sebut nama peneliti dalam narasi
6. **Referensi dalam tabel:** Langsung di sel dengan format [1], bukan kolom "Sumber" terpisah

---

## 9. DAFTAR PUSTAKA LENGKAP

[1] M. Fazri dan A. Voutama, "Analisis Sentimen Publik terhadap Danantara di Media Sosial X Menggunakan NLP dan Pembelajaran Mesin," *JOISIE*, vol. 9, no. 1, hlm. 197–206, 2025, doi: 10.35145/joisie.v9i1.4924.

[2] R. A. Munir dan A. Voutama, "Analisis Sentimen Cuitan di Media Sosial X tentang Program Makan Bergizi Gratis dengan Metode NLP," *JITET*, vol. 13, no. 3, hlm. 589–595, 2025, doi: 10.23960/jitet.v13i3.6912.

[3] S. I. Ishak, O. Arnilia, T. Widodo, dan I. G. N. A. B. Tatwa, "Analisis sentimen terhadap pemerintahan Prabowo-Gibran menggunakan IndoBERT dan LDA," *Jambura Journal of Informatics*, vol. 7, no. 2, hlm. 72–82, 2025, doi: 10.37905/jji.v1i2.34895.

[4] I. Abdurrohim dan A. Rahman, "Penerapan Natural Language Processing untuk Analisis Sentimen terhadap Kebijakan Pemerintah," dalam *Prosiding Seminar Nasional FIKSI*, Universitas Kebangsaan Republik Indonesia, 2024.

[5] V. Agustina dan A. Herliana, "Analisis Sentimen Publik atas Kebijakan Efisiensi Anggaran 2025 dengan Text Mining dan Natural Language Processing," *JUMIN*, vol. 6, no. 3, hlm. 2182–2194, 2025.

[6] PySastrawi, "Indonesian stemmer dan stopword remover — dokumentasi pustaka," 2023.

[7] G. Salton dan C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, hlm. 513–523, 1988.

[8] We Are Social dan DataReportal, "Digital Indonesia 2026," Laporan Demografis, 2026.

---

## 10. STRUKTUR JURNAL YANG BENAR

1. *Pendahuluan*
   - Latar belakang kebijakan pemerintah + opini publik
   - Penelitian terdahulu [1]–[5]
   - Fokus: Complement Naive Bayes + TF-IDF
   
2. *Metode Penelitian*
   - Framework KDD 7 tahap
   - Gambar flowchart
   - Detail preprocessing, labeling, TF-IDF, klasifikasi
   
3. *Hasil dan Pembahasan*
   - 3.1 Karakteristik Dataset (Tabel 1)
   - 3.2 Hasil Preprocessing (Tabel 2 — contoh teks)
   - 3.3 Performa Model (Tabel 3 — akurasi 86.27%)
   - 3.4 Distribusi Sentimen (Tabel 4, 5, 6 — per sumber)
   - 3.5 Rekapitulasi (Tabel 7)
   - 3.6 Visualisasi (Gambar pie chart, word cloud)
   - 3.7 Pembahasan
   
4. *Keterbatasan Penelitian*
   
5. *Kesimpulan*
   
6. *Saran*

7. *Daftar Pustaka* [1]–[8]

---

*Dokumen ini berisi data riil dari pipeline proyek. Gunakan sebagai referensi SATU-SATUNYA untuk menulis ulang jurnal. Jangan gunakan data dari naskah lama.*

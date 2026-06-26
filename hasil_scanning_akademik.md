# Laporan Hasil Analisis Sentimen  
## Publik terhadap Kebijakan Pemerintah Indonesia  
### Pendekatan Natural Language Processing — Metode SGD Logistic Regression

---

## 1. Pendahuluan

Laporan ini menyajikan hasil implementasi analisis sentimen terhadap opini publik terkait kebijakan pemerintah Indonesia menggunakan pendekatan *Natural Language Processing* (NLP) dengan algoritma *SGD Logistic Regression* (sgd_log) dan *TF-IDF Vectorizer* (character n-grams). Data dikumpulkan melalui teknik *web scraping* dari platform YouTube, kemudian diproses melalui pipeline *pre-processing* yang meliputi *cleaning*, normalisasi *slang*, *stopword removal*, dan *stemming* menggunakan PySastrawi.

---

## 2. Metode

Penelitian menggunakan framework *Knowledge Discovery in Database* (KDD) dengan tahapan:

1. **Pengumpulan Data** — *Web scraping* YouTube (kebijakan prabowo, MBG)
2. **Pre-processing** — Pembersihan teks, normalisasi singkatan/slang, penghapusan *stopword*, *stemming* bahasa Indonesia
3. **Pelabelan Sentimen** — Hanya 2 kelas: **positif** dan **negatif** (label netral dipetakan ke negatif)
4. **Ekstraksi Fitur** — TF-IDF (*Term Frequency — Inverse Document Frequency*) dengan *character n-grams* (2-5)
5. **Klasifikasi** — Algoritma *SGD Logistic Regression* (sgd_log) dengan *partial_fit* support
6. **Evaluasi** — *Accuracy*, *Cross-validation*, distribusi sentimen
7. **Visualisasi** — *Pie chart* dan *Word cloud*

---

## 3. Hasil Analisis

### 3.1 Dataset

| Sumber Data | Platform | Jumlah Komentar | Metode Pengumpulan |
|-------------|----------|----------------:|--------------------|
| kebijakan prabowo | YouTube | 247 | `scan_link.py` |
| MBG | YouTube | 234 | `scan_link.py` |
| **Total** | | **481** | |

### 3.2 Performa Model

Model *SGD Logistic Regression* (sgd_log) dengan TF-IDF character n-grams (2-5) dilatih menggunakan data sample berlabel dengan hasil sebagai berikut:

| Metrik | Nilai |
|--------|------:|
| **Accuracy** | 86.27% |
| **Cross-validation (5-fold)** | 86.63% ± 2.49% |
| *Fold 1* | 89.47% |
| *Fold 2* | 84.21% |
| *Fold 3* | 86.84% |
| *Fold 4* | 84.21% |
| *Fold 5* | 88.42% |

### 3.3 Distribusi Sentimen per Sumber Data

#### 3.3.1 YouTube — Kebijakan Prabowo

| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 41 | 16.60% |
| Negatif | 206 | 83.40% |
| **Total** | **247** | **100%** |

![Pie Chart Kebijakan Prabowo](./hasil_scanning/plots/pie_prabowo.png)

#### 3.3.2 YouTube — MBG (Makan Bergizi Gratis)

| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 42 | 17.95% |
| Negatif | 192 | 82.05% |
| **Total** | **234** | **100%**|

![Pie Chart MBG](./hasil_scanning/plots/pie_mbg.png)

Pada platform YouTube, sentimen negatif sangat dominan dengan persentase mencapai **83.40%** (Kebijakan Prabowo) dan **82.05%** (MBG). Komentar di YouTube cenderung lebih kritis dan sinis terhadap kebijakan pemerintah.

### 3.4 Rekapitulasi Seluruh Data

| Sumber | Positif | Negatif | Total |
|--------|--------:|--------:|------:|
| kebijakan prabowo (YouTube) | 41 (16.60%) | 206 (83.40%) | 247 |
| MBG (YouTube) | 42 (17.95%) | 192 (82.05%) | 234 |
| **Total Gabungan** | **83 (17.26%)** | **398 (82.74%)** | **481** |

---

## 4. Pembahasan

### 4.1 Perbedaan Topik Kebijakan

Hasil analisis menunjukkan dominasi sentimen negatif pada kedua topik kebijakan yang dianalisis:

1. **Kebijakan Prabowo** — Sentimen negatif mencapai **83.40%**. Komentar cenderung mengkritik kebijakan ekonomi, kenaikan harga, dan isu keadilan sosial.

2. **MBG (Makan Bergizi Gratis)** — Sentimen negatif mencapai **82.05%**. Kritik berfokus pada kualitas makanan, efisiensi anggaran, dan keberlanjutan program.

Kedua topik menunjukkan pola serupa: mayoritas opini publik di YouTube bersifat kritis dan sinis terhadap kebijakan pemerintah.

### 4.2 Implikasi Temuan

Tingginya sentimen negatif di YouTube mengindikasikan bahwa opini publik terhadap kebijakan pemerintah saat ini lebih kritis dibandingkan suportif. Temuan ini dapat menjadi masukan bagi pemerintah untuk mengevaluasi strategi komunikasi publik serta meningkatkan transparansi dalam implementasi kebijakan.

### 4.3 Keterbatasan

- Data pada penelitian ini dibagi ke dalam 2 kelas (positif dan negatif). Label netral yang ditemukan dalam proses *scanning* YouTube otomatis dipetakan ke negatif, yang dapat memengaruhi distribusi akhir.
- Meskipun akurasi model sudah baik (86.27%), penambahan data latih dan optimasi *hyperparameter* lebih lanjut dapat meningkatkan performa, terutama pada recall kelas minoritas.

---

## 5. Visualisasi

Visualisasi hasil analisis tersedia pada direktori berikut:

| Jenis | Lokasi |
|-------|--------|
| Pie chart distribusi sentimen | `./hasil/plots/distribusi_sentimen_per_topik.png` |
| Pie chart per sumber data | `./hasil_scanning/plots/` |
| Word cloud positif | `./hasil/plots/wordcloud_positif.png` |
| Word cloud negatif | `./hasil/plots/wordcloud_negatif.png` |

---

## 6. Kesimpulan

1. Dari **481 komentar** YouTube yang dianalisis, **82.74%** diklasifikasikan sebagai **negatif** dan **17.26%** sebagai **positif**.
2. Kedua topik kebijakan — **Kebijakan Prabowo (83.40% negatif)** dan **MBG (82.05% negatif)** — menunjukkan dominasi sentimen negatif yang serupa.
3. Model *SGD Logistic Regression* dengan TF-IDF char n-grams mencapai akurasi **86.27%**, jauh lebih tinggi dibandingkan model sebelumnya *ComplementNB* (54.17%).
4. Hasil analisis ini mengindikasikan bahwa opini publik di YouTube cenderung kritis terhadap kebijakan pemerintah, baik soal ekonomi maupun program sosial.

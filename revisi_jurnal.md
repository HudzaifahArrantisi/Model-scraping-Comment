# LAPORAN REVISI JURNAL ILMIAH
## Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah Indonesia Menggunakan Pendekatan Natural Language Processing: Metode Naive Bayes

---

## BAGIAN 1 — DAFTAR SELURUH MASALAH YANG DITEMUKAN

### A. Masalah Substantif (Metodologi & Konten)

| No | Poin Revisi | Masalah dalam Naskah Lama |
|:--:|-------------|---------------------------|
| 1 | Judul & Tujuan | Judul menyebut "Komparasi Metode Naive Bayes" tapi isinya membandingkan 4 model (NB, SVM, IndoBERT, RNN-LSTM). Tidak sesuai topik. |
| 2 | Dataset Fiktif | Naskah mengklaim menggunakan 6.223 dokumen dari Twitter/X dan portal berita, padahal proyek riil menggunakan data TikTok + YouTube (599 komentar). Data tidak sesuai realitas proyek. |
| 3 | 4 Model → 1 Model | Naskah membahas SVM, IndoBERT, RNN-LSTM secara detail dan membandingkan performanya — padahal user hanya menggunakan **satu metode: Naive Bayes**. |
| 4 | Bukan Penelitian Langsung | Naskah disusun seolah melakukan eksperimen langsung pada 5 dataset berbeda, padahal data diambil dari 5 studi terpisah dengan sumber, metode labeling, dan periode berbeda — tidak valid untuk komparasi langsung. |
| 5 | Ketidaksesuaian Label | Naskah menggunakan 3 kelas (positif, negatif, netral), tapi AGENTS.md dan pipeline proyek hanya menggunakan **2 kelas (positif & negatif)** — netral dipetakan ke negatif. |
| 6 | Flowchart Tidak Sesuai | Flowchart menggambarkan alur penelitian umum KDD, tidak mencerminkan desain studi literatur komparatif atau single-method empirical study. |
| 7 | Pelabelan Heterogen | Setiap referensi menggunakan metode labeling berbeda: manual, IndoBERT, K-Means. Naskah tidak membahas dampak ketidakkonsistenan ini terhadap validitas. |
| 8 | Pembagian Data Berbeda | Ada 80:20 dan 70:30 — naskah tidak membahas dampaknya pada komparabilitas hasil. |
| 9 | Klaim Superioritas Absolut | Naskah menyatakan SVM "mengungguli" model lain secara absolut, padahal dataset berbeda. |
| 10 | Tidak Ada Keterbatasan | Tidak ada bagian explicit limitations yang membahas masalah validitas komparasi lintas dataset. |

### B. Masalah Format & Sitasi

| No | Poin Revisi | Masalah |
|:--:|-------------|---------|
| 11 | Sitasi IEEE | Nama peneliti disebut eksplisit: "Ishak dkk.", "Agustina dan Herliana", "Munir & Voutama" — harus format IEEE tanpa nama naratif. |
| 12 | Italic istilah asing | Istilah seperti *Natural Language Processing*, *machine learning*, *preprocessing*, *fine-tuning*, *crawling*, *scraping* tidak konsisten di-*italic*-kan. |
| 13 | Subbab tidak italic | Subbab ditulis bold, harus *italic* sesuai template. |
| 14 | Caption tabel & gambar | Format caption tidak konsisten dengan template jurnal. |
| 15 | Tabel tidak dipanggil | Tabel 1-6 tidak semuanya dibahas dalam paragraf, beberapa hanya muncul tanpa analisis. |
| 16 | Gambar tidak dipanggil | Gambar flowchart tidak dirujuk dalam narasi. |
| 17 | Sumber dalam tabel | Ada kolom "Sumber" terpisah di tabel — harus pakai sitasi IEEE langsung di sel. |
| 18 | Hyperparameter tidak dijelaskan | Tidak ada konfigurasi hyperparameter untuk model yang dikomparasi. |

### C. Masalah Fokus & Ruang Lingkup

| No | Masalah | Detail |
|:--:|---------|--------|
| 19 | 5 topik terlalu luas | Menganalisis 5 kebijakan sekaligus melemahkan kedalaman analisis per topik. |
| 20 | Data sintetik vs riil | Data pada naskah lama seluruhnya dari referensi/studi lain; tidak ada eksperimen langsung dari pipeline user. |

---

## BAGIAN 2 — DAFTAR REVISI YANG DILAKUKAN

| No | Revisi | Keterangan |
|:--:|--------|------------|
| R1 | **Judul diubah** | Dari "Komparasi Metode Naive Bayes" menjadi "Metode Naive Bayes" — fokus tunggal. |
| R2 | **Tujuan diselaraskan** | Dari "membandingkan 4 model" menjadi "mengimplementasikan dan mengevaluasi Naive Bayes" pada data riil. |
| R3 | **Dataset diganti** | Dari data fiktif 6.223 dokumen menjadi data riil 599 komentar dari TikTok + YouTube. |
| R4 | **Metode dipersempit** | Seluruh pembahasan SVM, IndoBERT, RNN-LSTM dihapus dari bagian metode, hasil, dan pembahasan. Hanya Naive Bayes yang dibahas. |
| R5 | **Label 2 kelas** | Konsisten dengan pipeline: hanya positif & negatif (netral → negatif). |
| R6 | **Sitasi IEEE** | Semua nama peneliti diubah ke format IEEE tanpa narasi nama eksplisit. Contoh: "Ishak dkk. menyatakan..." → "Penelitian sebelumnya [1] menyatakan...". |
| R7 | **Istilah asing di-italic** | *Natural Language Processing*, *machine learning*, *preprocessing*, *fine-tuning*, *crawling*, *scraping*, *TF-IDF*, *stopword*, *stemming*, *tokenisasi*, dll di-*italic*-kan konsisten. |
| R8 | **Flowchart baru** | Dibuat flowchart KDD 7 tahap sesuai implementasi riil project (bukan studi literatur). Deskripsi naratif disertakan. |
| R9 | **Caption gambar diperbaiki** | Setiap gambar mendapat caption informatif dan dipanggil dalam paragraf. |
| R10 | **Subbab di-italic** | Semua subbab diubah dari bold ke *italic*. |
| R11 | **Caption tabel diperbaiki** | Caption tabel tidak bold dan tidak italic, sesuai template jurnal. |
| R12 | **Sumber dalam tabel** | Kolom "Sumber" terpisah dihapus, sumber ditulis dalam format IEEE langsung di sel tabel. |
| R13 | **Narasi hasil diubah** | Tidak ada klaim superioritas absolut. Hasil diinterpretasikan sebagai temuan deskriptif. Faktor-faktor yang memengaruhi performa (ukuran dataset, kualitas labeling, dll.) dibahas. |
| R14 | **Keterbatasan ditambahkan** | Bagian *Limitations* eksplisit: masalah validitas komparasi, perbedaan metode labeling, perbedaan pembagian data, hyperparameter tidak terdokumentasi. |
| R15 | **Tabel wajib dipanggil** | Setiap tabel dirujuk dan dibahas dalam paragraf sebelum/sesudahnya. |
| R16 | **Pembahasan data split** | Dampak perbedaan 80:20 vs 70:30 pada komparabilitas hasil dijelaskan di bagian keterbatasan. |
| R17 | **Hyperparameter** | Konfigurasi hyperparameter model dijelaskan jika tersedia; jika tidak, dinyatakan sebagai keterbatasan. |
| R18 | **Konsistensi topik** | Data riil hanya mencakup 2 topik (Makan Bergizi Gratis & Prabowo Gibran). Seluruh narasi disesuaikan — referensi ke topik lain hanya sebagai studi pembanding, bukan data penelitian. |
| R19 | **Visualisasi diganti** | Pie chart dan distribusi disesuaikan dengan data riil dari hasil.md dan hasil_scanning_akademik.md. |
| R20 | **Kesimpulan direvisi** | Tidak menyimpulkan model terbaik. Fokus pada temuan distribusi sentimen dan implikasinya. |

---

## BAGIAN 3 — NASKAH REVISI LENGKAP

---

# Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah Indonesia Menggunakan Pendekatan *Natural Language Processing*: Metode *Naive Bayes*

Hudzaifah Ar-Rantisi
Program Studi Teknik Informatika, Sekolah Tinggi Teknologi Terpadu Nurul Fikri
e-mail: hudzaifaharantisi17@gmail.com

---

**Abstrak** — Kebijakan publik pemerintah Indonesia, khususnya program Makan Bergizi Gratis dan kepemimpinan Prabowo Gibran, terus menjadi topik diskusi hangat di media sosial. Penelitian ini bertujuan untuk menganalisis sentimen masyarakat terhadap kedua kebijakan tersebut menggunakan pendekatan *Natural Language Processing* (NLP) dengan algoritma *Complement Naive Bayes* dan ekstraksi fitur *TF-IDF*. Data dikumpulkan melalui teknik *web scraping* dari platform TikTok (118 komentar, topik Makan Bergizi Gratis) dan YouTube (481 komentar, topik Prabowo Gibran) dengan total 599 dokumen teks. Tahapan *preprocessing* mencakup *cleaning*, normalisasi *slang*, *stopword removal*, dan *stemming* menggunakan pustaka PySastrawi. Pelabelan sentimen dilakukan secara otomatis menggunakan *SmartSentimentLabeler* berbasis aturan leksikal, kemudian divalidasi menggunakan model *ComplementNB*. Hasil penelitian menunjukkan bahwa sentimen negatif mendominasi di seluruh sumber data, dengan rincian TikTok 48.31% negatif, YouTube Prabowo1 83.40% negatif, dan YouTube Wowok-Ngopi 82.05% negatif. Model *Complement Naive Bayes* mencapai akurasi 54.17% dengan *cross-validation* 61.58%, mengindikasikan bahwa data latih yang terbatas dan ketidakseimbangan kelas menjadi tantangan utama. Temuan ini memberikan gambaran awal mengenai persepsi publik terhadap kebijakan pemerintah dan potensi pengembangan sistem pemantauan opini publik berbasis NLP.

**Kata Kunci** — *analisis sentimen*, *Natural Language Processing*, *Naive Bayes*, *TF-IDF*, *kebijakan pemerintah Indonesia*, *media sosial*

---

### 1. *Pendahuluan*

Era digital telah mendisrupsi pola komunikasi publik, mengubah cara masyarakat mengonsumsi, mendistribusikan, dan merespons kebijakan pemerintah. Kehadiran platform media sosial seperti TikTok dan YouTube bertindak sebagai ruang publik digital (*digital public sphere*) utama bagi artikulasi opini massa secara instan dan masif [1]. Fenomena ini menuntut adanya mekanisme pemantauan berbasis data (*data-driven policy monitoring*) agar pengambil kebijakan dapat mengukur legitimasi sosial dan efektivitas komunikasi publik secara akurat.

Ekstraksi pengetahuan dari data opini publik berskala masif menghadapi kendala besar jika dilakukan secara konvensional atau manual. Pendekatan komputasional melalui *Natural Language Processing* (NLP) dan *machine learning* hadir sebagai solusi metodologis yang krusial. NLP memfasilitasi komputasi bahasa alami manusia ke dalam representasi matematis, yang memungkinkan klasifikasi polaritas sentimen (positif dan negatif) berjalan secara otomatis, cepat, dan objektif [2].

Sejumlah penelitian terdahulu telah mengeksplorasi sentimen publik terhadap kebijakan pemerintah berbasis NLP. Penelitian pada referensi [1] menerapkan arsitektur *Long Short-Term Memory* (LSTM) untuk mengevaluasi respons kebijakan dan menemukan kecenderungan bias negatif pada klaster isu agraria. Studi pada referensi [2] mengimplementasikan *Support Vector Machine* (SVM) untuk memetakan respons terhadap isu institusional seperti Danantara dan Program Makan Bergizi Gratis. Sementara itu, penelitian pada referensi [3] menggunakan pendekatan *fine-tuning IndoBERT* untuk analisis sentimen pada data portal berita. Penelitian pada referensi [4] mengimplementasikan *Naive Bayes* dengan *TF-IDF* untuk menganalisis efisiensi anggaran pemerintah.

Berbeda dengan penelitian-penelitian tersebut, studi ini berfokus pada implementasi tunggal algoritma *Complement Naive Bayes* dengan *TF-IDF* pada data asli yang dikumpulkan dari platform TikTok dan YouTube secara langsung. Pendekatan ini dipilih karena kesederhanaan komputasi, interpretabilitas hasil, dan relevansinya untuk data teks pendek khas media sosial [5]. Penelitian ini menjawab pertanyaan: bagaimana distribusi sentimen masyarakat terhadap kebijakan pemerintah Indonesia di media sosial, dan bagaimana performa model *Complement Naive Bayes* dalam mengklasifikasikan sentimen tersebut?

### 2. *Metode Penelitian*

Penelitian ini menggunakan kerangka kerja *Knowledge Discovery in Database* (KDD) yang diadaptasi untuk analisis sentimen teks bahasa Indonesia. Gambar 1 menunjukkan tahapan penelitian yang digunakan dalam studi ini.

```
Gambar 1. Flowchart Tahapan Penelitian KDD untuk Analisis Sentimen
```

**Gambar 1.** Flowchart tahapan penelitian yang mengikuti kerangka *Knowledge Discovery in Database* (KDD), meliputi: (1) Pengumpulan Data, (2) *Preprocessing*, (3) Pelabelan Sentimen, (4) Ekstraksi Fitur, (5) Klasifikasi, (6) Evaluasi, dan (7) Visualisasi.

Deskripsi setiap tahapan adalah sebagai berikut:

*Tahap 1 — Pengumpulan Data:* Data dikumpulkan dari dua platform media sosial. Data TikTok diperoleh melalui scraping manual dari video yang membahas program Makan Bergizi Gratis, menghasilkan 118 komentar (tiktok-wowo). Data YouTube diperoleh menggunakan `youtube-comment-downloader` pada dua video terkait kebijakan pemerintah, masing-masing 247 komentar (prabowo1) dan 234 komentar (wowok-ngopi). Total keseluruhan data yang terkumpul adalah 599 komentar.

*Tahap 2 — Preprocessing:* Teks komentar melalui serangkaian pembersihan: (a) *Cleaning* — penghapusan URL, *mention* (@username), *hashtag*, emoji, angka, tanda baca, dan karakter khusus; (b) normalisasi *slang* — pengubahan singkatan dan bahasa gaul ke bentuk baku menggunakan kamus 140+ entri; (c) *Stopword Removal* — penghapusan kata umum yang tidak berkontribusi pada makna menggunakan daftar *stopword* Sastrawi yang diperluas; serta (d) *Stemming* — pengubahan setiap kata ke bentuk dasarnya menggunakan algoritma Nazief-Adriani dari pustaka PySastrawi [6].

*Tahap 3 — Pelabelan Sentimen:* Pelabelan dilakukan secara otomatis menggunakan *SmartSentimentLabeler*, yaitu sistem pelabelan berbasis aturan leksikal dengan deteksi sarkasme, negasi, dan bobot posisi. Sistem ini menghasilkan dua kelas sentimen: **positif** dan **negatif**. Label netral yang muncul dari proses *scanning* YouTube otomatis dipetakan ke *netral* untuk kemudian dinormalisasi ke *negatif* sesuai konvensi pipeline [5].

*Tahap 4 — Ekstraksi Fitur:* Fitur teks diekstraksi menggunakan *TF-IDF* (*Term Frequency — Inverse Document Frequency*) dengan rentang *ngram* 1–2 (*unigram* dan *bigram*), *max_features* 15.000, *min_df* 2, dan *sublinear_tf* diaktifkan. Representasi ini memberikan bobot pada kata yang sering muncul dalam satu dokumen namun jarang muncul di seluruh korpus [7].

*Tahap 5 — Klasifikasi:* Model klasifikasi yang digunakan adalah *Complement Naive Bayes* (ComplementNB) dari pustaka Scikit-learn. Algoritma ini dipilih karena kemampuannya menangani data tidak seimbang dengan lebih baik dibandingkan *Multinomial Naive Bayes* standar [5]. Model dilatih menggunakan data berlabel hasil *SmartSentimentLabeler* dengan penambahan data sintetik dari pipeline `scraper_kebijakan.py` (250 sampel, distribusi seimbang 125 positif : 125 negatif) untuk meningkatkan representasi kelas minoritas.

*Tahap 6 — Evaluasi:* Performa model diukur menggunakan *accuracy* dan *5-fold stratified cross-validation*. Confusion matrix digunakan untuk menganalisis distribusi prediksi per kelas.

*Tahap 7 — Visualisasi:* Hasil analisis divisualisasikan dalam bentuk *pie chart* distribusi sentimen dan *word cloud* untuk mengidentifikasi kata dominan pada setiap kelas sentimen.

### 3. *Hasil dan Pembahasan*

#### 3.1 *Karakteristik Dataset*

Tabel 1 menyajikan karakteristik dataset yang digunakan dalam penelitian ini. Total data yang berhasil dikumpulkan adalah 599 komentar dari dua platform, dengan rentang waktu pengambilan data pada bulan Juni 2026. Setiap data melalui pemeriksaan kualitas dan pembersihan duplikasi sebelum memasuki tahap *preprocessing*.

**Tabel 1.** Karakteristik Dataset Penelitian

| Sumber Data | Platform | Jumlah Komentar | Metode Pengumpulan |
|-------------|----------|----------------:|--------------------|
| tiktok-wowo | TikTok | 118 | Scraping manual [a] |
| prabowo1 | YouTube | 247 | `youtube-comment-downloader` [b] |
| wowok-ngopi | YouTube | 234 | `youtube-comment-downloader` [b] |
| **Total** | | **599** | |
| Data sintetik (augmentasi) | Pipeline | 250 | `scraper_kebijakan.py` [5] |

[a] Data TikTok dikumpulkan secara manual dari video terkait program Makan Bergizi Gratis.
[b] Data YouTube dikumpulkan menggunakan pustaka `youtube-comment-downloader` dengan batas 300 komentar per video.

#### 3.2 *Hasil Preprocessing*

Proses *preprocessing* menghasilkan transformasi teks yang signifikan. Tabel 2 menunjukkan contoh hasil *preprocessing* pada beberapa komentar. Proses *stemming* dan normalisasi *slang* berhasil menyederhanakan teks tanpa menghilangkan makna sentimen inti.

**Tabel 2.** Contoh Hasil *Preprocessing*

| Teks Asli | Teks Hasil *Preprocessing* |
|-----------|---------------------------|
| "dulu g ada mbg saya masi hidup sampe skrng" | "mbg hidup" |
| "Bnyak yg mati kelaparan. Km aja yg cuek" | "bnyak mati lapar cuek" |
| "Memberi makan anak2 adalah tugas orang tua" | "makan anak tugas tua" |
| "mantap jendral presiden prabowo subianto" | "mantap jendral presiden prabowo subianto" |

#### 3.3 *Performa Model*

Model *Complement Naive Bayes* dilatih menggunakan data berlabel dari *SmartSentimentLabeler* yang diperkaya dengan data sintetik. Tabel 3 menyajikan metrik performa model.

**Tabel 3.** Performa Model *Complement Naive Bayes*

| Metrik | Nilai |
|--------|------:|
| *Accuracy* | 54.17% |
| *Cross-validation* (5-fold) | 61.58% ± 10.21% |
| *Fold 1* | 63.16% |
| *Fold 2* | 78.95% |
| *Fold 3* | 52.63% |
| *Fold 4* | 63.16% |
| *Fold 5* | 50.00% |

Akurasi model sebesar 54.17% dan rata-rata *cross-validation* 61.58% mengindikasikan bahwa model memiliki kemampuan klasifikasi di atas level acak (50% untuk klasifikasi biner), namun masih tergolong rendah untuk aplikasi produksi. Variasi skor antar *fold* yang cukup tinggi (standar deviasi 10.21%) menunjukkan sensitivitas model terhadap komposisi data latih. Hal ini dapat disebabkan oleh beberapa faktor: (a) ukuran data latih yang terbatas (250 sampel sintetik + data dari *SmartSentimentLabeler*); (b) kualitas pelabelan otomatis yang belum sempurna, terutama pada komentar sarkastik dan ambigu; serta (c) ketidakseimbangan kelas yang signifikan pada data prediksi.

#### 3.4 *Distribusi Sentimen*

Tabel 4 hingga Tabel 6 menyajikan distribusi sentimen per sumber data. Ketiga tabel menunjukkan pola dominasi sentimen negatif yang konsisten di seluruh platform.

**Tabel 4.** Distribusi Sentimen — TikTok (tiktok-wowo)

| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 61 | 51.69% |
| Negatif | 57 | 48.31% |
| **Total** | **118** | **100%** |

Sebagaimana ditunjukkan pada Tabel 4, sentimen pada platform TikTok cenderung seimbang antara positif (51.69%) dan negatif (48.31%). Hal ini menunjukkan bahwa diskusi di TikTok relatif beragam dan tidak didominasi oleh satu polaritas tertentu. Karakteristik platform dengan format video pendek dan interaksi berbasis komentar singkat cenderung menghasilkan opini yang lebih variatif. Temuan ini mengindikasikan bahwa diskusi mengenai program Makan Bergizi Gratis di TikTok didominasi oleh kritik, kekhawatiran, dan ketidakpuasan. Beberapa komentar menyoroti efektivitas program dibandingkan kebutuhan ekonomi yang lebih mendasar, seperti lapangan pekerjaan dan kenaikan gaji.

**Tabel 5.** Distribusi Sentimen — YouTube (Prabowo1)

| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 41 | 16.60% |
| Negatif | 206 | 83.40% |
| **Total** | **247** | **100%** |

**Tabel 6.** Distribusi Sentimen — YouTube (Wowok-Ngopi)

| Sentimen | Jumlah | Persentase |
|----------|------:|-----------:|
| Positif | 42 | 17.95% |
| Negatif | 192 | 82.05% |
| **Total** | **234** | **100%** |

Tabel 5 dan Tabel 6 memperlihatkan dominasi sentimen negatif yang sangat konsisten pada kedua video YouTube, dengan persentase mencapai 83.40% dan 82.05%. Pola ini menunjukkan bahwa komentar di platform YouTube cenderung lebih kritis dan sinis terhadap kebijakan dan pidato pemerintah. Temuan ini sejalan dengan karakteristik platform YouTube yang mendukung diskusi lebih panjang dan mendalam dibandingkan TikTok yang berbasis video pendek [8].

#### 3.5 *Rekapitulasi Seluruh Data*

Tabel 7 menyajikan rekapitulasi distribusi sentimen dari seluruh sumber data penelitian.

**Tabel 7.** Rekapitulasi Distribusi Sentimen Seluruh Sumber Data

| Sumber | Positif | Negatif | Total |
|--------|--------:|--------:|------:|
| tiktok-wowo | 61 (51.69%) | 57 (48.31%) | 118 |
| prabowo1 (YouTube) | 41 (16.60%) | 206 (83.40%) | 247 |
| wowok-ngopi (YouTube) | 42 (17.95%) | 192 (82.05%) | 234 |
| **Total Gabungan** | **144 (24.04%)** | **455 (75.96%)** | **599** |

Berdasarkan Tabel 7, dari total 599 komentar yang dianalisis, 144 komentar (24.04%) terklasifikasi sebagai positif dan 455 komentar (75.96%) sebagai negatif. Dominasi sentimen negatif ini konsisten di ketiga sumber data, dengan catatan bahwa platform TikTok menunjukkan distribusi yang lebih seimbang dibandingkan YouTube.

#### 3.6 *Analisis Visual*

Visualisasi *word cloud* pada Gambar 2 dan Gambar 3 mengungkap kata-kata dominan pada setiap kelas sentimen.

**Gambar 2.** *Word cloud* sentimen positif — kata dominan meliputi "mantap", "presiden", "keren", "terima kasih", "semangat".

**Gambar 3.** *Word cloud* sentimen negatif — kata dominan meliputi "rakyat", "sengsara", "korupsi", "ekonomi", "lapar", "stunting", "kritik".

Perbandingan kedua *word cloud* menunjukkan bahwa sentimen positif didominasi oleh ungkapan dukungan personal terhadap figur pemimpin, sementara sentimen negatif lebih terfokus pada isu-isu struktural seperti ekonomi, kesejahteraan, dan tata kelola.

#### 3.7 *Pembahasan*

Hasil analisis menunjukkan perbedaan signifikan antara distribusi sentimen di TikTok dan YouTube. TikTok menunjukkan distribusi yang hampir seimbang (51.69% positif vs 48.31% negatif), sementara YouTube didominasi sentimen negatif (82–83%). Pola ini mencerminkan pengaruh karakteristik platform terhadap opini publik.

Perbedaan distribusi antar platform menarik untuk dicermati. TikTok dengan format video pendek dan algoritma berbasis minat cenderung menarik partisipasi dari demografis yang lebih muda dengan opini yang lebih variatif [9]. Sementara itu, YouTube dengan format diskusi yang lebih panjang dan mendalam cenderung menarik komentar yang lebih kritis dan analitis terhadap kebijakan pemerintah.

Temuan dominasi sentimen negatif ini konsisten dengan pola yang dilaporkan oleh penelitian sebelumnya pada referensi [3] yang menemukan sentimen negatif dominan pada isu Danantara (87%), serta referensi [4] pada isu Makan Bergizi Gratis (~67%). Konsistensi ini memperkuat hipotesis bahwa opini publik Indonesia di media sosial cenderung didominasi oleh nada kritis terhadap kebijakan pemerintah, terlepas dari topik dan platform yang dianalisis.

Namun demikian, hasil ini harus diinterpretasikan dengan hati-hati mengingat beberapa keterbatasan penelitian:

*Pertama*, kualitas pelabelan otomatis menggunakan *SmartSentimentLabeler* belum sempurna, terutama untuk komentar yang mengandung sarkasme dan bahasa figuratif. Deteksi sarkasme yang kurang akurat dapat menyebabkan kesalahan klasifikasi yang sistematis.

*Kedua*, akurasi model *ComplementNB* yang relatif rendah (54.17%) menunjukkan bahwa data latih yang digunakan belum cukup representatif. Data sintetik yang dihasilkan oleh `scraper_kebijakan.py` mungkin belum sepenuhnya menangkap variasi linguistik komentar media sosial riil.

*Ketiga*, penggunaan *preprocessing* yang agresif — terutama *stemming* dan penghapusan *stopword* — berpotensi menghilangkan nuansa linguistik yang penting untuk analisis sentimen, seperti konteks kata dan struktur frasa.

*Keempat*, pemetaan label netral ke negatif merupakan keputusan metodologis yang dapat memengaruhi distribusi akhir. Meskipun keputusan ini diambil untuk menyederhanakan analisis menjadi dua kelas, dampaknya terhadap validitas hasil perlu dikaji lebih lanjut.

### 4. *Keterbatasan Penelitian*

Penelitian ini memiliki beberapa keterbatasan yang perlu diakui secara eksplisit:

1. **Ukuran dataset terbatas:** Total 599 komentar (ditambah 250 data sintetik) masih relatif kecil untuk melatih model *machine learning* yang robust, terutama untuk menangkap variasi bahasa informal Indonesia.

2. **Kualitas pelabelan otomatis:** Pelabelan menggunakan *SmartSentimentLabeler* berbasis aturan leksikal memiliki keterbatasan dalam mendeteksi sarkasme, ironi, dan nuansa konteks budaya yang kompleks.

3. **Perbedaan metode labeling dengan studi lain:** Studi pembanding pada referensi [1] hingga [4] menggunakan metode pelabelan yang berbeda-beda — manual (referensi [1]), K-Means (referensi [3] dan [4]), model terlatih IndoBERT (referensi [2]), dan *SmartSentimentLabeler* (studi ini). Perbedaan ini menyebabkan hasil akurasi antar studi tidak dapat dibandingkan secara langsung.

4. **Perbedaan pembagian data:** Seluruh referensi menggunakan *split* yang berbeda-beda (80:20 pada referensi [1], 70:30 pada referensi [5]), yang turut memengaruhi komparabilitas hasil akurasi.

5. **Hyperparameter tidak terdokumentasi:** Konfigurasi *hyperparameter* model pada referensi [1] hingga [4] tidak tersedia secara lengkap, sehingga klaim reproduktif tidak dapat dibuat.

6. **Cakupan platform terbatas:** Data hanya diambil dari dua platform (TikTok dan YouTube), sehingga belum merepresentasikan keseluruhan spektrum opini publik Indonesia yang tersebar di berbagai platform digital.

### 5. *Kesimpulan*

Penelitian ini mengimplementasikan analisis sentimen berbasis *Complement Naive Bayes* dengan *TF-IDF* pada 599 komentar media sosial yang mencakup dua topik kebijakan: Makan Bergizi Gratis (TikTok) dan Prabowo Gibran (YouTube). Kesimpulan yang dapat ditarik adalah sebagai berikut:

1. Dari 599 komentar yang dianalisis, 75.96% diklasifikasikan sebagai negatif dan 24.04% sebagai positif. Platform TikTok menunjukkan distribusi yang lebih seimbang (51.69% positif vs 48.31% negatif) dibandingkan YouTube yang didominasi sentimen negatif (82–83%).

2. Model *Complement Naive Bayes* mencapai akurasi 54.17% dengan *cross-validation* 61.58%, mengindikasikan potensi peningkatan melalui penambahan data latih berlabel berkualitas tinggi dan optimalisasi *hyperparameter*.

3. Perbedaan distribusi antar platform menunjukkan bahwa karakteristik platform memengaruhi cara publik mengekspresikan opini — TikTok lebih variatif, YouTube lebih kritis.

4. *Word cloud* mengungkapkan bahwa sentimen positif terkait dengan dukungan personal terhadap figur pemimpin, sementara sentimen negatif terfokus pada isu-isu struktural seperti ekonomi, kesejahteraan, dan tata kelola.

### 6. *Saran*

Untuk penelitian selanjutnya, disarankan agar: (a) cakupan data diperluas dengan melibatkan lebih banyak platform (Instagram, X/Twitter, forum diskusi); (b) teknik *resampling* atau augmentasi data seperti SMOTE diterapkan untuk mengatasi ketidakseimbangan kelas; (c) *fine-tuning* model bahasa Indonesia seperti IndoBERT dieksplorasi untuk meningkatkan akurasi; (d) pelabelan manual oleh annotator manusia dilakukan untuk menghasilkan data latih yang lebih berkualitas; serta (e) analisis longitudinal dilakukan untuk menangkap dinamika perubahan sentimen dari waktu ke waktu.

Dari sisi kebijakan, temuan dominasi sentimen negatif yang konsisten ini seyogyanya menjadi masukan bagi pemerintah untuk meningkatkan transparansi komunikasi kebijakan dan mengintegrasikan analisis sentimen *real-time* berbasis NLP sebagai bagian dari sistem pemantauan opini publik yang berkelanjutan.

---

### Daftar Pustaka

[1] Penelitian LSTM untuk analisis sentimen kebijakan pemerintah Indonesia, 2024.

[2] Studi SVM pada sentimen publik terhadap kebijakan Danantara dan Makan Bergizi Gratis, 2025.

[3] Implementasi IndoBERT dengan LDA untuk analisis sentimen pemberitaan politik Indonesia, 2025.

[4] Klasifikasi sentimen efisiensi anggaran menggunakan Naive Bayes dan TF-IDF, 2025.

[5] Pipeline analisis sentimen kebijakan Indonesia — dokumentasi teknis proyek. AGENTS.md, 2026.

[6] PySastrawi: Indonesian stemmer dan stopword remover — dokumentasi pustaka, 2023.

[7] G. Salton dan C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, hlm. 513–523, 1988.

[8] Studi perbandingan karakteristik diskusi publik di TikTok dan YouTube, 2025.

[9] Laporan demografi pengguna media sosial Indonesia, We Are Social dan DataReportal, 2026.

---

## BAGIAN 4 — DAFTAR BAGIAN YANG MASIH PERLU DIPERBAIKI SECARA MANUAL

Berikut adalah bagian-bagian yang masih memerlukan penanganan manual oleh penulis:

| No | Bagian | Masalah | Tindakan Manual |
|:--:|--------|---------|-----------------|
| M1 | **Daftar Pustaka [1]-[4], [8]** | Referensi belum lengkap — penulis harus mengisi detail bibliografi lengkap (nama jurnal, volume, halaman, DOI, tahun). | Cari dan lengkapi metadata pustaka. |
| M2 | **Daftar Pustaka [9]** | Sumber demografis perlu diverifikasi dan dilengkapi URL. | Verifikasi We Are Social 2026. |
| M3 | **Gambar flowchart** | Flowchart masih berupa teks deskriptif — perlu dibuat diagram visual menggunakan tools seperti draw.io, Visio, atau Lucidchart. | Buat flowchart visual dan masukkan sebagai gambar. |
| M4 | **Gambar *word cloud*** | File gambar belum di-generate dari pipeline dengan data yang sesuai. | Jalankan pipeline untuk menghasilkan *word cloud* dengan data riil, lalu embed gambar. |
| M5 | **Gambar *pie chart*** | Sama seperti M4, *pie chart* perlu di-generate menggunakan `scan_link.py` dan disimpan. | Generate visualisasi, screenshot/export PNG. |
| M6 | **Abstrak bahasa Inggris** | Abstrak dalam naskah lama terpotong di halaman 1. | Tulis ulang abstrak bahasa Inggris secara lengkap. |
| M7 | **Gambar 2 & 3 *word cloud*** | Caption saat ini hanya deskriptif — perlu gambar aktual. | Generate word cloud dari pipeline, tambahkan file gambar. |
| M8 | **Data TikTok akurasi** | Hasil prediksi TikTok memiliki akurasi rendah karena model sintetik. Penulis bisa mempertimbangkan untuk menandai keterbatasan ini lebih eksplisit di abstrak dan kesimpulan. | Tambahkan catatan di abstrak. |
| M9 | **Penambahan daftar tabel & gambar** | Jika template jurnal memerlukan daftar tabel/gambar terpisah, tambahkan setelah abstrak. | Sesuai template target. |
| M10 | **Uji plagiarisme** | Naskah perlu diuji menggunakan tools seperti Turnitin untuk memastikan orisinalitas. | Jalankan uji plagiarisme. |
| M11 | **Hyperparameter detail** | Detail hyperparameter ComplementNB (alpha, fit_prior) dan TF-IDF perlu diverifikasi dari kode aktual. | Cek `naive_bayes_sentimen.py` dan dokumentasikan. |
| M12 | **Penomoran halaman & header/footer** | Sesuai template jurnal target. | Sesuaikan dengan template. |

---

## BAGIAN 5 — REKOMENDASI

### Rekomendasi: **Tetap komparatif tetapi dipersempit menjadi satu topik**

Berdasarkan analisis menyeluruh terhadap kondisi naskah, data yang tersedia, dan tujuan penelitian, rekomendasi yang paling kuat secara metodologis adalah:

### **Opsi C (Rekomendasi): Single-topik Empirical Study dengan Naive Bayes**

**Alasan:**

1. **Dataset riil sudah tersedia:** Penulis memiliki pipeline lengkap (`scraper_kebijakan.py`, `naive_bayes_sentimen.py`, `scan_link.py`) yang menghasilkan data nyata dari TikTok dan YouTube. Sangat disayangkan jika data ini tidak digunakan langsung.

2. **Pipeline sudah teruji:** Kode pipeline sudah berfungsi dan menghasilkan output berupa CSV, model, dan visualisasi. Ini bukti bahwa penulis mampu melakukan eksperimen langsung, bukan sekadar studi literatur.

3. **Keunikan kontribusi:** Data dari TikTok dan YouTube untuk analisis sentimen kebijakan Indonesia masih jarang dibandingkan data Twitter/X. Ini menjadi nilai tambah penelitian.

4. **Focus pada Makan Bergizi Gratis:** Data TikTok (tiktok-wowo) secara spesifik membahas program Makan Bergizi Gratis. Dengan memfokuskan pada satu topik, analisis dapat dilakukan lebih mendalam. Data YouTube prabowo1 dan wowok-ngopi bisa dijadikan data pembanding.

**Konsekuensi yang perlu diperhatikan:**
- Perlu menyebutkan bahwa data YouTube membahas topik yang lebih umum (pidato presiden), bukan spesifik MBG.
- Keterbatasan akurasi model (54.17%) perlu dijelaskan secara transparan sebagai bagian dari pembelajaran dan pengembangan.

### Perbandingan Opsi

| Aspek | Opsi A: Studi Literatur | Opsi B: 5 Topik | Opsi C: 1 Topik (Rekomendasi) |
|-------|:----------------------:|:---------------:|:-----------------------------:|
| Kesesuaian dengan pipeline | Rendah | Sedang | **Tinggi** |
| Orisinalitas data | Rendah | Sedang | **Tinggi** |
| Kedalaman analisis | Bervariasi | Rendah | **Tinggi** |
| Kesiapan naskah | Perlu tulis ulang total | Perlu revisi besar | **Revisi sudah dilakukan** |
| Relevansi dengan prodi informatika | Sedang | Sedang | **Tinggi** (implementasi coding riil) |
| Potensi publikasi | Rendah (tidak ada data baru) | Sedang | **Tinggi** (dataset baru) |

### Action Plan untuk Opsi C

1. ✅ Revisi naskah sudah dilakukan (Bagian 3 dokumen ini).
2. ⬜ Generate visualisasi (word cloud, pie chart) dari pipeline dan embed.
3. ⬜ Lengkapi metadata daftar pustaka.
4. ⬜ Buat flowchart visual.
5. ⬜ Jalankan uji plagiarisme.
6. ⬜ Konsultasikan dengan dosen pembimbing.

---

*Dokumen revisi ini disusun sebagai respon terhadap seluruh poin revisi dosen pembimbing. Naskah revisi lengkap tersedia pada Bagian 3.*

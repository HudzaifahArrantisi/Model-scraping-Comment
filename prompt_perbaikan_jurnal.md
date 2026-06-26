# PROMPT UNTUK CLAUDE — Perbaikan Jurnal Ilmiah

> **Cara pakai:**
> 1. Upload file `Karya ilmiah_ HudzaifahArrantisi.pdf` (naskah lama yang salah) ke Claude
> 2. Upload file `referensi_data_jurnal.md` (panduan data benar) ke Claude  
> 3. Upload file `revisi_jurnal.md` (panduan format revisi) ke Claude
> 4. Salin prompt di bawah ini dan kirim ke Claude

---

## ───── MULAI PROMPT ─────

Saya memiliki 3 file yang sudah saya upload:

1. **`Karya ilmiah_ HudzaifahArrantisi.pdf`** — Naskah jurnal LAMA saya yang isinya SALAH di banyak bagian.
2. **`referensi_data_jurnal.md`** — Data ASLI dari proyek pipeline saya. **WAJIB** dipakai sebagai sumber data SATU-SATUNYA.
3. **`revisi_jurnal.md`** — Panduan revisi dari dosen.

### TUGAS ANDA:

Tulis ulang naskah jurnal ini menjadi versi yang **BENAR** dengan ketentuan berikut:

---

### A. DATA YANG HARUS DIPAKAI (WAJIB — dari referensi_data_jurnal.md)

**Dataset:** 599 komentar
| Sumber | Platform | Jumlah |
|--------|----------|------:|
| tiktok-wowo | TikTok | 118 |
| prabowo1 | YouTube | 247 |
| wowok-ngopi | YouTube | 234 |
| **Total** | | **599** |

**Distribusi Sentimen ASLI:**
| Sumber | Positif | Negatif |
|--------|--------:|--------:|
| TikTok | 61 (51.69%) | 57 (48.31%) |
| YouTube Prabowo | 41 (16.60%) | 206 (83.40%) |
| YouTube MBG | 42 (17.95%) | 192 (82.05%) |
| **Total** | **144 (24.04%)** | **455 (75.96%)** |

---

### B. MODEL YANG DIGUNAKAN — INI PENTING!

Naskah lama menggunakan **SGD Logistic Regression**, itu **SALAH**. 
Ganti dengan yang benar sesuai pipeline:

1. **Complement Naive Bayes (ComplementNB)** — untuk preprocessing dan pelabelan awal data TikTok
   - TF-IDF: `ngram_range=(1,2)`, `max_features=15000`, `analyzer=word`
2. **SGD Logistic Regression (sgd_log)** — untuk scanning YouTube (melalui `scan_link.py`)
   - TF-IDF: `ngram_range=(2,5)`, `max_features=30000`, `analyzer=char_wb`
   - Akurasi: **86.27%**, Cross-validation: **86.63% ± 2.49%**

Performa kedua model harus disajikan secara jujur dan deskriptif.

---

### C. FORMAT PENULISAN (WAJIB — dari revisi_jurnal.md)

1. **Judul:** "Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah Indonesia Menggunakan Pendekatan *Natural Language Processing*: Metode *Naive Bayes*"
2. **Subbab:** *italic* (bukan bold) — " *1. Pendahuluan* "
3. **Istilah asing:** *italic* — *Natural Language Processing*, *preprocessing*, *stemming*, *stopword*, *TF-IDF*, *accuracy*, *cross-validation*, dll.
4. **Caption tabel:** TIDAK bold, TIDAK italic — "Tabel 1. Karakteristik Dataset"
5. **Caption gambar:** TIDAK bold — "Gambar 1. Flowchart tahapan penelitian"
6. **Sitasi:** Format IEEE [1], [2], [3] — JANGAN sebut nama peneliti dalam narasi
7. **Tidak ada kolom "Sumber" terpisah** di tabel — sumber ditulis dalam sel dengan [1], [2]
8. **Tidak ada klaim superioritas absolut** — interpretasi deskriptif saja
9. **Keterbatasan** jadi section terpisah (bukan sub-bab dari pembahasan)
10. **Daftar Pustaka:** Gunakan referensi dari `referensi_data_jurnal.md` bagian 9

---

### D. STRUCTURE OUTPUT

Beri saya naskah LENGKAP (bukan outline) dengan struktur:

1. **Judul + Abstrak** (Indonesia + Inggris)
2. **Kata Kunci**
3. ***Pendahuluan*** — latar belakang, penelitian terkait [1]-[5], fokus penelitian
4. ***Metode Penelitian*** — KDD 7 tahap + deskripsi pipeline
5. ***Hasil dan Pembahasan*** — semua data dari bagian 2 dan 3 di atas
   - Tabel dataset, preprocessing, performa model, distribusi per sumber + rekapitulasi
   - **Sebut** bahwa pie chart ada di Gambar 2 (Prabowo) dan Gambar 3 (MBG)
   - **Sebut** bahwa word cloud ada di Gambar 4 (positif) dan Gambar 5 (negatif)
6. ***Keterbatasan Penelitian*** — 5-6 poin eksplisit
7. ***Kesimpulan*** — 4-5 poin berdasarkan data asli
8. ***Saran*** — akademik + praktis
9. **Daftar Pustaka** — [1] sampai [8]

---

### E. HAL-HAL YANG HARUS DIHINDARI

- ❌ JANGAN pakai angka/data dari naskah PDF lama (semua data di PDF lama SALAH)
- ❌ JANGAN klaim SGD Logistic Regression sebagai "satu-satunya model" atau "model utama"
- ❌ JANGAN sebut nama peneliti dalam narasi (contoh: "Ishak dkk. menyatakan..." → SALAH)
- ❌ JANGAN buat tabel dengan kolom "Sumber" terpisah
- ❌ JANGAN tulis subbab dengan format bold

### F. CONTOH KALIMAT YANG BENAR

✅ "Data dikumpulkan dari dua platform: TikTok (118 komentar) dan YouTube (481 komentar) dengan total 599 dokumen teks."
✅ "Penelitian sebelumnya [1] menerapkan analisis sentimen pada topik Danantara menggunakan SVM dan mencapai akurasi 85%."
✅ "Model ComplementNB mencapai akurasi 54.17%, sedangkan SGD Logistic Regression dengan character n-grams mencapai 86.27%."

### G. CONTOH KALIMAT YANG SALAH

❌ "Ishak dkk. menerapkan analisis sentimen..." (sebut nama)
❌ "Model SGD Logistic Regression jauh melampaui model sebelumnya..." (klaim superioritas)
❌ "Data dikumpulkan sebanyak 6.223 dokumen..." (data fiktif)
❌ "Penelitian menggunakan 3 kelas sentimen..." (harusnya 2 kelas)

---

## ───── AKHIR PROMPT ─────

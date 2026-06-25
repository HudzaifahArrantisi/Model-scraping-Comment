# Analisis Sentimen Komentar Kebijakan Pemerintah

Project ini dipakai untuk:

- mengunduh / membaca komentar,
- memproses teks komentar bahasa Indonesia,
- mengklasifikasikan sentimen menjadi `positif`, `netral`, atau `negatif`,
- menyimpan hasil analisis ke CSV dan gambar,
- memakai model siap pakai tanpa training manual berulang.

## Inti Alur Sistem

Ada 2 alur utama.

### 1. Training model

Alur ini hanya perlu dijalankan saat Anda ingin membuat atau memperbarui model:

```text
CSV komentar -> naive_bayes_sentimen.py -> hasil/model_nb.pkl
```

Hasil training disimpan ke file model:

```text
./hasil/model_nb.pkl
```

Setelah file model ini sudah ada, user lain tidak perlu training ulang.

### 2. Scanning komentar YouTube

Alur ini dipakai untuk user biasa:

```text
Link YouTube -> scan_link.py -> ambil komentar -> preprocessing -> prediksi sentimen -> simpan hasil
```

Output scanning disimpan ke:

```text
./hasil_scanning/
./hasil_scanning/plots/
```

## Cara Pakai Singkat

### A. Jika ingin langsung pakai model

Kalau file `./hasil/model_nb.pkl` sudah ada, user cukup jalankan:

```powershell
python scan_link.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --limit 300
```

Script akan:

1. download komentar,
2. membersihkan teks,
3. membaca model dari `./hasil/model_nb.pkl`,
4. memberi prediksi sentimen,
5. menyimpan hasil CSV dan gambar.

### B. Jika ingin membuat / update model

Jalankan training dari dataset CSV:

```powershell
python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv
```

Setelah selesai, model baru akan tersimpan di:

```text
./hasil/model_nb.pkl
```

## File Penting

| File / Folder | Fungsi |
| --- | --- |
| `naive_bayes_sentimen.py` | Training model, prediksi CSV, dan analisis sentimen. |
| `scan_link.py` | Scan komentar dari link YouTube lalu prediksi sentimen. |
| `scraper_kebijakan.py` | Membuat dataset awal / sample komentar. |
| `hasil/model_nb.pkl` | Model AI siap pakai. Ini yang dipakai saat scanning. |
| `data_raw/*.csv` | Dataset mentah untuk training. |
| `hasil_scanning/` | Hasil scan komentar dan file output. |
| `hasil_scanning/plots/` | Gambar hasil analisis seperti pie chart dan wordcloud. |

## File Yang Boleh Dihapus

Kalau tujuan Anda hanya hemat storage lokal, file berikut aman dihapus setelah model sudah tersimpan:

```text
hasil_scanning/*.csv
hasil_scanning/plots/*
data_raw/*.csv
```

Yang jangan dihapus kalau ingin model tetap bisa dipakai:

```text
hasil/model_nb.pkl
```

## Alur Belajar Model

Model belajar saat training, bukan saat hanya scanning.

Jadi urutannya:

```text
scanning komentar -> hasil CSV -> koreksi / label jika perlu -> training ulang -> model_nb.pkl ter-update
```

Kalau CSV hasil scanning langsung dihapus tanpa training ulang, model tidak mendapat pengetahuan baru dari data itu.

## Instalasi

```powershell
pip install -r reqeuirments.txt
pip install youtube-comment-downloader
```

Jika memakai fitur tambahan tertentu, mungkin perlu paket lain seperti `wordcloud`, `PySastrawi`, atau `TikTokApi`.

## Contoh Output

Setelah scan YouTube, file yang umum dibuat:

```text
./hasil_scanning/nama_file_raw.csv
./hasil_scanning/nama_file.csv
./hasil_scanning/plots/nama_file_pie.png
./hasil_scanning/plots/nama_file_wordcloud_positif.png
./hasil_scanning/plots/nama_file_wordcloud_negatif.png
./hasil_scanning/plots/nama_file_wordcloud_netral.png
```

## Ringkasnya

- Kalau `model_nb.pkl` sudah ada, user tinggal scan.
- Kalau mau model lebih pintar, lakukan training ulang dari CSV.
- CSV dan gambar output boleh dihapus kalau sudah tidak dipakai.
- Model tetap aman selama `hasil/model_nb.pkl` tidak dihapus.


# Analisis Sentimen Komentar Kebijakan Pemerintah

Project ini digunakan untuk scraping komentar, membuat model sentimen, dan memfilter komentar menjadi `positif`, `netral`, atau `negatif` menggunakan Naive Bayes + TF-IDF.

Fokus utama project:

- Membaca komentar dari file CSV hasil scraping manual.
- Membuat model sentimen dari CSV.
- Scanning komentar YouTube dari link video.
- Menyimpan hasil analisis dalam bentuk CSV dan grafik.
- Memfilter hasil akhir hanya komentar `negatif` dan `positif`.

## Struktur File

| File | Fungsi |
| --- | --- |
| `naive_bayes_sentimen.py` | Script utama untuk analisis CSV, training model, prediksi sentimen, dan membuat grafik. |
| `scan_link.py` | Scanning komentar dari link YouTube, lalu prediksi sentimen memakai model yang sudah dibuat. |
| `scraper_kebijakan.py` | Membuat dataset awal/sample atau scraping berdasarkan topik dari Twitter/X dan TikTok. |
| `mbg-prabowo.csv` | Contoh CSV hasil scraping manual. File ini bukan file wajib, hanya contoh input. |
| `reqeuirments.txt` | Daftar dependency Python. |

## Instalasi

Jalankan dari folder project:

```powershell
pip install -r reqeuirments.txt
pip install youtube-comment-downloader
```

Jika memakai fitur scraping TikTok dari `scraper_kebijakan.py`, dependency tambahan mungkin diperlukan:

```powershell
pip install TikTokApi
python -m playwright install
```

## Konsep Alur Project

Ada 2 alur utama:

```text
CSV komentar manual -> naive_bayes_sentimen.py -> hasil sentimen + model
```

```text
Link YouTube -> scan_link.py -> komentar YouTube -> prediksi sentimen -> hasil scanning
```

Catatan penting: `scan_link.py` membutuhkan model yang sudah dibuat, yaitu:

```text
./hasil/model_nb.pkl
```

Model tersebut dibuat oleh `naive_bayes_sentimen.py`.

## Alur 1: Analisis Dari CSV Komentar Manual

Gunakan alur ini jika sudah punya file CSV hasil scraping manual.

Contoh format CSV:

```csv
comment_id,username,comment,like_count,comment_time
1,user_a,program ini bagus,120,2026-06-24
2,user_b,program ini cuma buang anggaran,89,2026-06-24
```

Yang penting ada kolom berisi teks komentar, misalnya `comment`.

Payload:

```powershell
python naive_bayes_sentimen.py --input .\mbg-prabowo.csv --text-col comment --no-stem
```

Jika nama file berbeda, ganti bagian `--input`:

```powershell
python naive_bayes_sentimen.py --input .\komentar_manual.csv --text-col comment --no-stem
```

Jika nama kolom komentar berbeda, ganti bagian `--text-col`.

Contoh jika kolomnya bernama `komentar`:

```powershell
python naive_bayes_sentimen.py --input .\komentar_manual.csv --text-col komentar --no-stem
```

Output:

```text
./hasil/laporan_sentimen.csv
./hasil/ringkasan_topik.csv
./hasil/model_nb.pkl
./hasil/plots/
```

## Alur 2: Scanning Komentar Dari Link YouTube

Gunakan alur ini jika punya link video YouTube.

Contoh link:

```text
https://www.youtube.com/watch?v=6oNtO5_kQxY
```

Langkah 1, buat model dulu dari CSV latihan:

```powershell
python naive_bayes_sentimen.py --input .\mbg-prabowo.csv --text-col comment --no-stem
```

Setelah berhasil, pastikan file ini ada:

```text
./hasil/model_nb.pkl
```

Langkah 2, scan komentar YouTube:

```powershell
python scan_link.py "https://www.youtube.com/watch?v=6oNtO5_kQxY" --limit 300
```

Saat script meminta nama file, isi misalnya:

```text
yt_mbg_prabowo
```

Output:

```text
./hasil_scanning/yt_mbg_prabowo_raw.csv
./hasil_scanning/yt_mbg_prabowo.csv
./hasil_scanning/plots/yt_mbg_prabowo_pie.png
./hasil_scanning/plots/yt_mbg_prabowo_wordcloud_negatif.png
./hasil_scanning/plots/yt_mbg_prabowo_wordcloud_positif.png
./hasil_scanning/plots/yt_mbg_prabowo_wordcloud_netral.png
```

## Filter Hanya Negatif dan Positif

Jika hasil berasal dari CSV manual, filternya dari folder `hasil`:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./hasil/laporan_sentimen.csv'); df=df[df['sentimen_pred'].isin(['negatif','positif'])]; df.to_csv('./hasil/laporan_sentimen_negatif_positif.csv', index=False, encoding='utf-8-sig')"
```

Output:

```text
./hasil/laporan_sentimen_negatif_positif.csv
```

Jika hasil berasal dari scan link YouTube, filternya dari folder `hasil_scanning`.

Contoh:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./hasil_scanning/yt_mbg_prabowo.csv'); df=df[df['sentimen_pred'].isin(['negatif','positif'])]; df.to_csv('./hasil_scanning/yt_mbg_prabowo_negatif_positif.csv', index=False, encoding='utf-8-sig')"
```

Output:

```text
./hasil_scanning/yt_mbg_prabowo_negatif_positif.csv
```

## Fungsi scraper_kebijakan.py

`scraper_kebijakan.py` dipakai untuk membuat dataset awal, bukan untuk membaca CSV manual yang sudah jadi.

Contoh membuat dataset sample:

```powershell
python scraper_kebijakan.py --sample --topik all --max 500
```

Contoh satu topik:

```powershell
python scraper_kebijakan.py --sample --topik makan_bergizi_gratis --max 500
```

Output:

```text
./data_raw/semua_topik.csv
./data_raw/makan_bergizi_gratis.csv
```

Setelah dataset dibuat, analisis dengan:

```powershell
python naive_bayes_sentimen.py --input .\data_raw\semua_topik.csv --no-stem
```

## Perbedaan Script

| Script | Dipakai Saat |
| --- | --- |
| `naive_bayes_sentimen.py` | Sudah punya CSV komentar dan ingin analisis/training model. |
| `scan_link.py` | Punya link YouTube dan ingin ambil komentar + prediksi sentimen. |
| `scraper_kebijakan.py` | Ingin membuat dataset awal/sample atau scraping berdasarkan topik. |

## Catatan Akurasi

Jika CSV tidak punya kolom `sentimen`, script akan membuat label otomatis memakai lexicon/kamus kata.

Kelemahan lexicon:

- Sulit membaca sindiran.
- Sulit membaca slang TikTok/YouTube.
- Komentar pendek sering dianggap `netral`.
- Hasil bisa tidak sesuai jika kata negatif/positif tidak ada di kamus.

Untuk hasil lebih akurat, tambahkan kolom `sentimen` manual pada CSV:

```csv
comment,sentimen
"dulu g ada mbg saya masi hidup sampe skrng",negatif
"program ini bagus buat anak sekolah",positif
"drunk text lagi",netral
```

Lalu jalankan:

```powershell
python naive_bayes_sentimen.py --input .\data_latih.csv --text-col comment --no-stem
```

Semakin banyak data berlabel yang benar, model akan semakin baik.

## Payload Cepat

Analisis CSV manual:

```powershell
python naive_bayes_sentimen.py --input .\komentar_manual.csv --text-col comment --no-stem
```

Scan YouTube:

```powershell
python naive_bayes_sentimen.py --input .\komentar_manual.csv --text-col comment --no-stem
python scan_link.py "https://www.youtube.com/watch?v=6oNtO5_kQxY" --limit 300
```

Filter negatif dan positif dari hasil CSV:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./hasil/laporan_sentimen.csv'); df=df[df['sentimen_pred'].isin(['negatif','positif'])]; df.to_csv('./hasil/laporan_sentimen_negatif_positif.csv', index=False, encoding='utf-8-sig')"
```

Filter negatif dan positif dari hasil YouTube:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./hasil_scanning/yt_mbg_prabowo.csv'); df=df[df['sentimen_pred'].isin(['negatif','positif'])]; df.to_csv('./hasil_scanning/yt_mbg_prabowo_negatif_positif.csv', index=False, encoding='utf-8-sig')"
```

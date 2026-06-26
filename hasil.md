# Ringkasan Hasil Filter Sentimen

## 1. Ringkasan Global

| Sumber | Metode | Total | Positif | Negatif |
|--------|--------|------:|--------:|--------:|
| tiktok-wowo.csv | `naive_bayes_sentimen.py` | 118 | ~28 (23.73%)* | ~90 (76.27%)* |
| prabowo1 (YouTube) | `scan_link.py` | 247 | 41 (16.60%) | 206 (83.40%) |
| wowok-ngopi (YouTube) | `scan_link.py` | 234 | 42 (17.95%) | 192 (82.05%) |
| **Total Gabungan** | | **599** | **~111 (18.53%)** | **~488 (81.47%)** |

> *TikTok prediksi tidak reliabel — lihat catatan di bawah.

---

## 2. Filter 1 — `naive_bayes_sentimen.py` (tiktok-wowo.csv)

Perintah:
```
python naive_bayes_sentimen.py --load-model ./hasil/model_nb.pkl --input .\tiktok-wowo.csv
```

**Distribusi (model synthetic):** 118 komentar — ~28 positif (23.73%), ~90 negatif (76.27%)

### ⚠️ PERINGATAN: Prediksi TikTok TIDAK AKURAT

Model dilatih dari **data sintetik** (`scraper_kebijakan.py --sample --max 250`) yang tidak merepresentasikan gaya bahasa TikTok asli. Masalah utama:

- **Sarkasme tidak terdeteksi:** Komentar seperti *"Memberi makan anak2 adalah tugas orang tua"* diprediksi **positif**, padahal itu kritik terselubung ke program MBG.
- **Ekspresi casual disalahartikan:** *"NAHH"*, *"trus solusinya apa"* diprediksi positif, padahal bernada skeptis/negatif.
- **Random tag diprediksi negatif:** *"@Raspatiwww | Versetile gakuat aku😭"* hanya tag teman, bukan sentimen negatif.

**Kesimpulan TikTok:** Untuk hasil akurat, diperlukan label manual pada sampel komentar TikTok, lalu retrain model dengan data riil tersebut.

---

## 3. Filter 2A — `scan_link.py` (prabowo1 YouTube)

Perintah:
```
python scan_link.py --url <youtube_url> --limit 300
```

**Distribusi:** 247 komentar — 41 positif (16.60%), 206 negatif (83.40%)

Contoh komentar **POSITIF**:
1. semangat bapak
2. keren pak presiden prabowo, terima kasih untuk ketulusan bapak
3. mantap jendral presiden prabowo subianto
4. indonesia ingin negara kuat, harus daulat pangan energi
5. tolong didengar keluhan kami

Contoh komentar **NEGATIF**:
1. sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara
2. dia pidato IHSG langsung ancurrrrr!!!! gila negara ini!!!
3. kesini cuma buat nonton ibu kesayangan kita semua bu sherly
4. stand up komedian favorite aku nich
5. pidato template ini lagi

---

## 4. Filter 2B — `scan_link.py` (wowok-ngopi YouTube)

Perintah:
```
python scan_link.py --url <youtube_url> --limit 300
```

**Distribusi:** 234 komentar — 42 positif (17.95%), 192 negatif (82.05%)

Contoh komentar **POSITIF**:
1. mantap jendral presiden prabowo subianto
2. keren sih, bapak ini pidato, kekayaan saya hilang karena IHSG lgsg trjun
3. ibu serli gubernur ku maluku utara ist the best
4. keren pak presiden prabowo, terima kasih
5. ngapain doain orang itu sehat

Contoh komentar **NEGATIF**:
1. sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara
2. dia pidato IHSG langsung ancurrrrr!!!! gila negara ini!!!
3. pidato template ini lagi
4. kesini cuma buat nonton ibu kesayangan kita semua bu sherly
5. petani dan nelayan hari ini adalah tni dan polri

---

## 5. Kesimpulan

- **YouTube (prabowo1 & wowok-ngopi):** Hasil **cukup reliable** — 83.40% dan 82.05% negatif. Komentar YouTube didominasi kritik dan sarkasme terhadap pidato/kebijakan pemerintah. Distribusi hampir identik antar kedua video.
- **TikTok (tiktok-wowo):** Hasil **tidak reliable** — model gagal memahami sarkasme dan bahasa casual TikTok. Data mentah tersedia untuk labeling manual jika ingin hasil akurat.
- **Total keseluruhan (dengan catatan):** ~81.47% negatif, ~18.53% positif — tapi angka ini bias karena data TikTok tidak akurat.

> **Rekomendasi:** Untuk hasil TikTok yang akurat, lakukan label manual pada ~50-100 komentar, lalu jalankan `python naive_bayes_sentimen.py --retrain-feedback --input ./data_raw/semua_topik.csv` agar model belajar dari data riil.

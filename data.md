# Ringkasan Hasil Filter Sentimen

## 1. Ringkasan Global

| Sumber | Metode | Total | Positif | Negatif |
|--------|--------|------:|--------:|--------:|
| tiktok-wowo.csv | `naive_bayes_sentimen.py` | 118 | 61 (51.69%) | 57 (48.31%) |
| prabowo1 (YouTube) | `scan_link.py` | 247 | 41 (16.60%) | 206 (83.40%) |
| wowok-ngopi (YouTube) | `scan_link.py` | 234 | 42 (17.95%) | 192 (82.05%) |
| **Total Gabungan** | | **599** | **144 (24.04%)** | **455 (75.96%)** |

---

## 2. Filter 1 — `naive_bayes_sentimen.py` (tiktok-wowo.csv)

Perintah:
```
python naive_bayes_sentimen.py --input .\tiktok-wowo.csv
```

**Distribusi:** 118 komentar — 61 positif (51.69%), 57 negatif (48.31%)

Contoh komentar **POSITIF**:
1. Memberi makan anak2 adalah tugas orang tua. Tugas negara adalah memastikan orangtuanya memiliki penghasilan yg layak
2. @Raspatiwww | Versetile gakuat aku
3. top komen
4. deym
5. ak aw seng kuliah ga oleh mbg, yo ga mati pak

Contoh komentar **NEGATIF**:
1. dulu g ada mbg saya masi hidup sampe skrng
2. Bnyak yg mati kelaparan. Km aja yg cuek nggap punya hati lihat bnyak org di luar sana sedang berjuang cri makan
3. gk gitu konsep ny dongo
4. tanya bapaknya yang cari nafkah. tugas negara itu MENCERDASKAN, dan ngasi lapangan kerja buat orang tuanya biar ga kelaparan. BUKAN NGASIH MAKAN
5. bagaimana murid yg kelaparan di rumahnya karna ekonomi?

---

## 3. Filter 2A — `scan_link.py` (prabowo1 YouTube)

Perintah:
```
python scan_link.py --url <youtube_url> --limit 300
```

**Distribusi:** 247 komentar — 41 positif (16.60%), 206 negatif (83.40%)

Contoh komentar **POSITIF**:
1. Semangat Bapak
2. Keren Pak Presiden Prabowo, terima kasih untuk ketulusan Bapak untuk selalu konsisten
3. Mantap Jendral Presiden Prabowo Subianto
4. Memang kalau Indonesia ingin menjadi Negara yang kuat, harus bisa berdaulat, sep...
5. TOLONG DI DENGARKAN KELUHAN KAMI. Yg setuju Like : Jika presiden ingin rakyat n...

Contoh komentar **NEGATIF**:
1. Sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara karena kebijakan anda dan kejahatan kroni2 anda
2. Dia pidato IHSG langsung ANCURRRRR!!!! GILA NEGARA INI!!!
3. Kesini cuma buat nonton ibu kesayangan kita semua bu sherly!! Yg lain di skip ajah
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
1. Mantap Jendral Presiden Prabowo Subianto
2. Keren sih, bapak ini pidato, kekayaan saya hilang karena IHSG lgsg trjun, hehehe
3. Ibu serli gubernur ku Maluku Utara ist the best
4. Keren Pak Presiden Prabowo, terima kasih untuk ketulusan Bapak untuk selalu kons...
5. @alitanjaya6142 ngapain doain orang itu sehat

Contoh komentar **NEGATIF**:
1. Sehat2 terus pak prabowo biar bisa melihat rakyat nya makin sengsara karena kebijakan anda dan kejahatan kroni2 anda
2. Dia pidato IHSG langsung ANCURRRRR!!!! GILA NEGARA INI!!!
3. pidato template ini lagi
4. Kesini cuma buat nonton ibu kesayangan kita semua bu sherly!! Yg lain di skip ajah
5. Petani dan nelayan hari ini adalah TNI dan Polri...

---

## 5. Kesimpulan

- **TikTok (tiktok-wowo):** Sentimen relatif seimbang (51.69% positif vs 48.31% negatif) — diskusi lebih variatif dan tidak semuanya negatif.
- **YouTube (prabowo1 & wowok-ngopi):** Sentimen sangat dominan **negatif (83.40% & 82.05%)** — komentar YouTube cenderung lebih kritis dan sinis terhadap kebijakan pemerintah.
- **Total keseluruhan:** Dari 599 komentar yang difilter, **75.96% negatif** dan **24.04% positif**.

> **Catatan:** Hasil filter dari `tiktok-wowo.csv` menggunakan model Naive Bayes yang sudah dilatih, sedangkan hasil YouTube menggunakan `scan_link.py` dengan model + fallback `SmartSentimentLabeler`. Perbedaan distribusi bisa dipengaruhi oleh platform dan topik video yang di-scan.

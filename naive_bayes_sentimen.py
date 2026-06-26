"""
==============================================================
SCRIPT 2: NAIVE BAYES SENTIMENT ANALYSIS
         Analisis Sentimen Kebijakan Pemerintah Indonesia
==============================================================
INPUT  : CSV dari scraper_kebijakan.py (./data_raw/*.csv)
         atau CSV manual dengan kolom 'text' dan 'topik'

OUTPUT :
  - ./hasil/laporan_sentimen.csv    -> hasil lengkap per baris
  - ./hasil/ringkasan_topik.csv     -> % positif/negatif/netral per topik
  - ./hasil/model_nb.pkl            -> model Naive Bayes tersimpan
  - ./hasil/plots/*.png             -> visualisasi bar chart & word cloud
  - ./hasil/laporan_analisis.json   -> laporan JSON lengkap

INSTALL:
  pip install pandas scikit-learn PySastrawi matplotlib seaborn wordcloud tqdm joblib

CARA PAKAI:
  # Dari CSV hasil scraping/generate:
  python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv

  # Pakai data sample (testing cepat):
  python naive_bayes_sentimen.py --sample

  # Prediksi 1 kalimat baru:
  python naive_bayes_sentimen.py --prediksi "danantara merugikan rakyat kecil"

  # Load model tersimpan + prediksi:
  python naive_bayes_sentimen.py --load-model ./hasil/model_nb.pkl \
    --prediksi "makan bergizi gratis program terbaik"
==============================================================
"""

import os
import re
import csv
import json
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

# Sastrawi (stemming bahasa Indonesia)
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    warnings.warn("[WARN] PySastrawi tidak terinstall. Stemming dinonaktifkan.\n"
                  "       Install: pip install PySastrawi")

warnings.filterwarnings("ignore")

os.makedirs("./hasil", exist_ok=True)
os.makedirs("./hasil/plots", exist_ok=True)


TEXT_COLUMN_CANDIDATES = [
    "text", "teks", "comment", "komentar", "caption", "content",
    "full_text", "tweet", "body", "message",
    "text_clean",  # fallback untuk feedback history CSV
]


def pilih_kolom_teks(df: pd.DataFrame, requested_col: str | None = None) -> str:
    """Pilih kolom teks dari CSV input."""
    if requested_col:
        if requested_col not in df.columns:
            raise ValueError(
                f"Kolom teks '{requested_col}' tidak ditemukan. "
                f"Kolom tersedia: {', '.join(df.columns)}"
            )
        return requested_col

    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        "Kolom teks tidak ditemukan otomatis. "
        "Gunakan --text-col NAMA_KOLOM. "
        f"Kolom tersedia: {', '.join(df.columns)}"
    )


# ─────────────────────────────────────────────
#  1. PREPROCESSING TEKS BAHASA INDONESIA
# ─────────────────────────────────────────────
class PreprocessorIndonesia:
    """
    Preprocessing teks bahasa Indonesia untuk analisis sentimen.
    Pipeline: clean -> normalize_slang -> remove_stopwords -> stem
    """

    # Kamus normalisasi slang/singkatan umum Twitter/TikTok (140+ kata)
    SLANG_DICT = {
        # Negasi
        "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak",
        "nggak": "tidak", "tdk": "tidak", "gbs": "tidak bisa", "gabisa": "tidak bisa",
        "gaada": "tidak ada", "gada": "tidak ada", "gamau": "tidak mau",
        "gabole": "tidak boleh", "gabakal": "tidak akan",
        "g": "tidak", "gda": "tidak ada", "gadapet": "tidak dapat", "gdpt": "tidak dapat",
        "gatau": "tidak tahu", "gtau": "tidak tahu",
        # Intensifier
        "bgt": "banget", "bngt": "banget", "bgtt": "banget",
        "bener2": "benar-benar", "bnr": "benar", "bnr2": "benar-benar",
        "bner": "benar", "sgt": "sangat", "amat": "sangat",
        # Pronoun
        "yg": "yang", "dg": "dengan", "dgn": "dengan", "utk": "untuk",
        "krn": "karena", "karna": "karena", "klo": "kalau", "kalo": "kalau",
        "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
        "blm": "belum", "blum": "belum", "msh": "masih", "masi": "masih",
        "sy": "saya", "gw": "saya", "gue": "saya", "ane": "saya",
        "lu": "kamu", "lo": "kamu", "kmu": "kamu", "elo": "kamu",
        "dr": "dari", "dri": "dari", "pd": "pada", "dlm": "dalam",
        "ttg": "tentang", "spt": "seperti", "sprt": "seperti",
        "sampe": "sampai", "skg": "sekarang", "skrng": "sekarang", "skr": "sekarang",
        "ortu": "orang tua", "ortunya": "orang tua", "jabat": "pejabat",
        "dll": "dan lain lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
        "hrs": "harus", "hrus": "harus", "musti": "harus",
        "dpt": "dapat", "bs": "bisa", "bsa": "bisa",
        "mo": "mau", "pengen": "ingin", "pgn": "ingin",
        "org": "orang", "org2": "orang-orang",
        "jg": "juga", "tp": "tapi", "ttp": "tetap",
        "emg": "memang", "emang": "memang", "memg": "memang",
        "gt": "begitu", "gitu": "begitu", "gini": "begini",
        "lg": "lagi", "nih": "ini", "tuh": "itu",
        "mkn": "makan", "mksd": "maksud", "mksud": "maksud",
        "tol0l": "tolol", "gobl0k": "goblok", "bgo": "bego", "bngst": "bangsat", "ajg": "anjing",
        "dongok": "dongo", "dunguk": "dungu", "sed0ngok": "dongo", "p3merintah": "pemerintah",
        # Kata terkait kebijakan
        "pemrintah": "pemerintah", "pmerintah": "pemerintah",
        "rkyt": "rakyat", "rkyat": "rakyat",
        "korup": "korupsi", "ngaco": "tidak benar",
        "prorakyat": "pro rakyat", "antikorupsi": "anti korupsi",
        # Kata slang umum
        "wkwk": "", "wkwkwk": "", "wkwkwkwk": "",
        "haha": "", "hahaha": "", "hehe": "", "hihi": "",
        "anjir": "", "anjay": "", "njir": "",
        "cuy": "", "bro": "", "sis": "", "gan": "",
        "btw": "", "fyi": "", "imho": "menurut saya",
        "tbh": "jujur", "smh": "",
        "no cap": "serius", "fr": "serius",
        # Emotikon teks
        ":)": "", ":(": "", ":D": "", "xD": "", ":'(": "",
    }

    # ── EMOJI ↔ SENTIMENT TOKEN ──
    EMOJI_MAP = {
        # Positif
        "👍": "positif_emoji", "🙏": "positif_emoji", "💪": "positif_emoji",
        "❤️": "positif_emoji", "😊": "positif_emoji", "😍": "positif_emoji",
        "🥰": "positif_emoji", "😁": "positif_emoji", "😂": "positif_emoji",
        "🤣": "positif_emoji", "🔥": "positif_emoji", "✨": "positif_emoji",
        "🌟": "positif_emoji", "🎉": "positif_emoji", "🇮🇩": "positif_emoji",
        "✅": "positif_emoji", "👏": "positif_emoji", "🙌": "positif_emoji",
        # Negatif
        "😡": "negatif_emoji", "😤": "negatif_emoji", "🤬": "negatif_emoji",
        "👎": "negatif_emoji", "😢": "negatif_emoji", "😭": "negatif_emoji",
        "💀": "negatif_emoji", "☠️": "negatif_emoji", "😠": "negatif_emoji",
        "🤮": "negatif_emoji", "😷": "negatif_emoji", "💔": "negatif_emoji",
        "❌": "negatif_emoji", "🚫": "negatif_emoji", "😞": "negatif_emoji",
        "😔": "negatif_emoji", "😒": "negatif_emoji", "🙄": "negatif_emoji",
        # Netral / tanya
        "❓": "netral_emoji", "⁉️": "netral_emoji", "🤔": "netral_emoji",
        "😐": "netral_emoji", "😑": "netral_emoji",
    }

    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming and SASTRAWI_AVAILABLE

        if self.use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()

        # Stopwords dari Sastrawi + tambahan manual
        if SASTRAWI_AVAILABLE:
            sw_factory = StopWordRemoverFactory()
            self.stopwords = set(sw_factory.get_stop_words())
        else:
            self.stopwords = set()

        # Tambahan stopwords khusus konteks kebijakan & sosmed
        extra_stopwords = {
            "ini", "itu", "yang", "dan", "di", "ke", "dari", "dengan",
            "untuk", "pada", "adalah", "ada", "juga", "sudah", "belum",
            "akan", "bisa", "harus", "bukan", "tidak", "tapi", "atau",
            "karena", "saat", "hari", "waktu", "tahun", "bulan",
            "katanya", "bilang", "kata", "kalau", "jika", "maka",
            "agar", "supaya", "tetapi", "namun", "meski", "walaupun",
            "kita", "kami", "mereka", "dia", "ia", "kamu", "saya",
            "http", "https", "www", "com", "co", "id", "twitter",
            "tiktok", "instagram", "rt", "via", "pic",
            "guys", "bro", "sis", "gan", "cuy", "deh", "dong", "sih",
            "aja", "doang", "banget", "nya", "lah", "kok", "ya",
        }
        self.stopwords.update(extra_stopwords)

    def clean_text(self, text: str) -> str:
        """Membersihkan teks dari noise."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)              # hapus URL
        text = re.sub(r"@\w+", "", text)                         # hapus mention
        text = re.sub(r"#(\w+)", r"\1", text)                    # hapus # tapi simpan kata

        # Ganti emoji dengan token sentimen (sebelum hapus tanda baca)
        for emoji, token in self.EMOJI_MAP.items():
            text = text.replace(emoji, f" {token} ")

        text = re.sub(r"[^\w\s]", " ", text)                     # hapus tanda baca sisa
        text = re.sub(r"\d+", "", text)                          # hapus angka
        text = re.sub(r"\b(\w)\1{2,}\b", r"\1", text)          # hapus huruf berulang (baguuus -> bagus)
        text = re.sub(r"\s+", " ", text).strip()                 # normalisasi spasi
        return text

    def normalize_slang(self, text: str) -> str:
        """Normalisasi kata slang/singkatan ke bentuk baku."""
        tokens = text.split()
        return " ".join(self.SLANG_DICT.get(t, t) for t in tokens)

    def remove_stopwords(self, text: str) -> str:
        """Hapus stopwords."""
        tokens = text.split()
        return " ".join(t for t in tokens if t not in self.stopwords and len(t) > 2)

    def stem(self, text: str) -> str:
        """Stemming menggunakan Sastrawi."""
        if not self.use_stemming:
            return text
        return self.stemmer.stem(text)

    def preprocess(self, text: str) -> str:
        """Pipeline preprocessing lengkap."""
        text = self.clean_text(text)
        text = self.normalize_slang(text)
        text = self.remove_stopwords(text)
        if self.use_stemming:
            text = self.stem(text)
        return text

    def preprocess_batch(self, texts: list) -> list:
        """Preprocessing batch dengan progress bar."""
        return [self.preprocess(t) for t in tqdm(texts, desc="  Preprocessing")]


# ─────────────────────────────────────────────
#  2. LABELING OTOMATIS CERDAS (Smart Reasoning)
#     Chain-of-Thought Sentiment Labeler
# ─────────────────────────────────────────────
class SmartSentimentLabeler:
    """
    Pelabelan sentimen CERDAS berbasis "thinking process" untuk data
    yang BELUM berlabel. Menggunakan logika reasoning multi-tahap:

    THINKING PROCESS:
    1. Identifikasi Subjek & Objek   → siapa/apa yang dikritik/dipuji?
    2. Pisahkan Fakta vs Emosi       → kritik membangun vs hujatan?
    3. Analisis Sarkasme & Budaya    → kata positif untuk sindiran?
    4. Bobot Dampak Akhir            → kalimat akhir lebih menentukan

    Fitur Cerdas:
    - Deteksi NEGASI  : "tidak bagus" → negatif (bukan positif)
    - Deteksi SARKASME: "hebat sekali, rakyat makin puasa" → negatif
    - Deteksi FRASA   : "tepat sasaran", "program gagal" → konteks utuh
    - Bobot POSISI    : kalimat di akhir teks punya bobot 1.5x
    - Deteksi KONTRADIKSI: positif + negatif bersamaan → analisis dominan
    """

    # ── KATA POSITIF (single-word) ──
    POSITIF_WORDS = {
        "bagus", "baik", "mantap", "dukung", "setuju", "bermanfaat",
        "berhasil", "sukses", "positif", "maju", "pro", "mendukung",
        "meningkat", "harapan", "solusi", "inovasi", "keren",
        "sejahtera", "makmur", "tepat", "efektif", "efisien",
        "transparan", "adil", "merata", "terlaksana", "terwujud",
        "apresiasi", "pujian", "terimakasih", "hebat",
        "wow", "amazing", "good", "best", "optimal", "brilian",
        "cerdas", "visioner", "progresif", "kompeten", "profesional",
        "berani", "tegas", "salut", "respect", "bangga", "proud",
        "senang", "gembira", "puas", "lega", "syukur", "alhamdulillah",
        "terbaik", "berkualitas", "unggul", "mumpuni", "andal",
        "membantu", "menolong", "memudahkan", "menyejahterakan",
        "lanjutkan", "semangat", "inspirasi", "mengagumkan",
        "stabil", "aman", "damai", "nyaman", "sehat", "kuat",
        "berterima kasih", "terbantu", "bersyukur", "mantul",
    }

    # ── KATA NEGATIF (single-word) ──
    NEGATIF_WORDS = {
        "buruk", "gagal", "tolak", "kecewa", "rugi", "korupsi",
        "bohong", "tipu", "palsu", "salah", "jelek", "parah",
        "menyedihkan", "merugikan", "diskriminasi", "ancam",
        "monopoli", "oligarki", "kapitalis", "melarat", "miskin",
        "lapar", "susah", "sulit", "masalah", "krisis", "darurat",
        "bencana", "chaos", "amburadul", "berantakan", "hancur",
        "bobrok", "busuk", "jahat", "ngawur", "bodoh",
        "pembohong", "penipu", "maling", "rampas", "rampok",
        "munafik", "otoriter", "nepotisme", "kolaps", "anjlok",
        "merosot", "memburuk", "mengancam", "menghancurkan",
        "korban", "sengsara", "menderita", "terancam", "terpuruk",
        "pengangguran", "inflasi", "utang", "defisit",
        "manipulasi", "penyelewengan", "penyalahgunaan", "korup",
        "berbahaya", "mengorbankan", "mengkhianati",
        "percuma", "sia-sia", "pencitraan", "keracunan", "basi",
        "sampah", "omong kosong", "pembodohan", "penipuan",
        "memperburuk", "memperparah", "menghambat", "menghalangi",
        "phk", "dipecat", "dirugikan", "ditindas", "dibungkam",
        # Medsos / kebijakan
        "tolol", "goblok", "bego", "bloon", "dongo", "dungu", "otak udang", "sok",
        "anjing", "ajg", "bangsat", "bngst", "nyet", "monyet", "kampret", "sialan", "babi",
        "stunting", "kelaparan", "mati", "meninggal", "tewas", "bunuh", "racun", "sakit",
        "malas", "males", "beban", "membebani", "mahal", "kemahalan", "potong", "dipotong",
        "emosi", "kesal", "marah", "benci", "suap", "menyuap", "disuap", "cacat",
        "bocil", "fomo", "penjilat", "penyepong", "buzzer", "buzzerp", "kritik", "hilang",
        "ilang", "tanya", "nyabu", "nyoli", "drunk"
    }

    # ── FRASA POSITIF (multi-word, konteks utuh) ──
    POSITIF_PHRASES = [
        "sangat membantu", "tepat sasaran", "pro rakyat", "luar biasa",
        "program bagus", "program terbaik", "kerja nyata", "bukti nyata",
        "anak senang", "makan gratis", "gizi terpenuhi", "semoga lancar",
        "dukung penuh", "terima kasih", "sangat bermanfaat",
        "langkah tepat", "langkah berani", "langkah progresif",
        "ekonomi tumbuh", "investasi meningkat", "harga turun",
        "harga stabil", "lapangan kerja", "daya saing",
        "angin segar", "harapan baru", "masa depan cerah",
        "rakyat sejahtera", "rakyat terbantu", "rakyat senang",
        "membantu siswa", "mengurangi stunting", "generasi sehat",
        "lebih baik", "sangat bagus", "sangat keren",
        "berkembang pesat", "tumbuh signifikan", "meningkat pesat",
    ]

    # ── FRASA NEGATIF (multi-word, konteks utuh) ──
    NEGATIF_PHRASES = [
        "program gagal", "tidak setuju", "tidak layak", "sakit perut",
        "buang anggaran", "hambur anggaran", "uang rakyat dipakai",
        "pencitraan belaka", "buat apa", "tidak perlu",
        "ga ada mbg", "gak ada mbg", "tidak ada mbg",
        "rakyat sengsara", "rakyat susah", "rakyat miskin",
        "harga naik", "harga sembako naik", "harga mahal",
        "korupsi merajalela", "korupsi makin", "rawan korupsi",
        "tanpa pengawasan", "tanpa transparansi", "tidak transparan",
        "gagal total", "cuma janji", "janji kosong", "janji palsu",
        "pro oligarki", "boneka oligarki", "mafia tanah",
        "phk massal", "phk dimana-mana", "dipecat massal",
        "potong dana", "anggaran dipotong", "subsidi dihapus",
        "kualitas buruk", "tidak bergizi", "makanan basi",
        "demo ditindas", "anti demokrasi", "anti kritik",
        "utang nambah", "utang naik", "rupiah melemah",
        "sama aja", "percuma saja", "sia-sia saja",
        # Kebijakan
        "bukan solusi", "bukan prioritas", "bukan ngasih makan", "bukan memberi makan",
        "bukan malah", "tidak mendidik", "ga mendidik", "gak mendidik",
        "mending kasih pancing", "mending kasih alat", "kasih pancing", "alat pancing",
        "bukan ngasih", "bukan memberi", "ngapain", "ga perlu", "gak perlu",
        "hambur uang", "buang uang", "pesta korupsi", "stunting naik",
        "tidak didengar", "tidak mendengar", "gak didengar", "ga didengar",
        "gimana sih", "gimana bisa", "kok bisa", "kok gitu", "turun drastis",
        "ganti presiden", "ganti program", "tidak sustain", "ga sustain", "gak sustain",
        "tidak tahu analogi", "ga tau analogi", "gatau analogi", "serba instan",
        "tidak pernah mikir", "ga pernah mikir", "gak pernah mikir", "tidak ada niatan",
        "tidak didengar", "tidak ada isi", "ga ada isi", "gak ada isi",
        "tidak dapet mbg", "ga dapet mbg", "gak dapet mbg",
        "mana paham", "tidak paham", "ga paham", "gak paham", "ngga paham",
        "dollar naik"
    ]

    # ── FRASA NETRAL (indikator teks netral) ──
    NETRAL_PHRASES = [
        "perlu evaluasi", "perlu dikaji", "perlu kajian",
        "belum bisa dinilai", "masih terlalu dini", "kita lihat dulu",
        "ada plus minusnya", "pro dan kontra", "pro kontra",
        "masih proses", "masih bertahap", "masih tahap awal",
        "perlu waktu", "menunggu hasil", "mari beri kesempatan",
        "perlu dipantau", "perlu diawasi", "perlu pengawasan",
        "belum ada data", "belum terlihat dampak",
        "perlu sosialisasi", "perlu penyesuaian",
        "wajar ada kendala", "masih dalam tahap",
    ]

    # ── POLA SARKASME (kata positif diikuti konteks negatif) ──
    SARKASME_PATTERNS = [
        # (trigger_positif, konteks_negatif_yang_mengikuti)
        ("hebat", ["rakyat puasa", "rakyat sengsara", "rakyat susah", "rakyat miskin",
                    "makin susah", "makin miskin", "gak bisa makan", "tidak bisa makan",
                    "untuk pejabat", "untuk oligarki", "buat korupsi", "buat konglomerat",
                    "miskinkan rakyat", "merugikan kecil", "menguntungkan besar"]),
        ("bagus", ["rakyat sengsara", "rakyat susah", "harga naik", "makin mahal",
                    "untuk pejabat", "buat korupsi", "buat konglomerat", "utang naik"]),
        ("mantap", ["rakyat sengsara", "phk", "harga naik", "korupsi",
                    "rakyat miskin", "utang naik", "inflasi", "menguntungkan besar"]),
        ("keren", ["rakyat sengsara", "rakyat susah", "harga naik",
                    "stunting naik", "miskinkan rakyat"]),
        ("luar biasa", ["rakyat sengsara", "rakyat susah", "korupsi", "gagal",
                        "utang naik", "inflasi", "harga naik"]),
        ("bravo", ["rakyat sengsara", "rakyat susah", "gagal", "korupsi"]),
        ("sukses", ["menghancurkan", "merusak", "merugikan", "memiskinkan",
                    "menguntungkan konglomerat", "menguntungkan oligarki"]),
        ("bagus sekali", ["rakyat sengsara", "rakyat susah", "harga naik", "stunting naik"]),
        ("mantap sekali", ["rakyat miskin", "utang naik", "harga naik", "phk"]),
        ("hebat sekali", ["miskinkan rakyat", "merugikan kecil", "menguntungkan besar"]),
        ("makan gratis", ["stunting naik", "makanan basi", "kualitas buruk", "tidak bergizi",
                          "hambur anggaran", "uang rakyat dipakai", "pencitraan"]),
        ("makan bergizi", ["stunting naik", "makanan basi", "kualitas buruk", "tidak bergizi",
                           "hambur anggaran", "pencitraan"]),
        ("program terbaik", ["stunting naik", "gagal", "korupsi", "hambur uang", "pencitraan"]),
        ("pro rakyat", ["oligarki", "konglomerat", "korupsi", "miskinkan", "merugikan kecil"]),
        ("semangat", ["harga naik", "susah hidup", "miskin", "utang", "inflasi"]),
    ]

    # ── KATA NEGASI (membalikkan sentimen kata berikutnya) ──
    NEGASI_WORDS = {
        "tidak", "gk", "ga", "gk", "ngga", "nggak", "bukan",
        "belum", "jangan", "tanpa", "tak", "enggak", "kagak",
        "bukanlah", "tidaklah", "takkan",
    }

    # ── KATA KONTRADIKSI/TRANSISI (menandai perubahan arah sentimen) ──
    KONTRADIKSI_WORDS = {
        "tapi", "tetapi", "namun", "sayangnya", "sayang",
        "meskipun", "walaupun", "walau", "kendati",
        "sementara", "padahal", "sebaliknya", "justru",
        "malah", "malahan", "nyatanya", "kenyataannya",
        "tp", "cuma", "hanya", "sekadar", "nggak lebih", "ga lebih", "gak lebih"
    }

    # ── EMOJI SARKASME (emoji positif di konteks negatif = sarkasme) ──
    SARKASME_EMOJI = {
        "😂", "🤣", "😭", "😹", "😆", "😅", "🙃", "😏", "😒", "🙄", "😑", "😐"
    }

    # ── EMOJI NEGATIF EKSPLISIT ──
    NEGATIF_EMOJI_EXPLICIT = {
        "😡", "😤", "🤬", "👎", "😢", "😭", "💀", "☠️", "😠",
        "🤮", "😷", "💔", "❌", "🚫", "😞", "😔", "😒", "🙄"
    }

    # ── EMOJI POSITIF EKSPLISIT ──
    POSITIF_EMOJI_EXPLICIT = {
        "👍", "🙏", "💪", "❤️", "😊", "😍", "🥰", "😁",
        "🔥", "✨", "🌟", "🎉", "🇮🇩", "✅", "👏", "🙌"
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _detect_sarcasm(self, text_lower: str, original_text: str = "") -> bool:
        """
        THINKING STEP 3: Analisis Sarkasme & Konteks Budaya
        Netizen Indonesia sering menggunakan kata positif untuk menyindir.
        """
        # Cek pola sarkasme berbasis kata
        for trigger, negative_contexts in self.SARKASME_PATTERNS:
            if trigger in text_lower:
                for context in negative_contexts:
                    if context in text_lower:
                        return True

        # Cek emoji sarkasme: emoji "positif" (😂🤣😭) di teks yg mengandung kata negatif
        has_negative_word = any(w in text_lower for w in self.NEGATIF_WORDS)
        has_negative_phrase = any(p in text_lower for p in self.NEGATIF_PHRASES)
        has_sarcasm_emoji = any(e in original_text for e in self.SARKASME_EMOJI)

        if has_sarcasm_emoji and (has_negative_word or has_negative_phrase):
            return True

        # Cek pola: kata positif + emoji sarkasme di akhir
        for trigger in [p[0] for p in self.SARKASME_PATTERNS]:
            if trigger in text_lower:
                for emoji in self.SARKASME_EMOJI:
                    if emoji in original_text:
                        # Cek apakah emoji muncul SETELAH kata trigger
                        trigger_idx = text_lower.find(trigger)
                        emoji_idx = original_text.find(emoji)
                        if emoji_idx > trigger_idx:
                            return True

        return False

    def _apply_negation(self, tokens: list) -> tuple:
        """
        THINKING STEP 2: Deteksi Negasi
        Jika ada kata negasi sebelum kata positif, maka menjadi negatif.
        """
        negated_pos = 0
        negated_neg = 0

        for i, token in enumerate(tokens):
            if token in self.NEGASI_WORDS and i + 1 < len(tokens):
                next_word = tokens[i + 1]
                if next_word in self.POSITIF_WORDS:
                    negated_pos += 1
                elif next_word in self.NEGATIF_WORDS:
                    negated_neg += 1
                if i + 2 < len(tokens):
                    second_word = tokens[i + 2]
                    if second_word in self.POSITIF_WORDS and next_word not in self.NEGATIF_WORDS:
                        negated_pos += 1
                    elif second_word in self.NEGATIF_WORDS and next_word not in self.POSITIF_WORDS:
                        negated_neg += 1

        return negated_pos, negated_neg

    def _score_with_position_weight(self, text_lower: str, original_text: str = "") -> tuple:
        """
        THINKING STEP 4: Bobot Dampak Akhir (Net Sentiment Impact)
        """
        segments = re.split(
            r'\b(?:' + '|'.join(re.escape(w) for w in self.KONTRADIKSI_WORDS) + r')\b',
            text_lower
        )

        total_pos = 0.0
        total_neg = 0.0

        num_segments = len(segments)
        for idx, segment in enumerate(segments):
            weight = 1.5 if idx == num_segments - 1 and num_segments > 1 else 1.0

            seg_lower = segment.strip()
            if not seg_lower:
                continue

            for phrase in self.POSITIF_PHRASES:
                if phrase in seg_lower:
                    total_pos += weight * 2

            for phrase in self.NEGATIF_PHRASES:
                if phrase in seg_lower:
                    total_neg += weight * 2

            seg_clean = re.sub(r"[^\w\s-]", " ", seg_lower)
            seg_tokens = seg_clean.split()
            seg_normalized = [PreprocessorIndonesia.SLANG_DICT.get(t, t) for t in seg_tokens]
            for token in seg_normalized:
                if token in self.POSITIF_WORDS:
                    total_pos += weight
                if token in self.NEGATIF_WORDS:
                    total_neg += weight

        # ── TAMBAHAN: Skor emoji eksplisit ──
        if original_text:
            for emoji in self.POSITIF_EMOJI_EXPLICIT:
                if emoji in original_text:
                    total_pos += 1.5
            for emoji in self.NEGATIF_EMOJI_EXPLICIT:
                if emoji in original_text:
                    total_neg += 1.5
            # Emoji sarkasme (😂🤣😭) di konteks negatif → tambah negatif
            for emoji in self.SARKASME_EMOJI:
                if emoji in original_text:
                    has_neg = any(w in text_lower for w in self.NEGATIF_WORDS) or \
                              any(p in text_lower for p in self.NEGATIF_PHRASES)
                    if has_neg:
                        total_neg += 1.0

        return total_pos, total_neg

    def _thinking_label(self, text: str) -> dict:
        """
        FULL CHAIN-OF-THROW REASONING
        """
        text_lower = text.lower().strip()
        
        # Bersihkan tanda baca dan lakukan normalisasi slang
        text_clean_punc = re.sub(r"[^\w\s-]", " ", text_lower)
        tokens_orig = text_clean_punc.split()
        normalized_tokens = [PreprocessorIndonesia.SLANG_DICT.get(t, t) for t in tokens_orig]
        text_normalized = " ".join(normalized_tokens)
        token_set = set(normalized_tokens)

        result = {
            "label": "positif",
            "confidence": "rendah",
            "reasoning": "",
            "aspek_positif": [],
            "aspek_negatif": [],
            "is_sarcasm": False,
            "is_ambiguous": False,
        }

        has_sentiment_word = any(t in self.POSITIF_WORDS or t in self.NEGATIF_WORDS for t in normalized_tokens)
        if not text_lower or (len(normalized_tokens) < 2 and not has_sentiment_word):
            result["reasoning"] = "Teks terlalu pendek untuk dianalisis"
            return result

        # ── STEP 1: Identifikasi aspek positif & negatif (exact match) ──
        for w in self.POSITIF_WORDS:
            if w in token_set:
                result["aspek_positif"].append(w)
        for phrase in self.POSITIF_PHRASES:
            if phrase in text_normalized:
                result["aspek_positif"].append(f"[frasa] {phrase}")

        for w in self.NEGATIF_WORDS:
            if w in token_set:
                result["aspek_negatif"].append(w)
        for phrase in self.NEGATIF_PHRASES:
            if phrase in text_normalized:
                result["aspek_negatif"].append(f"[frasa] {phrase}")

        # ── STEP 2: Deteksi sarkasme (pass original text for emoji check) ──
        is_sarcasm = self._detect_sarcasm(text_normalized, text)
        result["is_sarcasm"] = is_sarcasm
        if is_sarcasm:
            result["label"] = "negatif"
            result["confidence"] = "tinggi"
            result["reasoning"] = "Terdeteksi SARKASME: kata positif digunakan untuk menyindir"
            return result

        # ── STEP 3: Deteksi negasi ──
        negated_pos, negated_neg = self._apply_negation(normalized_tokens)

        # ── STEP 4: Hitung skor dengan bobot posisi + emoji ──
        pos_score, neg_score = self._score_with_position_weight(text_normalized, text)

        # Terapkan negasi
        pos_score = pos_score - (negated_pos * 1.5) + (negated_neg * 1.0)
        neg_score = neg_score + (negated_pos * 1.5) - (negated_neg * 1.0)

        # ── STEP 5: Tentukan label — hanya positif/negatif ──
        has_both = len(result["aspek_positif"]) > 0 and len(result["aspek_negatif"]) > 0
        result["is_ambiguous"] = has_both

        has_contradiction = any(w in text_normalized for w in self.KONTRADIKSI_WORDS)

        # Jika ada kontradiksi dan aspek ganda, bobotkan segmen setelah kontradiksi 2x
        if has_contradiction and has_both:
            result["reasoning"] = "Teks ambigu dengan kata kontradiksi; segmen pasca-kontradiksi didominasi negatif"
            # Tambah bobot negatif ekstra untuk kasus kontradiksi
            if neg_score > 0:
                neg_score *= 1.3
        elif has_both:
            result["reasoning"] = "Teks mengandung aspek positif dan negatif; skor dominan digunakan"

        diff = abs(pos_score - neg_score)
        if neg_score > pos_score:
            result["label"] = "negatif"
            result["confidence"] = "tinggi" if diff > 3.0 else "sedang"
            if not result["reasoning"]:
                result["reasoning"] = f"Skor negatif dominan ({neg_score:.1f} vs {pos_score:.1f})"
        elif pos_score > neg_score:
            result["label"] = "positif"
            result["confidence"] = "tinggi" if diff > 3.0 else "sedang"
            if not result["reasoning"]:
                result["reasoning"] = f"Skor positif dominan ({pos_score:.1f} vs {neg_score:.1f})"
        else:
            # Tie: default ke NEGATIF (lebih konservatif untuk kebijakan)
            result["label"] = "negatif"
            result["confidence"] = "rendah"
            if not result["reasoning"]:
                result["reasoning"] = "Skor identik, default negatif (konservatif)"

        return result

    def label(self, text: str) -> str:
        """Labeli satu teks menggunakan chain-of-thought reasoning."""
        result = self._thinking_label(text)
        if self.verbose:
            print(f"\n    ── THINKING ──")
            print(f"    Teks     : {text[:80]}...")
            print(f"    Positif  : {result['aspek_positif'][:5]}")
            print(f"    Negatif  : {result['aspek_negatif'][:5]}")
            print(f"    Sarkasme : {'YA' if result['is_sarcasm'] else 'Tidak'}")
            print(f"    Ambigu   : {'YA' if result['is_ambiguous'] else 'Tidak'}")
            print(f"    Reasoning: {result['reasoning']}")
            print(f"    Label    : {result['label'].upper()} ({result['confidence']})")
        return result["label"]

    def label_with_detail(self, text: str) -> dict:
        """Labeli satu teks dan kembalikan detail reasoning lengkap."""
        return self._thinking_label(text)

    def label_batch(self, texts: list) -> list:
        """Labeli batch teks menggunakan smart reasoning."""
        return [self.label(t) for t in texts]

    def label_batch_with_detail(self, texts: list) -> list:
        """Labeli batch teks dan kembalikan semua detail reasoning."""
        return [self._thinking_label(t) for t in texts]


# ── BACKWARD COMPATIBILITY: alias LexiconLabeler ──
LexiconLabeler = SmartSentimentLabeler


# ─────────────────────────────────────────────
#  3. MODEL NAIVE BAYES
# ─────────────────────────────────────────────
class NaiveBayesSentimen:
    """
    Pipeline Naive Bayes dengan TF-IDF untuk analisis sentimen
    teks kebijakan pemerintah Indonesia.

    Mendukung:
    - MultinomialNB: cocok untuk data seimbang
    - ComplementNB: cocok untuk data tidak seimbang (REKOMENDASI)
    """

    def __init__(self, nb_type: str = "complement", model_type: str = "sgd_log",
                 use_online: bool = True, use_char_ngrams: bool = True):
        """
        Parameters
        ----------
        nb_type : str
            Legacy param: "complement" | "multinomial" (hanya dipakai jika model_type="auto")
        model_type : str
            "auto" | "complement" | "multinomial" | "sgd_log" | "sgd_hinge" | "sgd_huber"
            Default: "sgd_log" (Logistic Regression, support partial_fit + char n-grams)
        use_online : bool
            True → aktifkan partial_fit (hanya untuk sgd_*). Default: True
        use_char_ngrams : bool
            True → pakai character n-grams (tangkapi typo/slang non-kamus). Default: True
        """
        self.use_online = use_online
        self.model_type = model_type
        self.use_char_ngrams = use_char_ngrams

        # Resolve model_type
        if model_type == "auto":
            if use_online:
                model_type = "sgd_log"
            elif nb_type == "complement":
                model_type = "complement"
            else:
                model_type = "multinomial"

        # TF-IDF config
        if use_char_ngrams:
            tfidf_kwargs = dict(
                ngram_range=(2, 5),
                max_features=30000,
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                analyzer="char_wb",
            )
        else:
            tfidf_kwargs = dict(
                ngram_range=(1, 2),
                max_features=15000,
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                analyzer="word",
            )

        # Pilih classifier
        if model_type in ("sgd_log", "sgd_hinge", "sgd_huber"):
            loss_map = {
                "sgd_log": "log_loss",
                "sgd_hinge": "hinge",
                "sgd_huber": "modified_huber",
            }
            clf = SGDClassifier(
                loss=loss_map[model_type],
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                learning_rate="optimal",
                # NOTE: partial_fit tidak support class_weight='balanced'
            )
        elif model_type == "multinomial":
            clf = MultinomialNB()
        else:  # complement
            clf = ComplementNB()

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", clf)
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.classes_ = None
        self.train_report = {}

    def train(self, X_train, y_train, X_test=None, y_test=None, cv_folds: int = 5):
        """Melatih model dan evaluasi."""
        print("\n  [*] Melatih model Naive Bayes...")

        # Encode label
        y_train_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_

        # Fit model
        self.pipeline.fit(X_train, y_train_enc)
        self.is_trained = True

        # Evaluasi training
        y_pred_train = self.pipeline.predict(X_train)
        train_acc = accuracy_score(y_train_enc, y_pred_train)
        print(f"  Training accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Cross-validation
        min_class_count = pd.Series(y_train).value_counts().min()
        effective_cv_folds = min(cv_folds, int(min_class_count))
        if effective_cv_folds > 1 and len(set(y_train)) > 1:
            cv = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.pipeline, X_train, y_train_enc,
                cv=cv, scoring="accuracy"
            )
            print(f"  Cross-val ({effective_cv_folds}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            self.train_report["cv_mean"] = cv_scores.mean()
            self.train_report["cv_std"] = cv_scores.std()
            self.train_report["cv_scores"] = cv_scores.tolist()
        else:
            print("  Cross-val dilewati: data per kelas terlalu sedikit")

        # Evaluasi test set
        if X_test is not None and y_test is not None:
            y_test_enc = self.label_encoder.transform(y_test)
            y_pred_test = self.pipeline.predict(X_test)
            test_acc = accuracy_score(y_test_enc, y_pred_test)

            report = classification_report(
                y_test_enc, y_pred_test,
                target_names=self.classes_,
                output_dict=True
            )

            print(f"\n  Test accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"\n  Classification Report:")
            print(f"  {'Kelas':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print("  " + "-" * 55)
            for cls in self.classes_:
                r = report[cls]
                print(f"  {cls:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                      f"{r['f1-score']:>10.4f} {r['support']:>10.0f}")
            print("  " + "-" * 55)
            print(f"  {'Akurasi':<15} {'':>10} {'':>10} {test_acc:>10.4f}")
            print(f"  {'Macro avg':<15} {report['macro avg']['precision']:>10.4f} "
                  f"{report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f}")
            print(f"  {'Weighted avg':<15} {report['weighted avg']['precision']:>10.4f} "
                  f"{report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f}")

            self.train_report["test_acc"] = test_acc
            self.train_report["classification_report"] = report
            self.train_report["confusion_matrix"] = confusion_matrix(y_test_enc, y_pred_test).tolist()

            return test_acc, report

        return train_acc, {}

    def partial_fit(self, texts: list, labels: list):
        """
        Online / incremental learning.
        Update model dengan data baru tanpa retrain dari awal.

        NOTE: TF-IDF vocabulary tetap (dari training pertama).
              Hanya classifier weights yang diupdate.
        """
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "partial_fit"):
            raise RuntimeError(
                f"Classifier '{type(clf).__name__}' tidak mendukung partial_fit. "
                "Gunakan --model-type sgd_log / sgd_hinge / sgd_huber"
            )

        if not self.is_trained:
            # First call: fit TF-IDF + encoder + partial_fit classifier
            self.label_encoder.fit(labels)
            self.classes_ = self.label_encoder.classes_

            X_tfidf = self.pipeline.named_steps["tfidf"].fit_transform(texts)
            y_enc = self.label_encoder.transform(labels)

            clf.partial_fit(X_tfidf, y_enc,
                            classes=np.arange(len(self.classes_)))
            self.is_trained = True
            print(f"  [ONLINE] Initial fit: {len(texts)} samples, "
                  f"classes={list(self.classes_)}")
        else:
            # Subsequent calls: transform only, then partial_fit
            X_tfidf = self.pipeline.named_steps["tfidf"].transform(texts)
            y_enc = self.label_encoder.transform(labels)
            clf.partial_fit(X_tfidf, y_enc)
            print(f"  [ONLINE] Updated with {len(texts)} samples")

    def predict(self, texts: list) -> list:
        """Prediksi label sentimen."""
        if not self.is_trained:
            raise RuntimeError("Model belum dilatih. Panggil .train() dulu.")
        preds = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, texts: list) -> np.ndarray:
        """Probabilitas prediksi per kelas."""
        if not self.is_trained:
            raise RuntimeError("Model belum dilatih.")
        return self.pipeline.predict_proba(texts)

    def save(self, path: str = "./hasil/model_nb.pkl"):
        """Simpan model ke file."""
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "classes_": self.classes_,
            "train_report": self.train_report,
            "model_type": getattr(self, "model_type", "auto"),
            "use_char_ngrams": getattr(self, "use_char_ngrams", False),
        }, path)
        print(f"  [SAVED] Model disimpan ke {path}")

    @classmethod
    def load(cls, path: str):
        """Load model dari file."""
        data = joblib.load(path)
        obj = cls(
            model_type=data.get("model_type", "auto"),
            use_char_ngrams=data.get("use_char_ngrams", False),
        )
        obj.pipeline = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        obj.classes_ = data["classes_"]
        obj.train_report = data["train_report"]
        obj.is_trained = True
        return obj


# ─────────────────────────────────────────────
#  4. ANALISIS & VISUALISASI
# ─────────────────────────────────────────────
def hitung_distribusi_sentimen(df: pd.DataFrame, kolom_label: str = "sentimen") -> pd.DataFrame:
    """Hitung persentase positif/negatif per topik."""
    hasil = []
    for topik, group in df.groupby("topik"):
        total = len(group)
        counts = group[kolom_label].value_counts()

        positif = counts.get("positif", 0)
        negatif = counts.get("negatif", 0)

        hasil.append({
            "topik": topik,
            "total_data": total,
            "negatif_count": negatif,
            "positif_count": positif,
            "negatif_pct": round(negatif / total * 100, 2),
            "positif_pct": round(positif / total * 100, 2),
            "sentimen_dominan": counts.idxmax() if not counts.empty else "positif"
        })
    return pd.DataFrame(hasil)


def plot_distribusi_sentimen(df_ringkasan: pd.DataFrame, output_dir: str = "./hasil/plots"):
    """Bar chart distribusi sentimen per topik."""
    fig, ax = plt.subplots(figsize=(14, 7))

    topik_labels = [t.replace("_", "\n").title() for t in df_ringkasan["topik"]]
    x = np.arange(len(topik_labels))
    width = 0.35

    colors = {"negatif": "#e74c3c", "positif": "#27ae60"}

    bars1 = ax.bar(x - width/2, df_ringkasan["negatif_pct"], width,
                   label="Negatif", color=colors["negatif"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, df_ringkasan["positif_pct"], width,
                   label="Positif", color=colors["positif"], edgecolor="white", linewidth=0.5)

    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    ax.set_xlabel("Topik Kebijakan", fontsize=12, labelpad=10)
    ax.set_ylabel("Persentase (%)", fontsize=12)
    ax.set_title("Distribusi Sentimen Masyarakat terhadap Kebijakan Pemerintah Indonesia\n"
                 "(Metode: Naive Bayes + TF-IDF)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(topik_labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "distribusi_sentimen_per_topik.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Plot -> {path}")


def plot_pie_per_topik(df: pd.DataFrame, output_dir: str = "./hasil/plots"):
    """Pie chart per topik kebijakan."""
    topik_list = df["topik"].unique()
    n = len(topik_list)

    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    colors_pie = ["#e74c3c", "#27ae60"]
    label_order = ["negatif", "positif"]

    for i, topik in enumerate(topik_list):
        group = df[df["topik"] == topik]["sentimen"].value_counts()
        labels = group.index.tolist()
        sizes  = group.values.tolist()
        colors = [colors_pie[label_order.index(l)] if l in label_order else "#95a5a6"
                  for l in labels]

        ax = axes[i]
        wedges, texts, autotexts = ax.pie(  # type: ignore
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10}
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(topik.replace("_", " ").title(), fontsize=11, fontweight="bold")

    # Sembunyikan axes kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribusi Sentimen per Kebijakan (Naive Bayes)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "pie_chart_per_topik.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Pie charts -> {path}")


def plot_confusion_matrix(cm_data: list, classes: list, output_dir: str = "./hasil/plots"):
    """Plot confusion matrix sebagai heatmap."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Aktual", fontsize=12)
    ax.set_title("Confusion Matrix - Naive Bayes\nAnalisis Sentimen Kebijakan Pemerintah",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] Confusion Matrix -> {path}")


def generate_wordcloud(df: pd.DataFrame, output_dir: str = "./hasil/plots"):
    """Word cloud per sentimen."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [SKIP] wordcloud tidak terinstall. pip install wordcloud")
        return

    sentimen_colors = {
        "negatif": "Reds",
        "positif": "Greens",
    }

    for sentimen, colormap in sentimen_colors.items():
        texts = df[df["sentimen"] == sentimen]["text_clean"]
        if texts.empty:
            continue
        corpus = " ".join(texts.tolist())
        if not corpus.strip():
            continue

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap=colormap,
            max_words=100,
            collocations=False
        ).generate(corpus)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud — Sentimen {sentimen.capitalize()}",
                  fontsize=14, fontweight="bold")
        path = os.path.join(output_dir, f"wordcloud_{sentimen}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] Word Cloud {sentimen} -> {path}")


def print_ringkasan(df_ringkasan: pd.DataFrame):
    """Print tabel ringkasan distribusi sentimen ke terminal."""
    print("\n" + "=" * 80)
    print("  RINGKASAN DISTRIBUSI SENTIMEN (NAIVE BAYES + TF-IDF)")
    print("=" * 80)
    header = (f"  {'Topik Kebijakan':<28} {'Total':>7} "
              f"{'Negatif':>10} {'Positif':>9} {'Dominan':>9}")
    print(header)
    print("  " + "-" * 65)
    for _, row in df_ringkasan.iterrows():
        topik_display = row["topik"].replace("_", " ").title()[:27]
        print(f"  {topik_display:<26} {row['total_data']:>7} "
              f"{row['negatif_pct']:>8.2f}%  "
              f"{row['positif_pct']:>7.2f}%  {row['sentimen_dominan']:>8}")
    print("  " + "-" * 65)

    # Hitung total keseluruhan
    total_all = df_ringkasan["total_data"].sum()
    neg_all = df_ringkasan["negatif_count"].sum()
    pos_all = df_ringkasan["positif_count"].sum()
    print(f"  {'TOTAL KESELURUHAN':<26} {total_all:>7} "
          f"{neg_all/total_all*100:>8.2f}%  "
          f"{pos_all/total_all*100:>7.2f}%")
    print("=" * 80)


def buat_laporan_json(df_ringkasan: pd.DataFrame, model_report: dict):
    """Simpan laporan lengkap ke JSON."""
    laporan = {
        "judul": "Analisis Sentimen Masyarakat Terhadap Kebijakan Pemerintah Indonesia",
        "metode": "Naive Bayes (ComplementNB) + TF-IDF Vectorizer",
        "generated_at": datetime.now().isoformat(),
        "metrik_model": {
            "accuracy": model_report.get("test_acc", "N/A"),
            "cv_mean": model_report.get("cv_mean", "N/A"),
            "cv_std": model_report.get("cv_std", "N/A"),
            "cv_scores": model_report.get("cv_scores", [])
        },
        "distribusi_sentimen": df_ringkasan.to_dict(orient="records"),
        "topik_kebijakan": [
            "1. Pemerintahan Prabowo-Gibran",
            "2. Omnibus Law Cipta Kerja",
            "3. Danantara (Badan Pengelola Investasi)",
            "4. Makan Bergizi Gratis (MBG)",
            "5. Efisiensi Anggaran 2025"
        ]
    }
    path = "./hasil/laporan_analisis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(laporan, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] Laporan JSON -> {path}")


# ─────────────────────────────────────────────
#  5. DATA SAMPLE (untuk testing cepat)
# ─────────────────────────────────────────────
def buat_data_sample() -> pd.DataFrame:
    """
    Data sample berlabel untuk demo/testing pipeline.
    Lebih kecil dari dataset realistis, tapi cukup untuk testing.
    """
    samples = [
        # Prabowo-Gibran
        ("prabowo_gibran", "kebijakan prabowo gibran bagus untuk rakyat indonesia maju", "positif"),
        ("prabowo_gibran", "100 hari prabowo gagal tidak ada perubahan berarti", "negatif"),
        ("prabowo_gibran", "prabowo presiden terbaik indonesia makmur sejahtera", "positif"),
        ("prabowo_gibran", "kecewa dengan kabinet prabowo korupsi merajalela lagi", "negatif"),
        ("prabowo_gibran", "program kerja prabowo pro rakyat kecil mendukung penuh", "positif"),
        ("prabowo_gibran", "prabowo bohong janji kampanye tidak ditepati rakyat kecewa", "negatif"),
        ("prabowo_gibran", "prabowo gibran melanjutkan pembangunan IKN dengan baik sekali", "positif"),
        ("prabowo_gibran", "harga sembako naik terus sejak prabowo menjabat susah hidup", "negatif"),

        # Omnibus Law
        ("omnibus_law", "omnibus law cipta kerja merugikan buruh tidak pro rakyat kecil", "negatif"),
        ("omnibus_law", "omnibus law membuka lapangan kerja investasi meningkat pesat", "positif"),
        ("omnibus_law", "tolak omnibus law hidup buruh terancam kebijakan deregulasi", "negatif"),
        ("omnibus_law", "omnibus law memudahkan izin usaha UMKM berkembang lebih cepat", "positif"),
        ("omnibus_law", "UU cipta kerja melanggar hak buruh upah minimum dihapus", "negatif"),
        ("omnibus_law", "investasi meningkat berkat omnibus law ekonomi tumbuh signifikan", "positif"),
        ("omnibus_law", "omnibus law merusak lingkungan izin tambang diberikan dengan mudah", "negatif"),
        ("omnibus_law", "omnibus law bikin iklim investasi kondusif mantap pemerintah", "positif"),

        # Danantara
        ("danantara", "danantara investasi BUMN rawan korupsi tidak transparan bahaya", "negatif"),
        ("danantara", "danantara bisa dorong ekonomi nasional maju pesat", "positif"),
        ("danantara", "dana rakyat dipakai danantara tanpa pengawasan yang jelas", "negatif"),
        ("danantara", "danantara pemerintah berkomitmen perbaiki tata kelola BUMN profesional", "positif"),
        ("danantara", "danantara proyek korupsi baru oligarki menguasai aset BUMN", "negatif"),
        ("danantara", "danantara bisa jadi solusi pengelolaan aset negara yang optimal", "positif"),
        ("danantara", "danantara ancaman privatisasi BUMN terselubung berbahaya", "negatif"),
        ("danantara", "danantara bikin BUMN lebih kompetitif di pasar global hebat", "positif"),

        # Makan Bergizi Gratis
        ("makan_bergizi_gratis", "program makan bergizi gratis sangat membantu siswa miskin", "positif"),
        ("makan_bergizi_gratis", "makan bergizi gratis amburadul banyak yang tidak layak makan", "negatif"),
        ("makan_bergizi_gratis", "makan bergizi gratis sia sia anggaran habis gizi tidak terpenuhi", "negatif"),
        ("makan_bergizi_gratis", "anak sekolah senang dapat makan bergizi gratis tiap hari sehat", "positif"),
        ("makan_bergizi_gratis", "makan gratis sekolah mengurangi stunting generasi muda sehat kuat", "positif"),
        ("makan_bergizi_gratis", "kualitas makan bergizi gratis buruk tidak layak konsumsi parah", "negatif"),
        ("makan_bergizi_gratis", "MBG program terbaik prabowo anak sekolah jadi semangat belajar", "positif"),
        ("makan_bergizi_gratis", "makanan MBG sering basi anak sekolah sakit perut keracunan", "negatif"),

        # Efisiensi Anggaran
        ("efisiensi_anggaran", "pemangkasan anggaran merugikan pendidikan kesehatan rakyat kecil", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran diperlukan untuk mengurangi pemborosan negara", "positif"),
        ("efisiensi_anggaran", "pemotongan anggaran gaji guru dipotong sangat kecewa sekali", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran pemerintah hemat dorong produktivitas tinggi", "positif"),
        ("efisiensi_anggaran", "efisiensi anggaran birokrasi ramping korupsi berkurang signifikan", "positif"),
        ("efisiensi_anggaran", "pemotongan anggaran pelayanan publik menurun drastis parah", "negatif"),
        ("efisiensi_anggaran", "anggaran sosial dipotong rakyat kecil semakin susah hidup", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran bikin pejabat gak bisa foya pakai uang negara", "positif"),
    ]

    # Duplikasi 8x dengan variasi kecil
    expanded = []
    for topik, text, label in samples * 8:
        expanded.append({
            "topik": topik,
            "platform": "sample",
            "text": text,
            "sentimen": label
        })

    return pd.DataFrame(expanded)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def _retrain_from_feedback(
    dataset_path: str,
    feedback_path: str,
    model_type: str = "auto",
    use_online: bool = False,
    use_char_ngrams: bool = False,
    no_stem: bool = False,
    test_size: float = 0.2,
):
    """
    Retrain model dari dataset asli + feedback history user.
    """
    print("\n" + "=" * 65)
    print("  RETRAIN MODEL DARI DATASET + FEEDBACK HISTORY")
    print("=" * 65)

    preprocessor = PreprocessorIndonesia(use_stemming=not no_stem)

    # 1. Load & preprocess dataset asli
    df_main = None
    if os.path.exists(dataset_path):
        print(f"\n  [*] Loading dataset: {dataset_path}")
        df_main = pd.read_csv(dataset_path, encoding="utf-8-sig")
        print(f"      -> {len(df_main)} rows from dataset")

        text_col = pilih_kolom_teks(df_main)
        label_col = find_label_column(df_main)

        if label_col is None:
            print("  [INFO] Melabeli dataset otomatis...")
            labeler = SmartSentimentLabeler()
            df_main["sentimen_auto"] = labeler.label_batch(
                df_main[text_col].fillna("").astype(str).tolist()
            )
            label_col = "sentimen_auto"

        df_main["text_clean"] = preprocessor.preprocess_batch(
            df_main[text_col].fillna("").astype(str).tolist()
        )

        if label_col != "sentimen":
            df_main["sentimen"] = df_main[label_col]
    else:
        print(f"  [WARN] Dataset tidak ditemukan: {dataset_path}")

    # 2. Load feedback history (already preprocessed)
    df_fb = None
    if os.path.exists(feedback_path):
        print(f"\n  [*] Loading feedback history: {feedback_path}")
        df_fb = pd.read_csv(feedback_path, encoding="utf-8-sig")
        if not df_fb.empty:
            print(f"      -> {len(df_fb)} rows from feedback history")
            if "sentimen" in df_fb.columns:
                dist = df_fb["sentimen"].value_counts().to_dict()
                print(f"      -> Feedback distribution: {dist}")
            # Ensure text_clean exists
            if "text_clean" not in df_fb.columns:
                text_col = pilih_kolom_teks(df_fb)
                df_fb["text_clean"] = preprocessor.preprocess_batch(
                    df_fb[text_col].fillna("").astype(str).tolist()
                )
    else:
        print(f"\n  [WARN] Feedback history tidak ditemukan: {feedback_path}")

    parts = [df_main, df_fb]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        print("  [ERROR] Tidak ada data untuk retrain!")
        return

    df = pd.concat(parts, ignore_index=True)

    # Dedup by text_clean (safe: both parts now have it)
    df.drop_duplicates(subset=["text_clean"], keep="last", inplace=True)
    print(f"\n  [*] Total after merge + dedup: {len(df)} rows")

    # 3. Normalize labels
    df["sentimen"] = df["sentimen"].str.lower().str.strip()
    df["sentimen"] = df["sentimen"].replace({
        "negative": "negatif", "neg": "negatif",
        "positive": "positif", "pos": "positif",
        "neutral": "negatif", "neu": "negatif",
        "netral": "negatif",
    })
    df = df[df["sentimen"].isin(["positif", "negatif"])].reset_index(drop=True)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"  Label distribution: {df['sentimen'].value_counts().to_dict()}")

    if len(df) < 5:
        print("  [ERROR] Data terlalu sedikit untuk retrain.")
        return

    # 4. Train model
    X = df["text_clean"].tolist()
    y = df["sentimen"].tolist()

    min_class = df["sentimen"].value_counts().min()
    n_classes = df["sentimen"].nunique()
    can_split = len(df) >= 30 and min_class >= 2 and int(len(df) * test_size) >= n_classes

    if not can_split:
        print(f"  [INFO] Data terbatas ({len(df)} rows, {n_classes} classes, "
              f"min_class={min_class}). Full data for training.")
        X_train, X_test, y_train, y_test = X, [], y, []
    else:
        stratify_labels = y if min_class >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=stratify_labels, random_state=42,
        )
        print(f"\n  Split: {len(X_train)} train / {len(X_test)} test")

    model = NaiveBayesSentimen(
        model_type=model_type,
        use_online=use_online,
        use_char_ngrams=use_char_ngrams,
    )
    model.train(X_train, y_train, X_test or None, y_test or None)

    # 5. Save model
    model.save("./hasil/model_nb.pkl")

    # 6. Ringkasan
    df["sentimen_pred"] = model.predict(X)
    if "topik" not in df.columns:
        df["topik"] = "feedback_retrain"
    df_ringkasan = hitung_distribusi_sentimen(df)
    df_ringkasan.to_csv("./hasil/ringkasan_topik.csv", index=False, encoding="utf-8-sig")
    print_ringkasan(df_ringkasan)
    print("\n  [DONE] Retrain selesai! Model diperbarui.")


def find_label_column(df: pd.DataFrame) -> str | None:
    """Cari kolom label sentimen."""
    for col in ["sentimen", "sentimen_manual", "label", "sentimen_auto"]:
        if col in df.columns:
            return col
    return None


def get_model_vocabulary(model: NaiveBayesSentimen) -> set[str]:
    """Ambil vocabulary TF-IDF untuk cek kata yang sudah dikenal model."""
    try:
        return set(model.pipeline.named_steps["tfidf"].get_feature_names_out())
    except Exception:
        return set()


def known_word_ratio(text_clean: str, vocabulary: set[str]) -> float:
    """Hitung rasio token komentar yang pernah dikenal model."""
    tokens = [token for token in text_clean.split() if token]
    if not tokens:
        return 0.0
    if not vocabulary:
        return 1.0
    known = sum(1 for token in tokens if token in vocabulary)
    return known / len(tokens)


def predict_with_learning_model(
    df_comments: pd.DataFrame,
    model: NaiveBayesSentimen,
    confidence_threshold: float = 0.45,
    min_known_ratio: float = 0.25,
    preprocessor: PreprocessorIndonesia = None,
) -> pd.DataFrame:
    """
    Prediksi komentar dengan model ML, lalu fallback untuk teks yang kurang dipahami.
    Mirip logika di scan_link.py.
    """
    fallback_labeler = SmartSentimentLabeler()

    if model is None:
        print("        [WARN] Model ML tidak tersedia. Memakai SmartSentimentLabeler untuk semua komentar.")
        details = fallback_labeler.label_batch_with_detail(df_comments["text_clean"].tolist())
        df_comments["sentimen_pred"] = [item["label"] for item in details]
        df_comments["confidence"] = [item["confidence"] for item in details]
        df_comments["confidence_score"] = 0.0
        df_comments["known_word_ratio"] = 0.0
        df_comments["metode_prediksi"] = "smart_labeler"
        return df_comments

    texts_clean = df_comments["text_clean"].tolist()
    pred_labels = model.predict(texts_clean)
    probas = model.predict_proba(texts_clean)
    max_probas = probas.max(axis=1)
    vocabulary = get_model_vocabulary(model)
    known_ratios = [known_word_ratio(text, vocabulary) for text in texts_clean]

    final_labels = []
    confidence_names = []
    methods = []
    reasonings = []

    for text_clean, pred, score, ratio in zip(texts_clean, pred_labels, max_probas, known_ratios):
        if score < confidence_threshold or ratio < min_known_ratio:
            detail = fallback_labeler.label_with_detail(text_clean)
            final_labels.append(detail["label"])
            confidence_names.append(detail["confidence"])
            methods.append("fallback_smart_labeler")
            reasonings.append(detail["reasoning"])
        else:
            final_labels.append(pred)
            if score >= 0.70:
                confidence_names.append("tinggi")
            elif score >= confidence_threshold:
                confidence_names.append("sedang")
            else:
                confidence_names.append("rendah")
            methods.append("model_nb")
            reasonings.append("Prediksi model ML")

    df_comments["sentimen_pred"] = final_labels
    df_comments["confidence"] = confidence_names
    df_comments["confidence_score"] = max_probas
    df_comments["known_word_ratio"] = known_ratios
    df_comments["metode_prediksi"] = methods
    df_comments["reasoning"] = reasonings
    return df_comments


def main():
    parser = argparse.ArgumentParser(
        description="Naive Bayes Sentimen Kebijakan Pemerintah Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CONTOH PENGGUNAAN:
  # Analisis dari CSV hasil scraping:
  python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv

  # Demo dengan data sample kecil:
  python naive_bayes_sentimen.py --sample

  # Prediksi sentimen kalimat baru:
  python naive_bayes_sentimen.py --prediksi "danantara merugikan rakyat kecil"

  # Load model tersimpan:
  python naive_bayes_sentimen.py --load-model ./hasil/model_nb.pkl \\
    --prediksi "makan bergizi gratis program terbaik"
        """
    )
    parser.add_argument("--input", type=str, default="./data_raw/semua_topik.csv",
                        help="Path CSV input (kolom teks otomatis: text/comment/komentar/dll)")
    parser.add_argument("--text-col", type=str, default=None,
                        help="Nama kolom teks jika ingin ditentukan manual")
    parser.add_argument("--sample", action="store_true",
                        help="Gunakan data sample built-in untuk demo cepat")
    parser.add_argument("--nb-type", choices=["complement", "multinomial"], default="complement",
                        help="Jenis Naive Bayes (default: complement, lebih baik untuk imbalanced)")
    parser.add_argument("--model-type", choices=["auto", "complement", "multinomial",
                                                  "sgd_log", "sgd_hinge", "sgd_huber"],
                        default="auto",
                        help="Model classifier (default: auto = complement / sgd_log jika --use-online)")
    parser.add_argument("--use-online", action="store_true",
                        help="Aktifkan online learning (partial_fit), model -> sgd_log")
    parser.add_argument("--use-char-ngrams", action="store_true",
                        help="Pakai character n-grams (tangkap typo/slang non-kamus)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporsi test set (default: 0.2 = 80/20 split)")
    parser.add_argument("--no-stem", action="store_true",
                        help="Nonaktifkan stemming Sastrawi (lebih cepat)")
    parser.add_argument("--prediksi", type=str, default=None,
                        help="Prediksi sentimen 1 kalimat baru")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Load model tersimpan dari file .pkl")
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                        help="Ambang confidence model sebelum fallback smart labeler (default: 0.45)")
    parser.add_argument("--min-known-ratio", type=float, default=0.25,
                        help="Rasio minimal kata yang dikenal model sebelum fallback (default: 0.25)")
    parser.add_argument("--retrain-feedback", action="store_true",
                        help="Retrain model dari dataset asli + feedback history")
    parser.add_argument("--feedback-path", type=str,
                        default="./hasil_scanning/feedback_history.csv",
                        help="Path ke feedback history CSV (default: ./hasil_scanning/feedback_history.csv)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  ANALISIS SENTIMEN KEBIJAKAN PEMERINTAH INDONESIA")
    model_label = args.model_type if args.model_type != "auto" else (
        "sgd_log" if args.use_online else
        "complement" if args.nb_type == "complement" else "multinomial"
    )
    feats = "char_ngrams" if args.use_char_ngrams else "word_unigram+bigram"
    print(f"  Model  : {model_label} | fitur={feats}")
    print("  Topik  : 5 Kebijakan Pemerintah")
    print("=" * 65)

    # ── Mode: retrain dari feedback history ──
    if args.retrain_feedback:
        return _retrain_from_feedback(
            dataset_path=args.input,
            feedback_path=args.feedback_path,
            model_type=args.model_type,
            use_online=args.use_online,
            use_char_ngrams=args.use_char_ngrams,
            no_stem=args.no_stem,
            test_size=args.test_size,
        )

    # ── Mode: prediksi kalimat tunggal dari model tersimpan ──
    if args.prediksi and args.load_model:
        print(f"\n  [*] Load model dari: {args.load_model}")
        model = NaiveBayesSentimen.load(args.load_model)
        preprocessor = PreprocessorIndonesia(use_stemming=not args.no_stem)

        text_clean = preprocessor.preprocess(args.prediksi)
        pred = model.predict([text_clean])[0]
        proba = model.predict_proba([text_clean])[0]
        proba_dict = dict(zip(model.classes_, proba))

        print(f"\n  Teks asli  : {args.prediksi}")
        print(f"  Teks bersih: {text_clean}")
        print(f"  Sentimen   : {pred.upper()}")
        print(f"\n  Probabilitas:")
        for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"    {cls:<10}: {p:.4f} ({p*100:.1f}%) {bar}")
        return

    # ── Mode: load model terlatih + prediksi CSV input (mirip scan_link.py) ──
    if args.input and args.load_model and not args.prediksi and not args.sample and not args.retrain_feedback:
        print(f"\n  [*] Load model dari: {args.load_model}")
        model = NaiveBayesSentimen.load(args.load_model)
        preprocessor = PreprocessorIndonesia(use_stemming=not args.no_stem)

        print(f"\n  [*] Membaca data dari: {args.input}")
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        print(f"  Total data: {len(df):,} baris")
        text_col = pilih_kolom_teks(df, args.text_col)
        print(f"  Kolom teks: {text_col}")

        if "topik" not in df.columns:
            topik_default = os.path.splitext(os.path.basename(args.input))[0]
            df["topik"] = topik_default
            print(f"  [INFO] Kolom topik tidak ditemukan. Menggunakan topik: {topik_default}")

        print("\n  [*] Preprocessing teks...")
        df["text_clean"] = preprocessor.preprocess_batch(df[text_col].fillna("").astype(str).tolist())
        df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
        print(f"  Setelah filter: {len(df):,} baris valid")

        # Prediksi dengan confidence threshold + fallback (seperti scan_link.py)
        print("  [*] Prediksi dengan model + fallback Smart Reasoning...")
        df = predict_with_learning_model(
            df, model,
            confidence_threshold=args.confidence_threshold,
            min_known_ratio=args.min_known_ratio,
            preprocessor=preprocessor,
        )

        # Simpan hasil
        output_csv = "./hasil/laporan_sentimen.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"  [SAVED] Hasil prediksi -> {output_csv}")

        # Ringkasan
        df_ringkasan = hitung_distribusi_sentimen(df, kolom_label="sentimen_pred")
        ringkasan_csv = "./hasil/ringkasan_topik.csv"
        df_ringkasan.to_csv(ringkasan_csv, index=False, encoding="utf-8-sig")
        print(f"  [SAVED] Ringkasan -> {ringkasan_csv}")
        print_ringkasan(df_ringkasan)

        # Visualisasi
        print("\n  [*] Membuat visualisasi...")
        plot_distribusi_sentimen(df_ringkasan)
        # plot_pie_per_topik butuh kolom 'sentimen', gunakan sentimen_pred
        df_viz = df.copy()
        df_viz["sentimen"] = df_viz["sentimen_pred"]
        plot_pie_per_topik(df_viz, "./hasil/plots")
        generate_wordcloud(df_viz, "./hasil/plots")

        print("\n  " + "=" * 55)
        print("  [DONE] Prediksi CSV selesai!")
        print(f"         Output  : ./hasil/")
        print(f"         Plots   : ./hasil/plots/")
        print("  " + "=" * 55)
        return

    if args.use_online:
        print("\n  [INFO] Online learning mode aktif. Model akan support partial_fit.")
        print("         Gunakan scan_link.py dengan feedback user untuk update model.")

    # ── Load / Generate Data ──
    if args.sample:
        print("\n  [INFO] Menggunakan data sample built-in (kecil, untuk demo)")
        df = buat_data_sample()
        label_col = "sentimen"
        text_col = "text"
    elif os.path.exists(args.input):
        print(f"\n  [*] Membaca data dari: {args.input}")
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        print(f"  Total data: {len(df):,} baris")
        text_col = pilih_kolom_teks(df, args.text_col)
        print(f"  Kolom teks: {text_col}")

        if "topik" not in df.columns:
            topik_default = os.path.splitext(os.path.basename(args.input))[0]
            df["topik"] = topik_default
            print(f"  [INFO] Kolom topik tidak ditemukan. Menggunakan topik: {topik_default}")

        # Tentukan kolom label
        if "sentimen" in df.columns:
            label_col = "sentimen"
        elif "sentimen_manual" in df.columns:
            label_col = "sentimen_manual"
        elif "label" in df.columns:
            label_col = "label"
        else:
            print("  [INFO] Kolom label tidak ditemukan. Melabeli otomatis dengan Smart Reasoning...")
            labeler = SmartSentimentLabeler()
            texts = df[text_col].fillna("").astype(str).tolist()
            df["sentimen_auto"] = labeler.label_batch(texts)
            label_col = "sentimen_auto"
    else:
        print(f"\n  [WARN] File tidak ditemukan: {args.input}")
        print("         Menggunakan data sample. Gunakan --sample untuk eksplisit.")
        print("         Atau generate dataset dulu: python scraper_kebijakan.py --sample")
        df = buat_data_sample()
        label_col = "sentimen"
        text_col = "text"

    print(f"  Distribusi label awal: {df[label_col].value_counts().to_dict()}")

    # ── Preprocessing ──
    print("\n  [*] Preprocessing teks bahasa Indonesia...")
    preprocessor = PreprocessorIndonesia(use_stemming=not args.no_stem)
    df["text_clean"] = preprocessor.preprocess_batch(df[text_col].fillna("").astype(str).tolist())

    # Filter baris kosong setelah preprocessing
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"  Setelah filter: {len(df):,} baris valid")

    # ── Normalisasi label ──
    if label_col == "sentimen_auto":
        df["sentimen"] = df["sentimen_auto"]
    elif label_col != "sentimen":
        df["sentimen"] = df[label_col]

    df["sentimen"] = df["sentimen"].str.lower().str.strip()
    df["sentimen"] = df["sentimen"].replace({
        "negative": "negatif", "neg": "negatif",
        "positive": "positif", "pos": "positif",
        "neutral": "negatif", "neu": "negatif",
        "netral": "negatif",
    })
    df = df[df["sentimen"].isin(["positif", "negatif"])].reset_index(drop=True)
    print(f"  Distribusi label final: {df['sentimen'].value_counts().to_dict()}")

    # ── Train / Test Split ──
    X = df["text_clean"].tolist()
    y = df["sentimen"].tolist()
    class_counts = df["sentimen"].value_counts()
    stratify_labels = y if class_counts.min() >= 2 else None

    if stratify_labels is None:
        print("  [WARN] Split tanpa stratify: ada kelas dengan kurang dari 2 data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=stratify_labels,
        random_state=42
    )
    print(f"\n  Data split: {len(X_train)} train / {len(X_test)} test "
          f"({int((1-args.test_size)*100)}/{int(args.test_size*100)})")

    # ── Training ──
    model = NaiveBayesSentimen(
        nb_type=args.nb_type,
        model_type=args.model_type,
        use_online=args.use_online,
        use_char_ngrams=args.use_char_ngrams,
    )
    acc, report = model.train(X_train, y_train, X_test, y_test)

    # ── Prediksi seluruh dataset ──
    print("\n  [*] Prediksi seluruh dataset...")
    df["sentimen_pred"] = model.predict(X)

    # Probabilitas per kelas
    proba_all = model.predict_proba(X)
    for i, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = proba_all[:, i]

    # ── Simpan hasil ──
    output_csv = "./hasil/laporan_sentimen.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  [SAVED] Hasil prediksi -> {output_csv}")

    # ── Hitung distribusi per topik ──
    df_ringkasan = hitung_distribusi_sentimen(df, kolom_label="sentimen_pred")
    ringkasan_csv = "./hasil/ringkasan_topik.csv"
    df_ringkasan.to_csv(ringkasan_csv, index=False, encoding="utf-8-sig")
    print(f"  [SAVED] Ringkasan -> {ringkasan_csv}")

    # ── Print ringkasan ──
    print_ringkasan(df_ringkasan)

    # ── Visualisasi ──
    print("\n  [*] Membuat visualisasi...")
    plot_distribusi_sentimen(df_ringkasan)
    plot_pie_per_topik(df, "./hasil/plots")
    generate_wordcloud(df, "./hasil/plots")

    if "confusion_matrix" in model.train_report:
        plot_confusion_matrix(
            model.train_report["confusion_matrix"],
            model.classes_.tolist(),
            "./hasil/plots"
        )

    # ── Simpan model & laporan ──
    model.save("./hasil/model_nb.pkl")
    buat_laporan_json(df_ringkasan, model.train_report)

    # ── Contoh prediksi kalimat baru ──
    if args.prediksi:
        print(f"\n  [*] Prediksi kalimat: '{args.prediksi}'")
        text_clean = preprocessor.preprocess(args.prediksi)
        pred = model.predict([text_clean])[0]
        proba = model.predict_proba([text_clean])[0]
        proba_dict = dict(zip(model.classes_, proba))
        print(f"  Hasil    : {pred.upper()}")
        print(f"  Probabilitas:")
        for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"    {cls:<10}: {p:.4f} ({p*100:.1f}%) {bar}")

    print("\n  " + "=" * 55)
    print("  [DONE] Analisis sentimen selesai!")
    print(f"         Output  : ./hasil/")
    print(f"         Model   : ./hasil/model_nb.pkl")
    print(f"         Plots   : ./hasil/plots/")
    print(f"         Laporan : ./hasil/laporan_analisis.json")
    print("  " + "=" * 55)


if __name__ == "__main__":
    main()

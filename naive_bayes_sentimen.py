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
    "full_text", "tweet", "body", "message"
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

    # Kamus normalisasi slang/singkatan umum Twitter/TikTok (120+ kata)
    SLANG_DICT = {
        # Negasi
        "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak",
        "nggak": "tidak", "tdk": "tidak", "gbs": "tidak bisa", "gabisa": "tidak bisa",
        "gaada": "tidak ada", "gada": "tidak ada", "gamau": "tidak mau",
        "gabole": "tidak boleh", "gabakal": "tidak akan",
        # Intensifier
        "bgt": "banget", "bngt": "banget", "bgtt": "banget",
        "bener2": "benar-benar", "bnr": "benar", "bnr2": "benar-benar",
        "bner": "benar", "sgt": "sangat", "amat": "sangat",
        # Pronoun
        "yg": "yang", "dg": "dengan", "dgn": "dengan", "utk": "untuk",
        "krn": "karena", "karna": "karena", "klo": "kalau", "kalo": "kalau",
        "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
        "blm": "belum", "blum": "belum", "msh": "masih",
        "sy": "saya", "gw": "saya", "gue": "saya", "ane": "saya",
        "lu": "kamu", "lo": "kamu", "kmu": "kamu", "elo": "kamu",
        "dr": "dari", "dri": "dari", "pd": "pada", "dlm": "dalam",
        "ttg": "tentang", "spt": "seperti", "sprt": "seperti",
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
        text = re.sub(r"[^\w\s]", " ", text)                     # hapus tanda baca & emoji
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
#  2. LABELING OTOMATIS (Rule-based + Lexicon)
# ─────────────────────────────────────────────
class LexiconLabeler:
    """
    Pelabelan sentimen berbasis lexicon untuk data yang BELUM berlabel.
    Digunakan sebagai inisiasi sebelum Naive Bayes dilatih.
    """

    POSITIF_WORDS = {
        "bagus", "baik", "mantap", "dukung", "setuju", "bermanfaat",
        "berhasil", "sukses", "positif", "maju", "pro", "mendukung",
        "meningkat", "harapan", "solusi", "inovasi", "keren",
        "sejahtera", "makmur", "tepat", "efektif", "efisien",
        "transparan", "adil", "merata", "terlaksana", "terwujud",
        "apresiasi", "pujian", "terimakasih", "luar biasa", "hebat",
        "wow", "amazing", "good", "best", "optimal", "brilian",
        "cerdas", "visioner", "progresif", "kompeten", "profesional",
        "berani", "tegas", "salut", "respect", "bangga", "proud",
        "senang", "gembira", "puas", "lega", "syukur", "alhamdulillah",
        "terbaik", "berkualitas", "unggul", "mumpuni", "andal",
        "membantu", "menolong", "memudahkan", "menyejahterakan",
        "bagus", "baik", "mantap", "dukung", "setuju",
        "program bagus", "sangat membantu", "bermanfaat",
        "anak senang", "makan gratis", "gizi terpenuhi",
        "lanjutkan", "semoga lancar", "tepat sasaran",
    }

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
        "PHK", "pengangguran", "inflasi", "utang", "defisit",
        "manipulasi", "penyelewengan", "penyalahgunaan", "korup",
        "berbahaya", "merugikan", "mengorbankan", "mengkhianati",
        "gagal", "tolak", "kecewa", "rugi", "korupsi",
        "bohong", "tipu", "palsu", "salah", "jelek", "parah",
        "g ada mbg", "ga ada mbg", "gak ada mbg", "tidak ada mbg",
        "dulu g ada", "dulu ga ada", "masih hidup",
        "buat apa", "percuma", "mending", "tidak perlu",
        "pencitraan", "buang anggaran", "hambur anggaran",
        "uang rakyat", "program gagal", "tidak setuju",
        "keracunan", "basi", "tidak layak", "sakit perut",
  
    }

    def label(self, text: str) -> str:
        """Labeli satu teks berdasarkan jumlah kata positif vs negatif."""
        text_lower = text.lower()
        pos_count = sum(1 for w in self.POSITIF_WORDS if w in text_lower)
        neg_count = sum(1 for w in self.NEGATIF_WORDS if w in text_lower)

        if neg_count > pos_count:
            return "negatif"
        elif pos_count > neg_count:
            return "positif"
        else:
            return "netral"

    def label_batch(self, texts: list) -> list:
        """Labeli batch teks."""
        return [self.label(t) for t in texts]


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

    def __init__(self, nb_type: str = "complement"):
        nb_clf = ComplementNB() if nb_type == "complement" else MultinomialNB()

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),        # unigram + bigram
                max_features=15000,         # top 15k fitur
                min_df=2,                   # minimal muncul di 2 dokumen
                max_df=0.85,                # abaikan kata terlalu umum (>85%)
                sublinear_tf=True,          # log TF untuk normalisasi
                analyzer="word"
            )),
            ("clf", nb_clf)
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
            "train_report": self.train_report
        }, path)
        print(f"  [SAVED] Model disimpan ke {path}")

    @classmethod
    def load(cls, path: str):
        """Load model dari file."""
        data = joblib.load(path)
        obj = cls()
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
    """Hitung persentase positif/negatif/netral per topik."""
    hasil = []
    for topik, group in df.groupby("topik"):
        total = len(group)
        counts = group[kolom_label].value_counts()

        positif = counts.get("positif", 0)
        negatif = counts.get("negatif", 0)
        netral  = counts.get("netral", 0)

        hasil.append({
            "topik": topik,
            "total_data": total,
            "negatif_count": negatif,
            "netral_count": netral,
            "positif_count": positif,
            "negatif_pct": round(negatif / total * 100, 2),
            "netral_pct": round(netral / total * 100, 2),
            "positif_pct": round(positif / total * 100, 2),
            "sentimen_dominan": counts.idxmax() if not counts.empty else "netral"
        })
    return pd.DataFrame(hasil)


def plot_distribusi_sentimen(df_ringkasan: pd.DataFrame, output_dir: str = "./hasil/plots"):
    """Bar chart distribusi sentimen per topik."""
    fig, ax = plt.subplots(figsize=(14, 7))

    topik_labels = [t.replace("_", "\n").title() for t in df_ringkasan["topik"]]
    x = np.arange(len(topik_labels))
    width = 0.25

    colors = {"negatif": "#e74c3c", "netral": "#f39c12", "positif": "#27ae60"}

    bars1 = ax.bar(x - width, df_ringkasan["negatif_pct"], width,
                   label="Negatif", color=colors["negatif"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, df_ringkasan["netral_pct"], width,
                   label="Netral", color=colors["netral"], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, df_ringkasan["positif_pct"], width,
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
    autolabel(bars3)

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

    colors_pie = ["#e74c3c", "#f39c12", "#27ae60"]
    label_order = ["negatif", "netral", "positif"]

    for i, topik in enumerate(topik_list):
        group = df[df["topik"] == topik]["sentimen"].value_counts()
        labels = group.index.tolist()
        sizes  = group.values.tolist()
        colors = [colors_pie[label_order.index(l)] if l in label_order else "#95a5a6"
                  for l in labels]

        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
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
    ax.set_title("Confusion Matrix — Naive Bayes\nAnalisis Sentimen Kebijakan Pemerintah",
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
        "netral": "Blues"
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
              f"{'Negatif':>10} {'Netral':>9} {'Positif':>9} {'Dominan':>9}")
    print(header)
    print("  " + "-" * 75)
    for _, row in df_ringkasan.iterrows():
        topik_display = row["topik"].replace("_", " ").title()[:27]
        print(f"  {topik_display:<26} {row['total_data']:>7} "
              f"{row['negatif_pct']:>8.2f}%  {row['netral_pct']:>7.2f}%  "
              f"{row['positif_pct']:>7.2f}%  {row['sentimen_dominan']:>8}")
    print("  " + "-" * 75)

    # Hitung total keseluruhan
    total_all = df_ringkasan["total_data"].sum()
    neg_all = df_ringkasan["negatif_count"].sum()
    neu_all = df_ringkasan["netral_count"].sum()
    pos_all = df_ringkasan["positif_count"].sum()
    print(f"  {'TOTAL KESELURUHAN':<26} {total_all:>7} "
          f"{neg_all/total_all*100:>8.2f}%  {neu_all/total_all*100:>7.2f}%  "
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
        ("prabowo_gibran", "pemerintahan prabowo gibran perlu waktu evaluasi lebih lanjut", "netral"),
        ("prabowo_gibran", "prabowo presiden terbaik indonesia makmur sejahtera", "positif"),
        ("prabowo_gibran", "kecewa dengan kabinet prabowo korupsi merajalela lagi", "negatif"),
        ("prabowo_gibran", "menunggu hasil kerja prabowo gibran satu tahun pertama", "netral"),
        ("prabowo_gibran", "program kerja prabowo pro rakyat kecil mendukung penuh", "positif"),
        ("prabowo_gibran", "prabowo bohong janji kampanye tidak ditepati rakyat kecewa", "negatif"),
        ("prabowo_gibran", "pemerintahan baru masih adaptasi wajar ada hambatan awal", "netral"),
        ("prabowo_gibran", "prabowo gibran melanjutkan pembangunan IKN dengan baik sekali", "positif"),
        ("prabowo_gibran", "harga sembako naik terus sejak prabowo menjabat susah hidup", "negatif"),
        ("prabowo_gibran", "kebijakan ekonomi prabowo masih dikaji dampaknya ke rakyat", "netral"),

        # Omnibus Law
        ("omnibus_law", "omnibus law cipta kerja merugikan buruh tidak pro rakyat kecil", "negatif"),
        ("omnibus_law", "omnibus law membuka lapangan kerja investasi meningkat pesat", "positif"),
        ("omnibus_law", "omnibus law masih perlu kajian mendalam dampak terhadap lingkungan", "netral"),
        ("omnibus_law", "tolak omnibus law hidup buruh terancam kebijakan deregulasi", "negatif"),
        ("omnibus_law", "omnibus law memudahkan izin usaha UMKM berkembang lebih cepat", "positif"),
        ("omnibus_law", "omnibus law masih diperdebatkan semua pihak perlu diskusi terbuka", "netral"),
        ("omnibus_law", "UU cipta kerja melanggar hak buruh upah minimum dihapus", "negatif"),
        ("omnibus_law", "investasi meningkat berkat omnibus law ekonomi tumbuh signifikan", "positif"),
        ("omnibus_law", "omnibus law perlu sosialisasi lebih luas kepada masyarakat buruh", "netral"),
        ("omnibus_law", "omnibus law merusak lingkungan izin tambang diberikan dengan mudah", "negatif"),
        ("omnibus_law", "omnibus law bikin iklim investasi kondusif mantap pemerintah", "positif"),
        ("omnibus_law", "implementasi omnibus law di daerah masih belum merata sempurna", "netral"),

        # Danantara
        ("danantara", "danantara investasi BUMN rawan korupsi tidak transparan bahaya", "negatif"),
        ("danantara", "danantara bisa dorong ekonomi nasional maju pesat", "positif"),
        ("danantara", "danantara masih dalam tahap pembentukan perlu dipantau terus", "netral"),
        ("danantara", "dana rakyat dipakai danantara tanpa pengawasan yang jelas", "negatif"),
        ("danantara", "danantara pemerintah berkomitmen perbaiki tata kelola BUMN profesional", "positif"),
        ("danantara", "danantara perlu regulasi yang kuat baru bisa jalan dengan baik", "netral"),
        ("danantara", "danantara proyek korupsi baru oligarki menguasai aset BUMN", "negatif"),
        ("danantara", "danantara bisa jadi solusi pengelolaan aset negara yang optimal", "positif"),
        ("danantara", "konsep danantara bagus tapi implementasi menentukan hasilnya", "netral"),
        ("danantara", "danantara ancaman privatisasi BUMN terselubung berbahaya", "negatif"),
        ("danantara", "danantara bikin BUMN lebih kompetitif di pasar global hebat", "positif"),
        ("danantara", "danantara masih baru belum bisa dinilai berhasil atau gagal", "netral"),

        # Makan Bergizi Gratis
        ("makan_bergizi_gratis", "program makan bergizi gratis sangat membantu siswa miskin", "positif"),
        ("makan_bergizi_gratis", "makan bergizi gratis amburadul banyak yang tidak layak makan", "negatif"),
        ("makan_bergizi_gratis", "MBG program bagus tapi implementasi perlu diperbaiki terus", "netral"),
        ("makan_bergizi_gratis", "makan bergizi gratis sia sia anggaran habis gizi tidak terpenuhi", "negatif"),
        ("makan_bergizi_gratis", "anak sekolah senang dapat makan bergizi gratis tiap hari sehat", "positif"),
        ("makan_bergizi_gratis", "program MBG baru berjalan evaluasi terus dilakukan pemerintah", "netral"),
        ("makan_bergizi_gratis", "makan gratis sekolah mengurangi stunting generasi muda sehat kuat", "positif"),
        ("makan_bergizi_gratis", "kualitas makan bergizi gratis buruk tidak layak konsumsi parah", "negatif"),
        ("makan_bergizi_gratis", "makan bergizi gratis memerlukan pengawasan distribusi ketat", "netral"),
        ("makan_bergizi_gratis", "MBG program terbaik prabowo anak sekolah jadi semangat belajar", "positif"),
        ("makan_bergizi_gratis", "makanan MBG sering basi anak sekolah sakit perut keracunan", "negatif"),
        ("makan_bergizi_gratis", "MBG masih tahap awal wajar ada kendala teknis di lapangan", "netral"),

        # Efisiensi Anggaran
        ("efisiensi_anggaran", "pemangkasan anggaran merugikan pendidikan kesehatan rakyat kecil", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran diperlukan untuk mengurangi pemborosan negara", "positif"),
        ("efisiensi_anggaran", "efisiensi anggaran 2025 perlu kajian dampak sosial lebih lanjut", "netral"),
        ("efisiensi_anggaran", "pemotongan anggaran gaji guru dipotong sangat kecewa sekali", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran pemerintah hemat dorong produktivitas tinggi", "positif"),
        ("efisiensi_anggaran", "anggaran 2025 dipangkas dampaknya masih belum terasa langsung", "netral"),
        ("efisiensi_anggaran", "efisiensi anggaran birokrasi ramping korupsi berkurang signifikan", "positif"),
        ("efisiensi_anggaran", "pemotongan anggaran pelayanan publik menurun drastis parah", "negatif"),
        ("efisiensi_anggaran", "kebijakan efisiensi anggaran masih perlu dievaluasi dampaknya", "netral"),
        ("efisiensi_anggaran", "anggaran sosial dipotong rakyat kecil semakin susah hidup", "negatif"),
        ("efisiensi_anggaran", "efisiensi anggaran bikin pejabat gak bisa foya pakai uang negara", "positif"),
        ("efisiensi_anggaran", "efisiensi anggaran perlu keseimbangan jangan asal potong saja", "netral"),
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
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporsi test set (default: 0.2 = 80/20 split)")
    parser.add_argument("--no-stem", action="store_true",
                        help="Nonaktifkan stemming Sastrawi (lebih cepat)")
    parser.add_argument("--prediksi", type=str, default=None,
                        help="Prediksi sentimen 1 kalimat baru")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Load model tersimpan dari file .pkl")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  ANALISIS SENTIMEN KEBIJAKAN PEMERINTAH INDONESIA")
    print("  Metode: Naive Bayes + TF-IDF")
    print("  Topik : 5 Kebijakan Pemerintah")
    print("=" * 65)

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
            print("  [INFO] Kolom label tidak ditemukan. Melabeli otomatis dengan lexicon...")
            labeler = LexiconLabeler()
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
        "neutral": "netral", "neu": "netral"
    })
    df = df[df["sentimen"].isin(["positif", "negatif", "netral"])].reset_index(drop=True)
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
    model = NaiveBayesSentimen(nb_type=args.nb_type)
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

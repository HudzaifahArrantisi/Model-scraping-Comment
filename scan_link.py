"""
==============================================================
SCRIPT 3: LINK SENTIMENT SCANNER (YouTube Comments)
          Scanning & Sentiment Analysis dari Link YouTube Real
==============================================================
Deskripsi:
  Script ini menerima input link/URL video YouTube, mengunduh
  komentarnya secara real-time (tanpa butuh API key), lalu
  meminta input nama file dari user, memproses komentar tersebut,
  dan mengklasifikasikan sentimennya menggunakan model Naive Bayes.

Output:
  Semua hasil akan disimpan ke folder `./hasil_scanning/`:
  - ./hasil_scanning/[nama_file]_raw.csv       -> Komentar mentah hasil download
  - ./hasil_scanning/[nama_file].csv           -> Komentar + prediksi sentimen
  - ./hasil_scanning/plots/[nama_file]_pie.png -> Grafik diagram lingkaran sentimen
  - ./hasil_scanning/plots/[nama_file]_wordcloud_[sentimen].png -> Word Cloud
==============================================================
"""

import os
import sys
import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

# Import preprocessor dan model dari script naive_bayes_sentimen
try:
    from naive_bayes_sentimen import (
        PreprocessorIndonesia,
        NaiveBayesSentimen,
        SmartSentimentLabeler,
        pilih_kolom_teks,
    )
except ImportError:
    print("\n  [ERROR] File naive_bayes_sentimen.py tidak ditemukan di folder ini.")
    sys.exit(1)

# Folder output scan utama sesuai request
os.makedirs("./hasil_scanning", exist_ok=True)
os.makedirs("./hasil_scanning/plots", exist_ok=True)


def extract_video_title_from_url(url: str) -> str:
    """Mengambil ID video atau judul ringkas dari URL YouTube."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    return "youtube_video"


def scrape_youtube_comments(url: str, limit: int = 500) -> pd.DataFrame:
    """Mengunduh komentar dari URL YouTube menggunakan youtube-comment-downloader."""
    if not ("youtube.com" in url or "youtu.be" in url):
        print("\n  [ERROR] Link yang dimasukkan bukan link YouTube!")
        print("          Saat ini scanning link otomatis hanya mendukung platform YouTube.")
        print("          TikTok dan Twitter/X memiliki keamanan ketat yang memblokir program otomatis tanpa login browser.")
        return pd.DataFrame()

    print(f"\n  [1/4] Mengunduh komentar dari: {url}")
    print(f"        (Membatasi hingga maksimal {limit} komentar)...")

    try:
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
    except ImportError:
        print("  [ERROR] youtube-comment-downloader tidak terinstall.")
        print("          Jalankan: pip install youtube-comment-downloader")
        return pd.DataFrame()

    downloader = YoutubeCommentDownloader()
    comments_list = []
    
    try:
        generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        pbar = tqdm(total=limit, desc="  Mendownload")
        
        for comment in generator:
            if len(comments_list) >= limit:
                break
            
            comments_list.append({
                "comment_id": comment.get("cid"),
                "author": comment.get("author"),
                "text": comment.get("text"),
                "time": comment.get("time"),
                "votes": comment.get("votes"),
                "photo": comment.get("photo")
            })
            pbar.update(1)
            
        pbar.close()
        
    except Exception as e:
        print(f"\n  [ERROR] Gagal mengunduh komentar: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(comments_list)
    if df.empty:
        print("  [WARN] Tidak ada komentar yang berhasil diunduh.")
    else:
        print(f"  [SUCCESS] Berhasil mengunduh {len(df)} komentar!")
    
    return df


def plot_scan_sentimen(df: pd.DataFrame, file_prefix: str, output_dir: str = "./hasil_scanning/plots"):
    """Membuat visualisasi pie chart untuk hasil scan."""
    counts = df["sentimen_pred"].value_counts()
    
    # Warna estetis
    colors = {"negatif": "#e74c3c", "netral": "#95a5a6", "positif": "#2ecc71"}
    plot_colors = [colors.get(label, "#3498db") for label in counts.index]
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.pie(
        counts, 
        labels=[f"{label.upper()} ({count} komentar)" for label, count in zip(counts.index, counts.values)],
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plot_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
    )
    
    title_display = file_prefix.replace("_", " ").title()
    plt.title(f"Analisis Sentimen Komentar Masyarakat\nTopik: {title_display}", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{file_prefix}_pie.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  [SAVED] Pie chart sentimen disimpan -> {output_path}")


def generate_scan_wordclouds(df: pd.DataFrame, file_prefix: str, output_dir: str = "./hasil_scanning/plots"):
    """Membuat Word Cloud dari hasil scanning sentimen."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [SKIP] wordcloud tidak terinstall. Word Cloud dilewati.")
        return

    sentimen_colors = {
        "negatif": "Reds",
        "positif": "Greens",
        "netral": "Blues"
    }

    for sentimen, colormap in sentimen_colors.items():
        texts = df[df["sentimen_pred"] == sentimen]["text_clean"]
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
        title_display = file_prefix.replace("_", " ").title()
        plt.title(f"Word Cloud - Sentimen {sentimen.capitalize()} ({title_display})", fontsize=14, fontweight="bold")
        path = os.path.join(output_dir, f"{file_prefix}_wordcloud_{sentimen}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] Word Cloud {sentimen} -> {path}")


def normalize_sentiment_labels(series: pd.Series) -> pd.Series:
    """Normalisasi label sentimen ke format internal."""
    return series.fillna("").astype(str).str.lower().str.strip().replace({
        "negative": "negatif", "neg": "negatif",
        "positive": "positif", "pos": "positif",
        "neutral": "netral", "neu": "netral",
    })


def find_label_column(df: pd.DataFrame) -> str | None:
    """Cari kolom label sentimen dari dataset training."""
    for col in ["sentimen", "sentimen_manual", "label", "sentimen_auto"]:
        if col in df.columns:
            return col
    return None


def train_model_from_dataset(dataset_path: str, model_path: str, use_stemming: bool = True) -> NaiveBayesSentimen | None:
    """Latih model dari dataset lokal agar scanner belajar dari data sebelumnya."""
    if not os.path.exists(dataset_path):
        print(f"        [WARN] Dataset training tidak ditemukan: {dataset_path}")
        return None

    print(f"        Membaca dataset training: {dataset_path}")
    df_train = pd.read_csv(dataset_path, encoding="utf-8-sig")
    if df_train.empty:
        print("        [WARN] Dataset training kosong.")
        return None

    text_col = pilih_kolom_teks(df_train)
    label_col = find_label_column(df_train)

    if label_col is None:
        print("        Kolom label tidak ada. Membuat label otomatis dari SmartSentimentLabeler...")
        labeler = SmartSentimentLabeler()
        df_train["sentimen_auto"] = labeler.label_batch(df_train[text_col].fillna("").astype(str).tolist())
        label_col = "sentimen_auto"

    df_train["sentimen"] = normalize_sentiment_labels(df_train[label_col])
    df_train = df_train[df_train["sentimen"].isin(["positif", "negatif", "netral"])].reset_index(drop=True)
    if df_train.empty or df_train["sentimen"].nunique() < 2:
        print("        [WARN] Dataset training harus punya minimal 2 kelas sentimen valid.")
        return None

    preprocessor = PreprocessorIndonesia(use_stemming=use_stemming)
    print("        Melatih model dari data yang sudah ada...")
    df_train["text_clean"] = preprocessor.preprocess_batch(df_train[text_col].fillna("").astype(str).tolist())
    df_train = df_train[df_train["text_clean"].str.strip() != ""].reset_index(drop=True)

    if len(df_train) < 5:
        print("        [WARN] Dataset training terlalu sedikit untuk melatih model.")
        return None

    model = NaiveBayesSentimen(nb_type="complement")
    model.train(df_train["text_clean"].tolist(), df_train["sentimen"].tolist())

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    return model


def load_or_train_model(model_path: str, dataset_path: str, force_retrain: bool, use_stemming: bool = True) -> NaiveBayesSentimen | None:
    """Load model tersimpan, atau latih ulang otomatis dari dataset jika perlu."""
    model_exists = os.path.exists(model_path)
    dataset_exists = os.path.exists(dataset_path)
    dataset_newer = dataset_exists and model_exists and os.path.getmtime(dataset_path) > os.path.getmtime(model_path)

    if force_retrain or not model_exists or dataset_newer:
        if force_retrain:
            print("        Retrain diminta user (--retrain).")
        elif not model_exists:
            print("        Model belum ada. Scanner akan melatih model otomatis dari dataset.")
        else:
            print("        Dataset lebih baru dari model. Scanner akan update model otomatis.")

        model = train_model_from_dataset(dataset_path, model_path, use_stemming=use_stemming)
        if model is not None:
            return model

        if not model_exists:
            return None
        print("        [WARN] Training otomatis gagal. Memakai model lama yang tersedia.")

    print(f"        Memuat model ML terlatih: {model_path}")
    return NaiveBayesSentimen.load(model_path)


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


def predict_comments_with_learning_model(
    df_comments: pd.DataFrame,
    model: NaiveBayesSentimen | None,
    confidence_threshold: float,
    min_known_ratio: float,
) -> pd.DataFrame:
    """Prediksi komentar dengan model ML, lalu fallback untuk teks yang kurang dipahami."""
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

    for text_clean, pred, score, ratio in zip(texts_clean, pred_labels, max_probas, known_ratios):
        if score < confidence_threshold or ratio < min_known_ratio:
            detail = fallback_labeler.label_with_detail(text_clean)
            final_labels.append(detail["label"])
            confidence_names.append(detail["confidence"])
            methods.append("fallback_smart_labeler")
        else:
            final_labels.append(pred)
            if score >= 0.70:
                confidence_names.append("tinggi")
            elif score >= confidence_threshold:
                confidence_names.append("sedang")
            else:
                confidence_names.append("rendah")
            methods.append("model_nb")

    df_comments["sentimen_pred"] = final_labels
    df_comments["confidence"] = confidence_names
    df_comments["confidence_score"] = max_probas
    df_comments["known_word_ratio"] = known_ratios
    df_comments["metode_prediksi"] = methods
    return df_comments


def main():
    parser = argparse.ArgumentParser(description="Scan sentimen komentar real-time dari link YouTube.")
    parser.add_argument("url_pos", type=str, nargs="?", help="URL/Link video YouTube (posisi argument pertama).")
    parser.add_argument("--url", type=str, help="URL/Link video YouTube (opsi flag).")
    parser.add_argument("--limit", type=int, default=300, help="Batas jumlah komentar maksimal (default: 300).")
    parser.add_argument("--model", type=str, default="./hasil/model_nb.pkl", help="Path ke model Naive Bayes terlatih.")
    parser.add_argument("--dataset", type=str, default="./data_raw/semua_topik.csv", help="Dataset training untuk update model otomatis.")
    parser.add_argument("--retrain", action="store_true", help="Paksa latih ulang model dari dataset sebelum scanning.")
    parser.add_argument("--confidence-threshold", type=float, default=0.45, help="Ambang confidence model sebelum fallback smart labeler.")
    parser.add_argument("--min-known-ratio", type=float, default=0.25, help="Rasio minimal kata yang dikenal model sebelum fallback.")
    args = parser.parse_args()

    url = args.url or args.url_pos
    if not url:
        print("\n=== LINK SENTIMENT SCANNER (YouTube Comments) ===")
        url = input("  Masukkan URL/Link Video YouTube: ").strip()
        if not url:
            print("  [ERROR] URL tidak boleh kosong!")
            sys.exit(1)

    # 1. Download Komentar Mentah
    df_comments = scrape_youtube_comments(url, limit=args.limit)
    if df_comments.empty:
        sys.exit(1)

    # 2. Minta Nama File Hasil Scan dari User
    print("\n  [2/4] Penamaan Berkas Hasil Scan")
    video_id = extract_video_title_from_url(url)
    default_name = f"scan_{video_id}"
    
    file_name_input = input(f"        Masukkan nama file (default: '{default_name}'): ").strip()
    if not file_name_input:
        file_name_input = default_name
    else:
        # Bersihkan karakter ilegal untuk nama file
        file_name_input = re.sub(r'[\\/*?:""<>| ]', '_', file_name_input)
        if file_name_input.endswith(".csv"):
            file_name_input = file_name_input[:-4]

    # Simpan CSV Mentah (Raw)
    raw_path = f"./hasil_scanning/{file_name_input}_raw.csv"
    df_comments.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"        [SAVED] Komentar mentah disimpan -> {raw_path}")

    # 3. Kirim ke Pipeline Preprocessing & Model ML
    print("\n  [3/4] Mengirimkan data ke pipeline & memfilter sentimen...")
    model = load_or_train_model(
        model_path=args.model,
        dataset_path=args.dataset,
        force_retrain=args.retrain,
        use_stemming=True,
    )

    preprocessor = PreprocessorIndonesia(use_stemming=True)

    print("        Melakukan prapemrosesan teks komentar (slang, stemming)...")
    df_comments["text_clean"] = [preprocessor.preprocess(t) for t in tqdm(df_comments["text"], desc="  Processing")]

    # Buang data yang setelah clean menjadi kosong
    df_comments = df_comments[df_comments["text_clean"].str.strip() != ""].reset_index(drop=True)
    if df_comments.empty:
        print("  [WARN] Semua komentar kosong setelah pembersihan (tidak dapat difilter).")
        sys.exit(1)

    # Prediksi kategori sentimen menggunakan model yang sudah belajar dari dataset.
    print("        Menganalisis sentimen menggunakan model ML + fallback kata tidak dikenal...")
    df_comments = predict_comments_with_learning_model(
        df_comments=df_comments,
        model=model,
        confidence_threshold=args.confidence_threshold,
        min_known_ratio=args.min_known_ratio,
    )

    # Simpan CSV Hasil Analisis yang sudah Difilter
    filtered_path = f"./hasil_scanning/{file_name_input}.csv"
    df_comments.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    print(f"        [SAVED] Komentar hasil filter disimpan -> {filtered_path}")

    # 4. Membuat Gambar Diagram (Visualisasi) ke Folder Hasil Scanning
    print("\n  [4/4] Membuat visualisasi hasil scanning...")
    plot_scan_sentimen(df_comments, file_name_input)
    generate_scan_wordclouds(df_comments, file_name_input)

    # Tampilkan Ringkasan di Terminal
    counts = df_comments["sentimen_pred"].value_counts()
    total = len(df_comments)
    
    print("\n" + "=" * 65)
    print("  RINGKASAN SCANNING SENTIMEN")
    print("=" * 65)
    print(f"  Topik File      : {file_name_input}")
    print(f"  Total Komentar  : {total} komentar (valid)")
    print(f"  Model           : {args.model if model is not None else 'fallback SmartSentimentLabeler'}")
    print("-" * 65)
    
    for label in ["positif", "netral", "negatif"]:
        cnt = counts.get(label, 0)
        pct = (cnt / total) * 100
        bar = "=" * int(pct / 4)
        print(f"  {label.upper():<10} : {cnt:>4} komentar ({pct:>6.2f}%) {bar}")

    if "metode_prediksi" in df_comments.columns:
        method_counts = df_comments["metode_prediksi"].value_counts().to_dict()
        fallback_count = method_counts.get("fallback_smart_labeler", 0) + method_counts.get("smart_labeler", 0)
        print("-" * 65)
        print(f"  Prediksi ML     : {method_counts.get('model_nb', 0)} komentar")
        print(f"  Fallback pintar : {fallback_count} komentar")
        
    print("=" * 65)
    print(f"\n  [DONE] Scanning selesai! Hasil tersimpan di folder './hasil_scanning/'")
    print("=" * 65 + "\n")

    # 5. Menu Interaktif Melihat Komentar
    while True:
        print("\n  =========================================")
        print("  PILIHAN LIHAT KOMENTAR BERDASARKAN SENTIMEN")
        print("  =========================================")
        print("  [1] Tampilkan Komentar POSITIF")
        print("  [2] Tampilkan Komentar NEGATIF")
        print("  [3] Tampilkan Komentar NETRAL")
        print("  [0] Keluar")
        
        pilihan = input("  Masukkan pilihan Anda (0-3): ").strip()
        
        if pilihan == '0':
            print("  Terima kasih telah menggunakan scanner ini!")
            break
        elif pilihan in ['1', '2', '3']:
            label_map = {'1': 'positif', '2': 'negatif', '3': 'netral'}
            sentimen_pilihan = label_map[pilihan]
            
            # Filter data sesuai pilihan
            df_filtered = df_comments[df_comments["sentimen_pred"] == sentimen_pilihan]
            jumlah = len(df_filtered)
            
            print(f"\n  --- MENAMPILKAN {jumlah} KOMENTAR {sentimen_pilihan.upper()} ---")
            if jumlah == 0:
                print("  (Tidak ada komentar dalam kategori ini)")
            else:
                # Tampilkan maksimal 50 komentar agar terminal tidak terlalu penuh
                limit_tampil = min(50, jumlah)
                
                # Header Tabel
                print("  " + "-" * 115)
                print(f"  | {'NO':<3} | {'NAMA AKUN':<25} | {'ISI KOMENTAR':<75} |")
                print("  " + "-" * 115)
                
                for idx, row in enumerate(df_filtered.head(limit_tampil).itertuples(), 1):
                    teks_asli = str(row.text).replace('\n', ' ')
                    # Memotong teks jika terlalu panjang agar tabel tetap rapi
                    akun = str(row.author)[:25] if pd.notna(row.author) else "Unknown"
                    komentar = teks_asli[:72] + "..." if len(teks_asli) > 75 else teks_asli
                    
                    print(f"  | {idx:<3} | {akun:<25} | {komentar:<75} |")
                
                print("  " + "-" * 115)
                
                if jumlah > 50:
                    print(f"  ... dan {jumlah - 50} komentar lainnya (lihat file CSV untuk lengkapnya).")
            
            input("\n  [Tekan Enter untuk kembali ke menu...]")
        else:
            print("  [ERROR] Pilihan tidak valid. Masukkan angka 0, 1, 2, atau 3.")


if __name__ == "__main__":
    main()

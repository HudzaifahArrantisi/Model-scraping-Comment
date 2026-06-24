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

# Import preprocessor dari script naive_bayes_sentimen
try:
    from naive_bayes_sentimen import PreprocessorIndonesia, NaiveBayesSentimen
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


def main():
    parser = argparse.ArgumentParser(description="Scan sentimen komentar real-time dari link YouTube.")
    parser.add_argument("url_pos", type=str, nargs="?", help="URL/Link video YouTube (posisi argument pertama).")
    parser.add_argument("--url", type=str, help="URL/Link video YouTube (opsi flag).")
    parser.add_argument("--limit", type=int, default=300, help="Batas jumlah komentar maksimal (default: 300).")
    parser.add_argument("--model", type=str, default="./hasil/model_nb.pkl", help="Path ke model Naive Bayes terlatih.")
    args = parser.parse_args()

    # Cek model terlatih
    if not os.path.exists(args.model):
        print(f"\n  [ERROR] Model Naive Bayes terlatih tidak ditemukan di: {args.model}")
        print("          Harap jalankan analisis utama terlebih dahulu untuk melatih model:")
        print("          python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv")
        sys.exit(1)

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

    # 3. Kirim ke Pipeline Preprocessing & Naive Bayes Filter
    print("\n  [3/4] Mengirimkan data ke pipeline & memfilter sentimen...")
    print(f"        Memuat model Naive Bayes dari {args.model}...")
    model = NaiveBayesSentimen.load(args.model)
    preprocessor = PreprocessorIndonesia(use_stemming=True)

    print("        Melakukan prapemrosesan teks komentar (slang, stemming)...")
    df_comments["text_clean"] = [preprocessor.preprocess(t) for t in tqdm(df_comments["text"], desc="  Processing")]

    # Buang data yang setelah clean menjadi kosong
    df_comments = df_comments[df_comments["text_clean"].str.strip() != ""].reset_index(drop=True)
    if df_comments.empty:
        print("  [WARN] Semua komentar kosong setelah pembersihan (tidak dapat difilter).")
        sys.exit(1)

    # Prediksi Kategori Sentimen (Positif, Negatif, Netral)
    df_comments["sentimen_pred"] = model.predict(df_comments["text_clean"].tolist())

    # Ambil nilai probabilitas sentimen
    proba = model.predict_proba(df_comments["text_clean"].tolist())
    for i, cls in enumerate(model.classes_):
        df_comments[f"prob_{cls}"] = proba[:, i]

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
    print("-" * 65)
    
    for label in ["positif", "netral", "negatif"]:
        cnt = counts.get(label, 0)
        pct = (cnt / total) * 100
        bar = "=" * int(pct / 4)
        print(f"  {label.upper():<10} : {cnt:>4} komentar ({pct:>6.2f}%) {bar}")
        
    print("=" * 65)
    print(f"\n  [DONE] Scanning selesai! Hasil tersimpan di folder './hasil_scanning/'")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

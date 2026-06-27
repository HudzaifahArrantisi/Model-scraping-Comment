"""
==============================================================
SCRIPT: FILTER KOMENTAR DARI FILE (CSV Sentiment Viewer)
==============================================================
Deskripsi:
  Script ini digunakan untuk membaca file CSV hasil scanning sentimen
  (baik dari folder ./hasil_scanning/ maupun ./hasil/) dan menampilkan
  komentarnya secara interaktif berdasarkan kategori sentimen
  (Positif, Negatif, Netral) atau filter kata kunci.

Cara Pakai:
  python filter_file.py
  python filter_file.py --cari "danantara"
  python filter_file.py --cari "prabowo gibran" --sentimen positif
  python filter_file.py --cari "makan bergizi" --limit 20
==============================================================
"""

import os
import sys
import re
import argparse
import pandas as pd

# Konfigurasi kandidat kolom
TEXT_COLUMN_CANDIDATES = [
    "text", "teks", "comment", "komentar", "caption", "content",
    "full_text", "tweet", "body", "message", "text_clean"
]

SENTIMENT_COLUMN_CANDIDATES = [
    "sentimen_pred", "sentimen", "label", "sentimen_manual", "sentimen_auto", "prediksi"
]


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def pilih_kolom_teks(df: pd.DataFrame) -> str | None:
    """Cari kolom teks di dalam DataFrame."""
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def pilih_kolom_sentimen(df: pd.DataFrame) -> str | None:
    """Cari kolom sentimen di dalam DataFrame."""
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in SENTIMENT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def list_csv_files():
    """Mendata semua file CSV yang ada di folder hasil_scanning, hasil, dan root."""
    paths = [
        ("./hasil_scanning", "Hasil Scanning"),
        ("./hasil", "Hasil Utama"),
        (".", "Folder Utama")
    ]
    
    csv_files = []
    
    for folder, desc in paths:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".csv") and not file.endswith("_raw.csv"): # skip _raw karena belum berlabel
                    csv_files.append((os.path.join(folder, file), desc))
                    
    return csv_files


def main():
    parser = argparse.ArgumentParser(description="Filter komentar dari CSV berdasarkan kata kunci dan/atau sentimen")
    parser.add_argument("--cari", "-c", type=str, default=None,
                        help="Kata kunci untuk filter komentar (case-insensitive, bisa multi kata)")
    parser.add_argument("--sentimen", "-s", type=str, default=None,
                        choices=["positif", "negatif", "netral", "pos", "neg"],
                        help="Filter berdasarkan sentimen")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Jumlah maksimal komentar yang ditampilkan")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Langsung tentukan path file CSV (skip menu pilih file)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Simpan hasil filter ke file CSV baru")
    args = parser.parse_args()
    
    clear_screen()
    print("=========================================================")
    print("         CSV COMMENT FILTER & SENTIMENT VIEWER           ")
    print("=========================================================")
    
    # Mapping sentimen
    sentimen_map = {"positif": "positif", "pos": "positif",
                    "negatif": "negatif", "neg": "negatif",
                    "netral": "netral"}
    sentimen_target = sentimen_map.get(args.sentimen) if args.sentimen else None
    
    # 1. Pilih File CSV
    is_cli_mode = bool(args.cari or args.sentimen or args.limit is not None or args.output)

    if args.file:
        if not os.path.exists(args.file):
            print(f"\n  [ERROR] File '{args.file}' tidak ditemukan.")
            sys.exit(1)
        selected_path = args.file
        filename = os.path.basename(selected_path)
    else:
        files = list_csv_files()
        if not files:
            print("\n  [WARN] Tidak ditemukan file CSV hasil analisis sentimen.")
            print("          Pastikan Anda sudah melakukan scanning terlebih dahulu.")
            sys.exit(0)
            
        if is_cli_mode:
            # CLI mode: auto-pilih file pertama (non-interaktif)
            selected_path = files[0][0]
            filename = os.path.basename(selected_path)
            print(f"\n  [INFO] Menggunakan file: {filename}")
        else:
            print("\n  Daftar file CSV yang tersedia:")
            for idx, (path, desc) in enumerate(files, 1):
                fname = os.path.basename(path)
                print(f"  [{idx}] {fname:<30} (Folder: {desc})")
            print("  [0] Keluar")
            
            try:
                pilihan_file = int(input(f"\n  Pilih nomor file yang ingin dilihat (0-{len(files)}): ").strip())
            except ValueError:
                print("  [ERROR] Input harus berupa angka.")
                sys.exit(1)
            if pilihan_file == 0:
                print("  Keluar program.")
                sys.exit(0)
            if pilihan_file < 1 or pilihan_file > len(files):
                print("  [ERROR] Pilihan tidak tersedia.")
                sys.exit(1)
            selected_path = files[pilihan_file - 1][0]
            filename = os.path.basename(selected_path)
    
    # 2. Baca File
    try:
        df = pd.read_csv(selected_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [ERROR] Gagal membaca file: {e}")
        sys.exit(1)
        
    if df.empty:
        print(f"  [ERROR] File `{filename}` kosong.")
        sys.exit(1)
        
    # 3. Cari Kolom yang Tepat
    text_col = pilih_kolom_teks(df)
    sentimen_col = pilih_kolom_sentimen(df)
    
    if not text_col:
        print("  [ERROR] Tidak dapat menemukan kolom teks/komentar di file ini.")
        print(f"          Kolom tersedia: {df.columns.tolist()}")
        sys.exit(1)
        
    if not sentimen_col:
        print("  [ERROR] Tidak dapat menemukan kolom label/sentimen di file ini.")
        print(f"          Kolom tersedia: {df.columns.tolist()}")
        sys.exit(1)
        
    # Normalisasi data sentimen di kolom
    df[sentimen_col] = df[sentimen_col].fillna("").astype(str).str.lower().str.strip()
    df[sentimen_col] = df[sentimen_col].replace({
        "positive": "positif", "pos": "positif",
        "negative": "negatif", "neg": "negatif",
        "neutral": "netral", "neu": "netral",
    })
    
    # Author column (optional)
    author_col = None
    for candidate in ["author", "nama", "username", "user", "nama_akun", "pengirim"]:
        for col in df.columns:
            if col.lower() == candidate:
                author_col = col
                break
        if author_col:
            break

    # 4. Filter berdasarkan kata kunci (--cari)
    keyword_filter_applied = False
    if args.cari:
        keyword = args.cari.strip().lower()
        print(f"\n  [FILTER] Mencari komentar mengandung: '{args.cari}'")
        # Cari di kolom teks (case-insensitive)
        mask = df[text_col].astype(str).str.lower().str.contains(keyword, na=False)
        df = df[mask].reset_index(drop=True)
        keyword_filter_applied = True
        print(f"  [FILTER] Ditemukan {len(df)} komentar.")
    
    # 5. Filter berdasarkan sentimen (--sentimen)
    if sentimen_target:
        print(f"\n  [FILTER] Menampilkan sentimen: {sentimen_target.upper()}")
        df = df[df[sentimen_col] == sentimen_target].reset_index(drop=True)
        print(f"  [FILTER] Tersisa {len(df)} komentar.")
    
    # 6. Tampilkan Ringkasan Singkat File
    counts = df[sentimen_col].value_counts()
    total = len(df)
    
    if total == 0:
        print("\n  [INFO] Tidak ada komentar yang cocok dengan filter.")
        if not is_cli_mode:
            input("\n  Tekan Enter untuk keluar...")
        sys.exit(0)
    
    clear_screen()
    print("=========================================================")
    print(f"  FILE : {filename}")
    print(f"  Total Data : {total} baris")
    if keyword_filter_applied:
        print(f"  FILTER KATA KUNCI : '{args.cari}'")
    print("=========================================================")
    for label in ["positif", "netral", "negatif"]:
        cnt = counts.get(label, 0)
        pct = (cnt / total) * 100 if total > 0 else 0
        bar = "=" * int(pct / 4)
        print(f"  {label.upper():<10} : {cnt:>4} komentar ({pct:>6.2f}%) {bar}")
    print("=========================================================")
    
    # 6a. Jika --output, simpan langsung tanpa menu interaktif
    if args.output:
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n  [OUTPUT] Hasil filter disimpan ke: {args.output}")
        print("  Selesai.")
        return
    
    # Jika CLI mode (ada args), tampilkan hasil langsung tanpa menu interaktif
    if args.cari or args.sentimen or args.file or args.limit:
        limit_tampil = min(args.limit or 50, len(df), 50)
        print(f"\n  --- HASIL FILTER ({total} komentar, ditampilkan {limit_tampil}) ---")
        print("  " + "-" * 115)
        print(f"  | {'NO':<3} | {'NAMA AKUN':<25} | {'ISI KOMENTAR':<75} |")
        print("  " + "-" * 115)
        
        for idx, row in enumerate(df.head(limit_tampil).itertuples(), 1):
            teks_raw = getattr(row, text_col)
            teks_asli = str(teks_raw).replace('\n', ' ') if pd.notna(teks_raw) else ""
            akun_raw = getattr(row, author_col) if author_col else "Anonymous"
            akun = str(akun_raw)[:25] if pd.notna(akun_raw) else "Anonymous"
            komentar = teks_asli[:72] + "..." if len(teks_asli) > 75 else teks_asli
            print(f"  | {idx:<3} | {akun:<25} | {komentar:<75} |")
        
        print("  " + "-" * 115)
        if total > limit_tampil:
            print(f"  ... dan {total - limit_tampil} komentar lainnya.")
        print()
        return
    
    # 7. Menu Interaktif Filter Sentimen
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
            print("\n  Terima kasih telah menggunakan filter ini!")
            break
        elif pilihan in ['1', '2', '3']:
            label_map = {'1': 'positif', '2': 'negatif', '3': 'netral'}
            sentimen_pilihan = label_map[pilihan]
            
            # Filter
            df_filtered = df[df[sentimen_col] == sentimen_pilihan]
            jumlah = len(df_filtered)
            
            print(f"\n  --- MENAMPILKAN KOMENTAR {sentimen_pilihan.upper()} (Total: {jumlah}) ---")
            if jumlah == 0:
                print("  (Tidak ada komentar dalam kategori ini)")
            else:
                # Tampilkan maksimal 50 komentar
                limit_tampil = min(50, jumlah)
                
                # Header Tabel
                print("  " + "-" * 115)
                print(f"  | {'NO':<3} | {'NAMA AKUN':<25} | {'ISI KOMENTAR':<75} |")
                print("  " + "-" * 115)
                
                for idx, row in enumerate(df_filtered.head(limit_tampil).itertuples(), 1):
                    # Ambil teks asli
                    teks_raw = getattr(row, text_col)
                    teks_asli = str(teks_raw).replace('\n', ' ') if pd.notna(teks_raw) else ""
                    
                    # Ambil author
                    akun_raw = getattr(row, author_col) if author_col else "Anonymous"
                    akun = str(akun_raw)[:25] if pd.notna(akun_raw) else "Anonymous"
                    
                    # Potong jika terlalu panjang
                    komentar = teks_asli[:72] + "..." if len(teks_asli) > 75 else teks_asli
                    
                    print(f"  | {idx:<3} | {akun:<25} | {komentar:<75} |")
                
                print("  " + "-" * 115)
                
                if jumlah > 50:
                    print(f"  ... dan {jumlah - 50} komentar lainnya (silakan cek file CSV langsung untuk daftar lengkap).")
            
            input("\n  [Tekan Enter untuk kembali ke menu...]")
        else:
            print("  [ERROR] Pilihan tidak valid. Masukkan angka 0, 1, 2, atau 3.")


if __name__ == "__main__":
    main()

"""
==============================================================
SCRIPT: FILTER KOMENTAR DARI FILE (CSV Sentiment Viewer)
==============================================================
Deskripsi:
  Script ini digunakan untuk membaca file CSV hasil scanning sentimen
  (baik dari folder ./hasil_scanning/ maupun ./hasil/) dan menampilkan
  komentarnya secara interaktif berdasarkan kategori sentimen
  (Positif, Negatif, Netral).

Cara Pakai:
  python filter_file.py
==============================================================
"""

import os
import sys
import re
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
    clear_screen()
    print("=========================================================")
    print("         CSV COMMENT FILTER & SENTIMENT VIEWER           ")
    print("=========================================================")
    
    # 1. Pilih File CSV
    files = list_csv_files()
    if not files:
        print("\n  [WARN] Tidak ditemukan file CSV hasil analisis sentimen.")
        print("          Pastikan Anda sudah melakukan scanning terlebih dahulu.")
        input("\n  Tekan Enter untuk keluar...")
        sys.exit(0)
        
    print("\n  Daftar file CSV yang tersedia:")
    for idx, (path, desc) in enumerate(files, 1):
        filename = os.path.basename(path)
        print(f"  [{idx}] {filename:<30} (Folder: {desc})")
        
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

    # 4. Tampilkan Ringkasan Singkat File
    counts = df[sentimen_col].value_counts()
    total = len(df)
    
    clear_screen()
    print("=========================================================")
    print(f"  FILE : {filename}")
    print(f"  Total Data : {total} baris")
    print("=========================================================")
    for label in ["positif", "netral", "negatif"]:
        cnt = counts.get(label, 0)
        pct = (cnt / total) * 100 if total > 0 else 0
        bar = "=" * int(pct / 4)
        print(f"  {label.upper():<10} : {cnt:>4} komentar ({pct:>6.2f}%) {bar}")
    print("=========================================================")
    
    # 5. Menu Interaktif Filter Sentimen
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

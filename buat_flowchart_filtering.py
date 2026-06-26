"""
Flowchart Alur Filtering Komentar — Metode Penelitian
Menggambarkan pipeline analisis sentimen dari YouTube scraping hingga klasifikasi
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Konfigurasi warna ──
C_START    = "#2C3E50"   # Start / End
C_PROCESS  = "#2980B9"   # Proses umum
C_INPUT    = "#8E44AD"   # Input data
C_DECISION = "#E67E22"   # Decision diamond
C_ML       = "#16A085"   # ML Model
C_FALLBACK = "#D35400"   # Fallback / rule-based
C_OUTPUT   = "#27AE60"   # Output hasil
C_ACTIVE   = "#C0392B"   # Active learning
C_STORAGE  = "#7F8C8D"   # File storage
C_PREPRO   = "#3498DB"   # Preprocessing
C_LABEL    = "#ECF0F1"   # Text color

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis("off")

title = "Flowchart Alur Filtering Komentar\nPipeline Analisis Sentimen — Metode Penelitian"
ax.text(7, 21.5, title, ha="center", va="center", fontsize=15, fontweight="bold",
        color="#2C3E50", linespacing=1.5)

# ── Helper: draw box ──
def draw_box(ax, x, y, w, h, text, color=C_PROCESS, text_color="white", fontsize=9,
             sub_text="", sub_color="#bdc3c7", boxstyle="round,pad=0.3"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=boxstyle,
                          facecolor=color, edgecolor=color, linewidth=2,
                          zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.08 if sub_text else 0), text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
    if sub_text:
        ax.text(x, y - 0.28, sub_text, ha="center", va="center",
                fontsize=7.5, color=sub_color, style="italic", zorder=4)
    return box

def draw_diamond(ax, x, y, s, text, color=C_DECISION, text_color="white", fontsize=9):
    diamond = mpatches.Polygon(
        [[x, y + s/2], [x + s, y], [x, y - s/2], [x - s, y]],
        closed=True, facecolor=color, edgecolor=color, linewidth=2, zorder=3
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    return diamond

def draw_arrow(ax, x1, y1, x2, y2, color="#7F8C8D", lw=1.8, style="arc3,rad=0",
               label="", label_pos="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle=style),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, ha="left", va="center",
                fontsize=7.5, color=color, fontweight="bold", style="italic")

def draw_label(ax, x, y, text, color="#34495E", fontsize=8, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, style="italic", fontweight="bold")

# ══════════════════════════════════════════
#  BAGIAN 1: INPUT & SCRAPING
# ══════════════════════════════════════════
bx, bw = 3, 3.2
draw_box(ax, 7, 20, 4, 0.7, "START", C_START, fontsize=10,
         sub_text="Mulai proses scanning")

draw_arrow(ax, 7, 19.65, 7, 18.9)

draw_box(ax, 7, 18.3, 4.5, 0.9, "Input URL YouTube", C_INPUT, fontsize=10,
         sub_text="User memasukkan link video YouTube")

draw_arrow(ax, 7, 17.85, 7, 17.1)

draw_box(ax, 7, 16.5, 4.5, 0.9, "Scrape Komentar YouTube", C_PROCESS, fontsize=10,
         sub_text="youtube-comment-downloader, max limit komentar")

draw_arrow(ax, 7, 16.05, 7, 15.3)

draw_box(ax, 7, 14.7, 4.5, 0.9, "Simpan CSV Mentah", C_STORAGE, fontsize=10,
         sub_text="./hasil_scanning/[nama]_raw.csv", sub_color="#ecf0f1")

draw_arrow(ax, 7, 14.25, 7, 13.55)

# ══════════════════════════════════════════
#  BAGIAN 2: PREPROCESSING
# ══════════════════════════════════════════
draw_box(ax, 7, 12.9, 5.5, 1.0, "Pipeline Preprocessing Teks", C_PREPRO, fontsize=10,
         sub_text="1. Clean (URL, mention, emoji→token)  2. Normalize Slang (140+ kata)")
draw_box(ax, 7, 11.9, 5.5, 0.7, "3. Remove Stopwords  4. Stemming (PySastrawi)", C_PREPRO, fontsize=9,
         sub_text="Hasil: teks bersih → kolom 'text_clean'")

draw_arrow(ax, 7, 11.55, 7, 10.85)

# ══════════════════════════════════════════
#  BAGIAN 3: MODEL LOAD/TRAIN
# ══════════════════════════════════════════
draw_diamond(ax, 7, 10.1, 1.1, "Model\ntersedia?", C_DECISION, fontsize=9)

draw_arrow(ax, 6.45, 10.1, 4.5, 10.1, label="Tidak", label_pos="center")
draw_arrow(ax, 7.55, 10.1, 9.5, 10.1, label="Ya", label_pos="center")

draw_box(ax, 3.2, 10.1, 2.2, 0.7, "Train dari Dataset", C_ML, fontsize=9,
         sub_text="./data_raw/semua_topik.csv")

draw_box(ax, 9.5, 10.1, 2, 0.7, "Load Model\nyang Ada", C_ML, fontsize=9,
         sub_text="./hasil/model_nb.pkl")

draw_arrow(ax, 3.2, 9.75, 5.85, 9.1, label="", label_pos="center", style="arc3,rad=-0.3")
draw_arrow(ax, 9.5, 9.75, 8.15, 9.1, label="", label_pos="center", style="arc3,rad=0.3")

# merge point — model siap
draw_box(ax, 7, 8.5, 3.5, 0.7, "Model ML Siap", C_ML, fontsize=10,
         sub_text="SGD Logistic Regression + TF-IDF char n-grams (2-5)")

draw_arrow(ax, 7, 8.15, 7, 7.45)

# ══════════════════════════════════════════
#  BAGIAN 4: PREDIKSI & FALLBACK
# ══════════════════════════════════════════
draw_diamond(ax, 7, 6.75, 1.0, "Conf ≥ thr\n& Known\nRatio ≥ thr?", C_DECISION, fontsize=8,
             )

draw_arrow(ax, 6.5, 6.75, 4.2, 6.75, label="Tidak", label_pos="center")
draw_arrow(ax, 7.5, 6.75, 9.8, 6.75, label="Ya", label_pos="center")

draw_box(ax, 4.2, 5.8, 2.8, 0.8, "Fallback:\nSmartSentimentLabeler", C_FALLBACK, fontsize=9,
         sub_text="Rule-based: sarkasme, negasi,\nkontradiksi, bobot posisi")

draw_box(ax, 9.8, 5.8, 2.8, 0.8, "Prediksi Model ML", C_ML, fontsize=10,
         sub_text="SGD Logistic Regression\nconfidence score")

draw_arrow(ax, 4.2, 5.4, 5.85, 4.8, label="", label_pos="center", style="arc3,rad=-0.3")
draw_arrow(ax, 9.8, 5.4, 8.15, 4.8, label="", label_pos="center", style="arc3,rad=0.3")

# merge point
draw_box(ax, 7, 4.2, 3, 0.7, "Hasil Prediksi Sentimen", C_OUTPUT, fontsize=10,
         sub_text="Positif / Negatif (2 kelas)")

draw_arrow(ax, 7, 3.85, 7, 3.15)

# ══════════════════════════════════════════
#  BAGIAN 5: ACTIVE LEARNING
# ══════════════════════════════════════════
draw_diamond(ax, 7, 2.45, 0.9, "Low-conf\n& online\nmode?", C_DECISION, fontsize=8)

draw_arrow(ax, 6.55, 2.45, 4.5, 2.45, label="Ya", label_pos="center")
draw_arrow(ax, 7.45, 2.45, 9.5, 2.45, label="Tidak", label_pos="center")

draw_box(ax, 4.5, 1.45, 3.2, 0.9, "Active Learning:\nUser Koreksi Prediksi", C_ACTIVE, fontsize=9,
         sub_text="Model update via partial_fit()\nFeedback → feedback_history.csv")

draw_arrow(ax, 4.5, 1.0, 4.5, 1.5, color="#C0392B", lw=1.5, style="arc3,rad=1.5",
           label="Loop until all corrected", label_pos="center")

draw_box(ax, 9.5, 1.45, 2.5, 0.7, "Simpan CSV\nHasil Filter", C_OUTPUT, fontsize=9,
         sub_text="./hasil_scanning/[nama].csv")

draw_arrow(ax, 4.5, 0.55, 7, 0.55, style="arc3,rad=0")
draw_arrow(ax, 9.5, 1.1, 8.15, 0.55, label="", style="arc3,rad=0.3")

# ══════════════════════════════════════════
#  BAGIAN 6: VISUALISASI & END
# ══════════════════════════════════════════
draw_box(ax, 7, 0.55, 3, 0.7, "Visualisasi", C_PROCESS, fontsize=9,
         sub_text="Pie chart + Word Cloud")

draw_arrow(ax, 7, 0.2, 7, -0.5)

draw_box(ax, 7, -1.1, 3, 0.7, "END", C_START, fontsize=10,
         sub_text="Ringkasan ditampilkan ke terminal")

# ── Legend ──
legend_x, legend_y = 11.3, 20
legend_items = [
    ("    Proses / Aksi", C_PROCESS),
    ("    Input Data", C_INPUT),
    ("    Keputusan", C_DECISION),
    ("    Model ML", C_ML),
    ("    Fallback Rule-based", C_FALLBACK),
    ("    Hasil / Output", C_OUTPUT),
    ("    Active Learning", C_ACTIVE),
    ("    File Storage", C_STORAGE),
    ("    Preprocessing", C_PREPRO),
]
ax.text(legend_x, legend_y + 0.3, "LEGEND:", fontsize=10, fontweight="bold", color="#2C3E50")
for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - 0.45 - i * 0.45
    patch = mpatches.Rectangle((legend_x, y_pos - 0.12), 0.35, 0.24,
                                facecolor=color, edgecolor=color, linewidth=1)
    ax.add_patch(patch)
    ax.text(legend_x + 0.45, y_pos, label, fontsize=8, va="center", color="#2C3E50")

plt.tight_layout()
plt.savefig("./hasil_scanning/flowchart_filtering_komentar.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

print("[OK] Flowchart berhasil dibuat -> ./hasil_scanning/flowchart_filtering_komentar.png")

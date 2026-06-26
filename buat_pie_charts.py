"""
Generate Pie Charts untuk laporan akademik
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data Kebijakan Prabowo ──
labels = ["Negatif", "Positif"]
colors = ["#e74c3c", "#2ecc71"]
sizes_prabowo = [206, 41]
sizes_mbg = [192, 42]

def buat_pie(sizes, title, filename, topik_label):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    wedges, texts, autotexts = ax.pie(  # type: ignore
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5, "antialiased": True},
        explode=(0.03, 0.03),
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    
    total = sum(sizes)
    ax.text(0, -1.4, f"Total: {total} komentar", ha="center", va="center",
            fontsize=11, color="#7F8C8D", style="italic")
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color="#2C3E50")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Pie chart saved -> {filename}")

buat_pie(
    sizes_prabowo,
    "Distribusi Sentimen - Kebijakan Prabowo (YouTube)",
    "./hasil_scanning/plots/pie_prabowo.png",
    "Kebijakan Prabowo"
)

buat_pie(
    sizes_mbg,
    "Distribusi Sentimen - MBG (YouTube)",
    "./hasil_scanning/plots/pie_mbg.png",
    "MBG"
)

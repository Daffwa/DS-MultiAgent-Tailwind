"""Script untuk membuat dan menyimpan Sequence Workflow Diagram ke format gambar PNG berkualitas tinggi (300 DPI).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "workflow_sequence.png")

def create_sequence_image():
    # Setup Canvas
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
    fig.patch.set_facecolor('#0b1120')
    ax.set_facecolor('#0b1120')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Header Title
    ax.text(8, 9.5, "DIAGRAM ALUR KERJA SIKLUS (SEQUENCE WORKFLOW)", 
            fontsize=16, fontweight='bold', color='#38bdf8', ha='center', va='center')
    ax.text(8, 9.1, "Interaksi Kronologis Antar-Node dalam Menyelesaikan Tugas Data Science", 
            fontsize=11, color='#94a3b8', ha='center', va='center')

    # Actors / Columns (x-positions)
    actors = [
        ("User", 1.8, '#38bdf8'),
        ("Streamlit UI", 4.3, '#6366f1'),
        ("Supervisor", 7.0, '#eab308'),
        ("Doc & Vision", 9.8, '#38bdf8'),
        ("Data Coder", 12.5, '#4ade80'),
        ("Final Writer", 14.8, '#ec4899'),
    ]

    # Draw Actor Headers and Lifelines
    for name, x, col in actors:
        box = patches.FancyBboxPatch((x - 1.0, 8.2), 2.0, 0.6,
                                    boxstyle="round,pad=0.15",
                                    facecolor='#1e293b', edgecolor=col, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 8.5, name, fontsize=10, fontweight='bold', color='#ffffff', ha='center', va='center')
        
        # Lifeline (dashed vertical line)
        ax.plot([x, x], [8.2, 0.8], color='#334155', linestyle='--', linewidth=1.2, zorder=1)

    # Function for sequence messages
    def draw_msg(step_num, x1, x2, y, text, color='#38bdf8', is_dashed=False):
        style = '--' if is_dashed else '-'
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=6,
                                    linestyle=style, shrink=0.03), zorder=2)
        # Message text box
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.18, f"{step_num}. {text}", fontsize=8.5, color='#f1f5f9',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.9), zorder=3)

    # Sequence Steps
    draw_msg(1, 1.8, 4.3, 7.5, "Upload Soal.docx + Data.csv + Query", '#38bdf8')
    draw_msg(2, 4.3, 7.0, 6.8, "StateGraph.stream(initial_state)", '#6366f1')
    draw_msg(3, 7.0, 9.8, 6.1, "Rute: 'doc_reader' (Ekstrak isi soal & gambar)", '#eab308')
    draw_msg(4, 9.8, 7.0, 5.4, "Teks soal & ringkasan diekstrak -> Update State", '#38bdf8', is_dashed=True)
    draw_msg(5, 7.0, 12.5, 4.7, "Rute: 'data_analyst' (Kalkulasi Python & Plot)", '#eab308')
    draw_msg(6, 12.5, 7.0, 4.0, "Hasil komputasi statistik & plot tersimpan", '#4ade80', is_dashed=True)
    draw_msg(7, 7.0, 14.8, 3.3, "Rute: 'FINISH' (Semua tugas selesai)", '#eab308')
    draw_msg(8, 14.8, 4.3, 2.6, "Kompilasi Laporan Terstruktur (Markdown)", '#ec4899')
    draw_msg(9, 4.3, 1.8, 1.9, "Render Jawaban Akademis, Tabel & Visualisasi", '#38bdf8')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"SUCCESS: Sequence diagram disimpan ke: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_sequence_image()

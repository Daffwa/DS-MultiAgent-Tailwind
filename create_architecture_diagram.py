"""Script untuk membuat dan menyimpan Diagram Arsitektur Multi-Agent ke format gambar PNG berkualitas tinggi (300 DPI).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "arsitektur_multi_agent.png")

def create_architecture_image():
    # Setup Canvas (16:10 Ratio, High Resolution)
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=300)
    fig.patch.set_facecolor('#0b1120')
    ax.set_facecolor('#0b1120')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Header Title
    ax.text(8, 10.4, "ARSITEKTUR HIERARCHICAL SUPERVISOR MULTI-AGENT SYSTEM", 
            fontsize=17, fontweight='bold', color='#38bdf8', ha='center', va='center', family='sans-serif')
    ax.text(8, 9.95, "Data Science Problem Solver dengan LangGraph, Gemini Vision & Streamlit", 
            fontsize=11.5, color='#94a3b8', ha='center', va='center', family='sans-serif')

    # Helper function for drawing rounded boxes
    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle="", radius=0.25):
        box = patches.FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad={radius}",
                                    facecolor=bg_color,
                                    edgecolor=border_color,
                                    linewidth=1.8)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.35, title, fontsize=11, fontweight='bold', 
                    color='#ffffff', ha='center', va='center', family='sans-serif')
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.18, subtitle, fontsize=9, 
                    color='#cbd5e1', ha='center', va='center', family='sans-serif')

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, color='#38bdf8', label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=7, headlength=7, shrink=0.05))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.12, label, fontsize=8.5, color='#fde047', 
                    ha='center', va='center', fontweight='bold', family='sans-serif',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.9))

    # --- LAYER 1: UI & PERSISTENCE (Top) ---
    draw_card(1.0, 7.8, 4.0, 1.4, '#1e293b', '#38bdf8', "[USER] Mahasiswa / User", "Upload PDF/Word/CSV/Excel\n& Lampiran Gambar/Video")
    draw_card(6.0, 7.8, 4.0, 1.4, '#1e293b', '#6366f1', "[UI] Streamlit Web Dashboard", "app.py | Live Activity Tracker\nPop-over Media Attachment")
    draw_card(11.0, 7.8, 4.0, 1.4, '#1e293b', '#a855f7', "[STORAGE] Persistence Manager", "utils/persistence.py\n.config.json & chat_history.json")

    draw_arrow(5.2, 8.5, 5.8, 8.5, '#38bdf8', "Input")
    draw_arrow(10.2, 8.5, 10.8, 8.5, '#a855f7', "Auto-Save")

    # --- LAYER 2: ORCHESTRATION / SUPERVISOR (Middle Upper) ---
    draw_card(4.5, 5.5, 7.0, 1.4, '#1e1b4b', '#eab308', "[SUPERVISOR] Supervisor Agent Node", 
              "agent/supervisor.py | Routing Cerdas (Gemini/Gemma)\nMenentukan: 'doc_reader' | 'data_analyst' | 'FINISH'")

    draw_arrow(8.0, 7.6, 8.0, 7.1, '#38bdf8', "StateGraph.stream()")

    # --- LAYER 3: WORKER AGENTS (Middle Lower) ---
    draw_card(1.2, 3.2, 6.2, 1.5, '#0f2b46', '#38bdf8', "[WORKER 1] Doc & Vision Reader Node", 
              "agent/worker_reader.py\nEkstraksi Teks PDF/Word & Analisis Visual\nMultimodal Base64 (Gemini Vision)")

    draw_card(8.6, 3.2, 6.2, 1.5, '#064e3b', '#4ade80', "[WORKER 2] Data Science & Coder Node", 
              "agent/worker_coder.py\nInspeksi Tabel CSV/Excel + Generator Kode\nEksekusi Python Sandbox & Plotting")

    # Routing Arrows from Supervisor to Workers
    draw_arrow(6.5, 5.3, 4.3, 4.9, '#38bdf8', "Rute: 'doc_reader'")
    draw_arrow(9.5, 5.3, 11.7, 4.9, '#4ade80', "Rute: 'data_analyst'")

    # Loop Back Arrows to Supervisor
    draw_arrow(4.3, 4.9, 6.3, 5.3, '#94a3b8', "Update State")
    draw_arrow(11.7, 4.9, 9.7, 5.3, '#94a3b8', "Update State")

    # --- LAYER 4: TOOLS & EXECUTION LAYER (Bottom) ---
    draw_card(1.2, 0.8, 6.2, 1.6, '#1e293b', '#475569', "[TOOLS] Document & Media Extractors", 
              "tools/doc_tools.py (pypdf, docx)\ntools/media_tools.py (Base64 Encode)\nSmart File Path Resolver")

    draw_card(8.6, 0.8, 6.2, 1.6, '#1e293b', '#475569', "[TOOLS] Tabular Data & Python Sandbox", 
              "tools/data_tools.py (Pandas, Numpy)\nexecute_python_code() Sandbox\ngenerated_plots/ (Plot PNG)")

    draw_arrow(4.3, 3.0, 4.3, 2.6, '#38bdf8', "Tool Calling")
    draw_arrow(11.7, 3.0, 11.7, 2.6, '#4ade80', "Exec Python")

    # --- LAYER 5: FINAL OUTPUT (Right Side Bottom) ---
    draw_card(11.0, 5.5, 4.0, 1.4, '#311042', '#ec4899', "[FINAL] Lead Assistant / Writer", 
              "utils/formatters.py\nLaporan Terstruktur + Plot Image\nJawaban Akademis Lengkap")

    draw_arrow(11.7, 6.2, 11.7, 7.6, '#ec4899', "Rute: 'FINISH'")

    # Footer Notes
    ax.text(8, 0.3, "Hierarchical Supervisor Multi-Agent Architecture | Google DeepMind & LangGraph Framework", 
            fontsize=9, color='#64748b', ha='center', va='center', family='sans-serif')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"SUCCESS: Gambar diagram arsitektur disimpan ke: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_architecture_image()

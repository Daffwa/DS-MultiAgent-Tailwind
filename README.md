# 🧠 Data Science Multi-Agent Solver (Adaptive 3-Tier Multi-Agent Stack)
### *Autonomous Adaptive Hybrid Multi-Agent Platform for Data Engineering, Inferential Statistics, Machine Learning, and Academic Report Synthesis*

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Adaptive_Hybrid_MAS-orange.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4_Modern_Dark-38bdf8.svg)
![KaTeX](https://img.shields.io/badge/KaTeX-Math_Equations-38bdf8.svg)

---

## 🏛️ Arsitektur Sistem: 3-Tier Adaptive Hybrid Orchestrator

Sistem ini menerapkan pola **Adaptive Hybrid Multi-Agent Architecture (Generasi 3 SOTA 2024–2026)** yang mengeliminasi kelemahan rantai linier (*Waterfall Bottleneck*) dengan membagi alur eksekusi secara dinamis ke dalam **3 Tingkatan Beban Kerja (*3-Tier Execution Engine*)**:

```
                               ┌──────────────────────────────────────────────┐
                               │             PERTANYAAN PENGGUNA              │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │    👑 SUPERVISOR AGENT        │
                                       │    (Adaptive Intent Triage)  │
                                       └──────────────┬───────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
   [ TIER 1: FAST-PATH ]                    [ TIER 2: SINGLE-DISPATCH ]                  [ TIER 3: MULTI-STAGE ]
   Sapaan / Teori Kasual                    Tugas 1 Domain (EDA / ML / OCR)              Tugas Lengkap (Baca + Olah + ML)
   Alur: Langsung ke Penjawab               Alur: Router ➔ 1 Agen Spesialis              Alur: Forward DAG Coordinated Plan
   ⏱️ Waktu: 1 - 2 detik                    ⏱️ Waktu: 5 - 10 detik                       ⏱️ Waktu: 12 - 20 detik
         │                                            │                                            │
         └────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   🏁 FINAL SYNTHESIZER       │
                                       │   (Laporan & Ekspor Word)    │
                                       └──────────────────────────────┘
```

---

## 👥 Rincian Divisi 5 Agen Terkoordinasi

| No | Identitas Agen | File Sumber | Spesialisasi & Tanggung Jawab |
| :---: | :--- | :--- | :--- |
| **1** | 👑 **Supervisor Agent** | [`agent/supervisor.py`](file:///D:/Capstone/multi%20agent%20hierarchy%20system%20+%20tailwind%20CSS/agent/supervisor.py) | **Project Manager & Traffic Controller:** Menganalisis niat (*intent*) pengguna secara instan (<1ms) dan menentukan alur eksekusi (Tier 1, Tier 2, atau Tier 3). |
| **2** | 📄 **Doc & Vision Reader** | [`agent/worker_reader.py`](file:///D:/Capstone/multi%20agent%20hierarchy%20system%20+%20tailwind%20CSS/agent/worker_reader.py) | **Spesialis Dokumen & Visual:** Mengekstrak teks/soal dari berkas **PDF & Word (.docx)**, serta membaca grafik/diagram dari gambar menggunakan Multimodal Vision AI. |
| **3** | 📊 **Data & Stats Engineer** | [`agent/worker_coder.py`](file:///D:/Capstone/multi%20agent%20hierarchy%20system%20+%20tailwind%20CSS/agent/worker_coder.py) | **Data Cleaning & Statistik Inferensial:** Menangani *missing values*, audit *outlier* (IQR), kalkulasi Teori Antrian ($M/M/1, M/M/s$), serta pengujian hipotesis formal ($t$-test, ANOVA, Shapiro-Wilk, $p < 0.05$). |
| **4** | 🤖 **Machine Learning Specialist** | [`agent/worker_ml.py`](file:///D:/Capstone/multi%20agent%20hierarchy%20system%20+%20tailwind%20CSS/agent/worker_ml.py) | **Pemodelan Prediktif & Auto-ML:** *Supervised* (Random Forest, Logistic, SVC, Regresi Linear/Ridge), *Unsupervised* (K-Means, PCA 2D, Isolation Forest Anomaly), 5-Fold Cross Validation, dan Confusion Matrix. |
| **5** | 🏁 **Final Synthesizer** | [`agent/graph.py`](file:///D:/Capstone/multi%20agent%20hierarchy%20system%20+%20tailwind%20CSS/agent/graph.py) | **Academic Writer:** Merangkum seluruh temuan teknis menjadi laporan terstruktur rapi dengan notasi LaTeX KaTeX serta ekspor Word (`.docx`). |

---

## 🖥️ 5 Tab Workspace Sinkron di Frontend

1. 📊 **Tab 1: Live Dataset Explorer & Data Health:** Menampilkan tabel baris data secara real-time, pencarian instan (*live search*), persentase *missing values*, dan banner **Rekomendasi Auto-ML**.
2. 📈 **Tab 2: Interactive Charts (Multi-Plot Gallery):** Menampilkan seluruh grafik visual resolusi tinggi yang dibuat oleh agen dengan bilah tab pemilih (`Plot 1`, `Plot 2`, dst.) dan tombol unduh.
3. 🤖 **Tab 3: Machine Learning Arena:** Menampilkan kartu metrik performa model (*Accuracy, F1-Score, Split Ratio*), **Heatmap Confusion Matrix**, **PCA Projection**, dan **Scatter Cluster**.
4. 💻 **Tab 4: Multi-Agent Interactive Sandbox:**
   * **Sub-Tab Data & Stats:** Editor kode Python yang **bisa diedit** + tombol `▶️ Jalankan Kode (Run)` + terminal eksekusi live.
   * **Sub-Tab Doc Reader:** Penguji ekstraksi teks dan rumus dari berkas PDF/Word.
   * **Sub-Tab Supervisor:** Simulator pengujian logika perutean (*Routing Decision Tester*).
5. 📑 **Tab 5: Academic Report Preview:** Pratinjau dokumen laporan tugas formal yang siap diekspor 1-klik ke Microsoft Word (.docx).

---

## 🛡️ Fitur Unggulan Sistem

* ⚡ **_Zero-Delay Conversational Bypass_:** Percakapan ramah atau pertanyaan teori dijawab seketika ($\approx 1\text{ detik}$) tanpa memanggil agen pekerja berat.
* 🔄 **_Self-Healing Python Execution_:** Jika script Python mengalami salah nama kolom (*KeyError*) atau missing import, sandbox otomatis mengoreksi kodenya sendiri dan menjalankan ulang eksekusi hingga sukses tanpa *crash*.
* ⏹️ **Tombol STOP & Modal Konfirmasi Reset:** Pengguna dapat membatalkan request seketika via `AbortController` atau mereset sesi secara aman melalui dialog modal konfirmasi.
* ⏱️ **Live Stopwatch Timer:** Mencatat durasi eksekusi tiap langkah agen secara presisi milidetik.
* 📐 **KaTeX Mathematical Equations:** Notasi formula LaTeX (seperti $\lambda$, $\mu$, $\rho$, $L_q$, $W_q$, $\frac{a}{b}$) dirender otomatis menjadi persamaan matematika standar publikasi.

---

## 🚀 Cara Menjalankan Aplikasi

### Opsi 1: Menjalankan via File Batch Web (Rekomendasi Dev)
1. Masuk ke folder proyek: `D:\Capstone\multi agent hierarchy system + tailwind CSS\`
2. Klik dua kali (*double-click*) pada file:
   👉 **`Jalankan_Web_Tailwind.bat`**
3. Browser otomatis terbuka di: **`http://localhost:8000`**.

### Opsi 2: Menjalankan Standalone Executable (.exe)
1. Buka folder distribusi:
   `dist\DS-MultiAgent-Tailwind\`
2. Jalankan berkas:
   👉 **`DS-MultiAgent-Tailwind.exe`**

"""Worker Node: Data Engineering & Statistical Analysis Specialist Agent.
Bertanggung jawab atas pembersihan dataset (missing value & outlier IQR), kalkulasi parameter teori antrian,
analisis statistik deskriptif, dan pengujian hipotesis inferensial formal (t-test, ANOVA, Shapiro-Wilk, Pearson r).
"""

import os
import time
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools.data_tools import inspect_tabular_data, execute_python_code, resolve_file
from utils.formatters import clean_and_format_output, extract_python_code
from agent.state import AgentState

DATA_ANALYST_SYSTEM_PROMPT = """Anda adalah 'Data & Statistical Engineer Agent' spesialis dalam data cleaning, rekayasa tabel, kalkulasi teori antrian, pengujian hipotesis statistik, dan Exploratory Data Analysis (EDA).

ATURAN UTAMA EKSEKUSI:
1. Jika pengguna meminta untuk membersihkan (cleaning), memodifikasi, menganalisis statistik deskriptif, menghitung rumus/teori antrian, melakukan uji hipotesis (t-test, ANOVA, uji normalitas), atau membuat grafik tren, Anda WAJIB MENULIS KODE PYTHON LENGKAP di dalam:
```python
# Kode Python Anda di sini
```
2. PANDUAN EKSPOR BERKAS & GRAFIK:
   - Simpan dataset bersih ke: `generated_files/nama_file_cleaned.csv` atau `.xlsx`
   - Simpan grafik distribusi/tren ke: `generated_plots/nama_plot.png` (dpi=200, bbox_inches='tight')
3. PANDUAN STATISTIK INFERENSIAL:
   - Gunakan `scipy.stats` untuk uji hipotesis (`stats.shapiro`, `stats.ttest_ind`, `stats.f_oneway`, `stats.pearsonr`).
   - Selalu sertakan kesimpulan nilai p (p-value): Jika p < 0.05 maka tolak H0 (terdapat perbedaan/pengaruh signifikan).
4. SINKRONISASI REKOMENDASI MACHINE LEARNING (EDA ➔ ML AGENT HANDOVER):
   - Setelah melakukan pembersihan data atau EDA, Anda WAJIB menyertakan analisis kesesuaian Machine Learning dalam format:
   ### 🤖 Rekomendasi Machine Learning (Berdasarkan Hasil EDA):
   - **Tipe Tugas Terbaik:** [Supervised Klasifikasi / Supervised Regresi / Unsupervised Clustering / PCA]
   - **Algoritma yang Direkomendasikan:** [e.g. Random Forest, Logistic Regression, K-Means]
   - **Rasional Pemilihan:** [Jelaskan alasan matematis/karakteristik kolomnya]
5. ATURAN ANTI-LOOPING: Dilarang keras membuat garis pembatas komentar berulang seperti '/////' atau '====='. Langsung tulis kode Python di dalam blok kode.
"""

def create_data_analyst_node(llm: Any):
    """Factory untuk membuat node Data & Statistical Engineer yang tangguh dan modular."""

    def data_analyst_node(state: AgentState) -> dict:
        uploaded_files = state.get("uploaded_files", {})
        file_list_str = "\n".join([f"- {fname}: {fpath}" for fname, fpath in uploaded_files.items()]) or "Tidak ada file terdaftar."
        
        dataset_context = ""
        tabular_files = [f for f in uploaded_files.keys() if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
        for fname in tabular_files:
            try:
                summary = inspect_tabular_data.invoke({"file_path": resolve_file(uploaded_files[fname])})
                dataset_context += f"\n\nRingkasan Dataset [{fname}]:\n{summary}\n"
            except Exception:
                pass

        system_msg = SystemMessage(
            content=f"{DATA_ANALYST_SYSTEM_PROMPT}\n\nFile yang tersedia:\n{file_list_str}{dataset_context}"
        )
        
        user_query_str = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    user_query_str = m.content
                elif isinstance(m.content, list):
                    for block in m.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_query_str = block.get("text", "")

        prompt_context = f"Instruksi Pengguna: {user_query_str}\n\nSilakan rancang dan jalankan script Python untuk menyelesaikan tugas pembersihan data, statistik, atau visualisasi ini."
        
        response = llm.invoke([system_msg, HumanMessage(content=prompt_context)])
        response_text = str(response.content)
        
        extracted_code = extract_python_code(response_text)
        
        # Fallback 1: Pembersihan Data jika model lupa blok kode
        if not extracted_code and any(kw in user_query_str.lower() for kw in ["clean", "bersih", "hapus missing", "edit", "manipulasi", "olah"]):
            if tabular_files:
                target_f = tabular_files[0]
                target_path = resolve_file(uploaded_files[target_f])
                base_no_ext = os.path.splitext(target_f)[0]
                
                if target_f.lower().endswith('.csv'):
                    extracted_code = f"""import pandas as pd
df = pd.read_csv(r'{target_path}')
df_clean = df.dropna().drop_duplicates()
df_clean.to_csv(r'generated_files/{base_no_ext}_cleaned.csv', index=False)
print("Pembersihan dataset otomatis selesai. File tersimpan di generated_files/{base_no_ext}_cleaned.csv")
"""
                else:
                    extracted_code = f"""import pandas as pd
df = pd.read_excel(r'{target_path}')
df_clean = df.dropna().drop_duplicates()
df_clean.to_excel(r'generated_files/{base_no_ext}_cleaned.xlsx', index=False)
print("Pembersihan dataset otomatis selesai. File tersimpan di generated_files/{base_no_ext}_cleaned.xlsx")
"""

        # Fallback 2: Visualisasi Tren / Distribusi / Heatmap Korelasi
        if not extracted_code and any(kw in user_query_str.lower() for kw in ["grafik", "plot", "visual", "tren", "trend", "antrian", "diagram", "chart", "distribusi", "korelasi", "heatmap"]):
            if tabular_files:
                target_f = tabular_files[0]
                target_path = resolve_file(uploaded_files[target_f])
                read_func = f"pd.read_csv(r'{target_path}')" if target_f.lower().endswith('.csv') else f"pd.read_excel(r'{target_path}')"
                extracted_code = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = {read_func}
numeric_df = df.select_dtypes(include=[np.number])

# 1. Plot Matriks Korelasi jika ada >= 2 kolom numerik
if len(numeric_df.columns) >= 2:
    plt.figure(figsize=(7, 5))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Matriks Korelasi Pearson (EDA Correlation Heatmap)', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig('generated_plots/correlation_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Heatmap Korelasi berhasil disimpan ke generated_plots/correlation_matrix.png")

# 2. Plot Tren Deret Data
plt.figure(figsize=(10, 4.5))
queue_cols = [c for c in df.columns if any(k in c.lower() for k in ['antrian', 'pelayanan', 'kedatangan', 'waktu', 'lama'])]

if queue_cols:
    target_col = queue_cols[0]
    plt.plot(df.index, df[target_col], marker='o', color='#06b6d4', linewidth=2, label=target_col)
    plt.title(f'Grafik Tren {{target_col}}', fontsize=13, color='#38bdf8', pad=10)
elif len(numeric_df.columns) > 0:
    target_col = numeric_df.columns[0]
    plt.plot(df.index, df[target_col], marker='o', color='#10b981', linewidth=2, label=target_col)
    plt.title(f'Grafik Tren {{target_col}}', fontsize=13, color='#10b981', pad=10)
else:
    plt.plot([1, 2, 3], [1, 2, 3], color='#06b6d4')
    plt.title('Grafik Analisis Data', color='#38bdf8')

plt.xlabel('Indeks Data')
plt.ylabel('Nilai')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('generated_plots/grafik_tren_analisis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Grafik tren visualisasi berhasil disimpan ke generated_plots/grafik_tren_analisis.png")
"""

        execution_output = ""
        if extracted_code:
            try:
                execution_output = execute_python_code.invoke({"code": extracted_code})
            except Exception as e:
                execution_output = f"Gagal mengeksekusi kode Python: {str(e)}"

        combined_response = response_text
        if extracted_code and not extract_python_code(response_text):
            combined_response = f"Berikut adalah script Python yang dieksekusi:\n```python\n{extracted_code}\n```"

        if execution_output:
            combined_response += f"\n\n**Hasil Eksekusi Nyata (Output Terminal):**\n```text\n{execution_output}\n```"

        cleaned_content = clean_and_format_output(combined_response)
        
        new_messages = [
            AIMessage(content=cleaned_content, name="data_analyst")
        ]

        logs = state.get("activity_logs", [])
        action_desc = "Menyelesaikan visualisasi data & kalkulasi statistik" if "grafik" in user_query_str.lower() else "Menyelesaikan audit dataset & statistik"
        logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": "📊 Data & Stats Engineer",
            "message": f"{action_desc}."
        })

        return {
            "messages": new_messages,
            "activity_logs": logs
        }

    return data_analyst_node

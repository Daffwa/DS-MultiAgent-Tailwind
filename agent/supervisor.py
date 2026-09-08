"""Supervisor Node: Project Manager & 3-Tier Adaptive Triage Router.
Mengimplementasikan Adaptive Hybrid Multi-Agent Orchestration:
1. Tier 1: Fast-Path Bypass (Sapaan, Pertanyaan Teori Ringan -> 1-2s)
2. Tier 2: Targeted Single-Dispatch (Tugas 1 Domain Spesifik -> 5-10s)
3. Tier 3: Coordinated Multi-Stage Pipeline (Tugas Komprehensif Terencana -> 12-20s)
"""

import time
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import AgentState

SUPERVISOR_SYSTEM_PROMPT = """Anda adalah 'Supervisor Agent' (Project Manager) dalam Tim Data Science Multi-Agent.

Tugas Anda adalah menganalisis niat pengguna dan konteks berkas, kemudian memilih agen spesialis yang tepat:
1. 'doc_reader'   : Membaca dokumen soal (PDF/Word) atau menganalisis gambar/foto soal.
2. 'data_analyst' : Pembersihan data (*missing/outliers*), statistik inferensial, teori antrian, & grafik.
3. 'ml_specialist': Pelatihan model Machine Learning (Klasifikasi, Regresi, Clustering, PCA, Anomali).
4. 'final_writer' : Merangkum laporan akhir atau menjawab percakapan umum.

Format Output:
Hanya tuliskan nama agen tujuan: 'doc_reader', 'data_analyst', 'ml_specialist', atau 'final_writer'.
"""


def detect_user_intents(query: str, has_docs: bool, has_tabular: bool, attached_media: bool) -> Dict[str, bool]:
    """Mendeteksi seluruh niat tugas (multi-intent) dari instruksi pengguna."""
    q = query.lower().strip()
    
    # 1. Niat Sapaan / Percakapan Kasual
    greeting_words = [
        "halo", "hai", "hi", "hello", "hey", "pagi", "siang", "sore", "malam",
        "assalamualaikum", "terima kasih", "makasih", "thanks", "siapa kamu",
        "kamu siapa", "apa kemampuanmu", "bisa bantu apa", "fitur apa", "tes", "test",
        "ping", "oke", "ok", "siap", "mantap", "sip"
    ]
    is_greeting = any(q == w or q.startswith(w + " ") or q.endswith(" " + w) for w in greeting_words)
    
    # 2. Niat Baca Dokumen / OCR
    doc_keywords = ["baca", "dokumen", "pdf", "word", "soal", "ekstrak", "lampiran", "gambar", "foto", "ocr", "teks"]
    wants_doc = attached_media or (has_docs and any(k in q for k in doc_keywords))
    
    # 3. Niat Machine Learning
    ml_keywords = [
        "ml", "machine learning", "prediksi", "predict", "klasifikasi", "classification",
        "regresi", "regression", "cluster", "clustering", "random forest", "model",
        "akurasi", "accuracy", "confusion matrix", "train", "training", "latih", "evaluasi",
        "pca", "isolation forest", "svm", "knn", "decision tree", "ridge", "lasso"
    ]
    wants_ml = has_tabular and any(k in q for k in ml_keywords)
    
    # 4. Niat Pembersihan Data / Statistik / Grafik
    data_keywords = [
        "clean", "bersih", "hapus missing", "outlier", "iqr", "antrian", "statistik",
        "uji hipotesis", "t-test", "anova", "grafik", "plot", "tren", "visual", "eda",
        "korelasi", "heatmap", "deskriptif", "normalitas", "shapiro"
    ]
    wants_data = has_tabular and any(k in q for k in data_keywords)
    
    # Deteksi apakah perintah meminta analisis komprehensif dari awal sampai akhir
    comprehensive_keywords = ["lengkap", "semua", "komprehensif", "menyeluruh", "keseluruhan", "end-to-end"]
    wants_all = has_tabular and any(k in q for k in comprehensive_keywords)
    if wants_all:
        wants_data = True
        wants_ml = True
        if attached_media or (has_docs and any(k in q for k in ["baca", "dokumen", "pdf", "word", "soal", "tugas", "lampiran"])):
            wants_doc = True
        else:
            wants_doc = False
            
    return {
        "greeting": is_greeting,
        "doc": wants_doc,
        "ml": wants_ml,
        "data": wants_data,
        "comprehensive": wants_all
    }


def create_supervisor_node(llm: Any):
    """Factory untuk membuat supervisor router yang deterministik, adaptif, dan anti-hanging."""

    def supervisor_node(state: AgentState) -> dict:
        user_query = ""
        for m in state.get("messages", []):
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    user_query = m.content
                elif isinstance(m.content, list):
                    for block in m.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_query = block.get("text", "")

        uploaded_files = state.get("uploaded_files", {})
        attached_media = state.get("attached_media", [])

        has_docs = any(f.lower().endswith(('.pdf', '.docx', '.doc')) for f in uploaded_files.keys())
        has_tabular = any(f.lower().endswith(('.csv', '.xlsx', '.xls')) for f in uploaded_files.keys())

        # Evaluasi multi-intent
        intents = detect_user_intents(user_query, has_docs, has_tabular, bool(attached_media))
        
        # Inisialisasi rencana eksekusi
        execution_tier = "single_dispatch"
        execution_plan: List[str] = []
        next_node = "final_writer"
        router_reason = "Merangkum respon langsung ke pengguna."

        # =====================================================================
        # TIER 1: FAST-PATH BYPASS (Sapaan & Pertanyaan Teori Kasual)
        # =====================================================================
        if intents["greeting"] and not (intents["doc"] or intents["data"] or intents["ml"]):
            execution_tier = "fast_path"
            execution_plan = []
            next_node = "final_writer"
            router_reason = "Merute cepat ke perangkum untuk respon sapaan/percakapan kasual (Tier 1: Fast-Path)."

        # =====================================================================
        # TIER 3: COORDINATED MULTI-STAGE PIPELINE (Tugas Gabungan Terencana)
        # =====================================================================
        elif sum([intents["doc"], intents["data"], intents["ml"]]) >= 2 or intents["comprehensive"]:
            execution_tier = "multi_stage"
            planned_steps = []
            if intents["doc"]:
                planned_steps.append("doc_reader")
            if intents["data"]:
                planned_steps.append("data_analyst")
            if intents["ml"]:
                planned_steps.append("ml_specialist")
                
            execution_plan = planned_steps
            next_node = planned_steps[0] if planned_steps else "final_writer"
            router_reason = f"Menyusun alur terkoordinasi multi-tahap: {' ➔ '.join(planned_steps)} (Tier 3: Multi-Stage Pipeline)."

        # =====================================================================
        # TIER 2: TARGETED SINGLE-SPECIALIST DISPATCH (Tugas 1 Domain Spesifik)
        # =====================================================================
        elif intents["ml"]:
            execution_tier = "single_dispatch"
            execution_plan = ["ml_specialist"]
            next_node = "ml_specialist"
            router_reason = "Mendelegasikan tugas khusus ke Machine Learning Specialist (Tier 2: Single-Dispatch ML)."

        elif intents["data"]:
            execution_tier = "single_dispatch"
            execution_plan = ["data_analyst"]
            next_node = "data_analyst"
            router_reason = "Mendelegasikan tugas khusus ke Data & Stats Engineer (Tier 2: Single-Dispatch Data)."

        elif intents["doc"]:
            execution_tier = "single_dispatch"
            execution_plan = ["doc_reader"]
            next_node = "doc_reader"
            router_reason = "Mendelegasikan pembacaan berkas ke Doc & Vision Reader (Tier 2: Single-Dispatch Doc)."

        elif has_tabular:
            execution_tier = "single_dispatch"
            execution_plan = ["data_analyst"]
            next_node = "data_analyst"
            router_reason = "Menganalisis dataset aktif melalui Data & Stats Engineer (Tier 2: Default Tabular)."

        else:
            execution_tier = "fast_path"
            execution_plan = []
            next_node = "final_writer"
            router_reason = "Menjawab pertanyaan atau konsultasi teori secara langsung (Tier 1: Direct Synthesizer)."

        logs = state.get("activity_logs", [])
        logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": "👑 Supervisor Agent",
            "message": f"Supervisor ({execution_tier.upper()}): {router_reason}"
        })

        return {
            "next": next_node,
            "execution_tier": execution_tier,
            "execution_plan": execution_plan,
            "activity_logs": logs
        }

    return supervisor_node

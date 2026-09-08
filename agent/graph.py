"""LangGraph Multi-Agent Graph Builder.
Mengompilasi graf alur kerja StateGraph Hierarki Data Science Multi-Agent.
Mendukung 3-Tier Adaptive Hybrid Orchestration:
- Tier 1: Fast-Path Zero-Delay
- Tier 2: Targeted Single-Dispatch
- Tier 3: Coordinated Multi-Stage Forward DAG (Anti-Looping)
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.llm_factory import get_llm
from agent.supervisor import create_supervisor_node
from agent.worker_reader import create_doc_reader_node
from agent.worker_coder import create_data_analyst_node
from agent.worker_ml import create_ml_specialist_node
from utils.formatters import clean_and_format_output

FINAL_WRITER_SYSTEM_PROMPT = """Anda adalah 'Lead Assistant & Academic Report Synthesizer' dalam sistem Data Science Multi-Agent.

TUGAS ANDA:
1. RESPON CEPAT UNTUK PERCAKAPAN RAMAH: Jika pengguna hanya menyapa (seperti 'halo', 'hai', 'selamat pagi') atau bertanya umum tentang kemampuan AI, jawablah secara ramah, ringkas, dan jelas dalam 1-2 paragraf singkat, perkenalkan 4 agen spesialis yang siap membantu.
2. SINTESIS AKADEMIS LENGKAP: Jika ada hasil eksekusi dari tim agen spesialis (Doc Reader, Data Analyst, ML Specialist), rangkum temuan secara mendalam, terstruktur, dan berwibawa dengan format akademik formal.
3. Gunakan notasi LaTeX untuk rumus matematika ($...$ untuk inline, $$...$$ untuk baris terpusat).
4. Jika ada tabel evaluasi metrik (Akurasi, F1, R2, atau Uji Hipotesis), sajikan dalam tabel Markdown yang rapi.
5. ATURAN ANTI-LOOPING: Dilarang keras mengulang kata atau membuat baris karakter berulang seperti '/////' atau '====='.
"""


def create_final_writer_node(llm: Any):
    """Factory untuk membuat node perangkum laporan akhir."""

    def final_writer_node(state: AgentState) -> dict:
        messages_history = state.get("messages", [])
        tier = state.get("execution_tier", "single_dispatch")

        # FAST-PATH: Jika tier adalah fast_path dan tidak ada pekerja yang dijalankan
        worker_ran = any(getattr(m, "name", "") in ["doc_reader", "data_analyst", "ml_specialist"] for m in messages_history)

        last_human_query = ""
        for m in reversed(messages_history):
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    last_human_query = m.content
                elif isinstance(m.content, list):
                    for b in m.content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            last_human_query = b.get("text", "")
                break

        q_clean = last_human_query.lower().strip()
        instant_greetings = [
            "halo", "hai", "hi", "hello", "hey", "selamat pagi", "selamat siang",
            "selamat sore", "selamat malam", "assalamualaikum", "pagi", "siang",
            "malam", "tes", "test", "ping", "bisa bantu apa", "apa kemampuanmu", "siapa kamu"
        ]

        if not worker_ran and (tier == "fast_path" or any(q_clean == w or q_clean.startswith(w + " ") for w in instant_greetings)):
            fast_greeting_msg = (
                "Halo! 👋 Selamat datang di **Data Science Multi-Agent Solver**.\n\n"
                "Saya bersama 4 Agen Spesialis siap membantu Anda:\n"
                "* 👑 **Supervisor Agent**: Mengatur alur kerja kolaboratif secara adaptif.\n"
                "* 📄 **Doc & Vision Reader**: Membaca dokumen tugas/soal (PDF & Word) serta gambar/foto tabel.\n"
                "* 📊 **Data & Stats Engineer**: Pembersihan *missing values/outliers* (IQR), analisis antrian, dan uji hipotesis formal.\n"
                "* 🤖 **Machine Learning Specialist**: Melatih model prediktif (Klasifikasi, Regresi, Clustering, PCA) & Confusion Matrix.\n\n"
                "Silakan unggah dataset atau ketikkan instruksi analisis Anda di bawah!"
            )
            return {
                "messages": [AIMessage(content=fast_greeting_msg, name="final_writer")],
                "final_response": fast_greeting_msg
            }
        
        system_msg = SystemMessage(content=FINAL_WRITER_SYSTEM_PROMPT)
        prompt_content = "Berikut adalah riwayat percakapan dan hasil eksekusi agen:\n\n"
        
        for msg in messages_history:
            sender = getattr(msg, "name", "User" if isinstance(msg, HumanMessage) else "Assistant")
            prompt_content += f"[{sender}]:\n{msg.content}\n\n"
            
        prompt_content += "\nSilakan susun laporan sintesis akhir yang lengkap, padat, berwibawa, dan terstruktur rapi untuk pengguna."

        response = llm.invoke([system_msg, HumanMessage(content=prompt_content)])
        cleaned_response = clean_and_format_output(response.content)

        return {
            "messages": [AIMessage(content=cleaned_response, name="final_writer")],
            "final_response": cleaned_response
        }

    return final_writer_node


def route_after_doc_reader(state: AgentState) -> str:
    """Menentukan tujuan berikutnya setelah Doc Reader selesai membaca dokumen."""
    plan = state.get("execution_plan", [])
    if "data_analyst" in plan:
        return "data_analyst"
    elif "ml_specialist" in plan:
        return "ml_specialist"
    return "final_writer"


def route_after_data_analyst(state: AgentState) -> str:
    """Menentukan tujuan berikutnya setelah Data Analyst selesai memproses data."""
    plan = state.get("execution_plan", [])
    if "ml_specialist" in plan:
        return "ml_specialist"
    return "final_writer"


def build_multiagent_graph(
    gemini_api_key: str = "",
    model_name: str = "gemma-4-31b-it",
    provider: str = "gemini",
    base_url: Optional[str] = None
):
    """Membangun dan mengompilasi graf StateGraph Multi-Agent dengan LangGraph (Forward DAG)."""

    # Inisialisasi LLM melalui Universal Factory
    llm = get_llm(
        provider=provider,
        model_name=model_name,
        api_key=gemini_api_key,
        base_url=base_url,
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=3000
    )

    # Inisialisasi seluruh node pekerja
    supervisor_node = create_supervisor_node(llm)
    doc_reader_node = create_doc_reader_node(llm)
    data_analyst_node = create_data_analyst_node(llm)
    ml_specialist_node = create_ml_specialist_node(llm)
    final_writer_node = create_final_writer_node(llm)

    # Buat Graph State
    workflow = StateGraph(AgentState)

    # Registrasi simpul (nodes)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("doc_reader", doc_reader_node)
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("ml_specialist", ml_specialist_node)
    workflow.add_node("final_writer", final_writer_node)

    # Titik masuk utama selalu ke Supervisor
    workflow.set_entry_point("supervisor")

    # 1. Routing kondisional dari Supervisor ke titik awal eksekusi
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "final_writer"),
        {
            "doc_reader": "doc_reader",
            "data_analyst": "data_analyst",
            "ml_specialist": "ml_specialist",
            "final_writer": "final_writer"
        }
    )

    # 2. Routing setelah Doc Reader (Single-Dispatch -> final_writer, Multi-Stage -> data_analyst/ml_specialist)
    workflow.add_conditional_edges(
        "doc_reader",
        route_after_doc_reader,
        {
            "data_analyst": "data_analyst",
            "ml_specialist": "ml_specialist",
            "final_writer": "final_writer"
        }
    )

    # 3. Routing setelah Data Analyst (Single-Dispatch -> final_writer, Multi-Stage -> ml_specialist)
    workflow.add_conditional_edges(
        "data_analyst",
        route_after_data_analyst,
        {
            "ml_specialist": "ml_specialist",
            "final_writer": "final_writer"
        }
    )

    # 4. Routing setelah ML Specialist (Selalu langsung ke final_writer)
    workflow.add_edge("ml_specialist", "final_writer")

    # 5. Node final_writer adalah ujung akhir eksekusi graf
    workflow.add_edge("final_writer", END)

    return workflow.compile()

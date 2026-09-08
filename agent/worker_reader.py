"""Worker Node: Document & Multimodal Media Reader.
Mengekstrak teks dari dokumen (PDF/Word) serta menganalisis gambar diagram dan video secara visual dengan Gemini Vision.
"""

import os
import time
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools.doc_tools import read_pdf, read_word, resolve_file
from tools.media_tools import build_multimodal_content_blocks
from utils.formatters import clean_and_format_output
from agent.state import AgentState

DOC_READER_SYSTEM_PROMPT = """Anda adalah 'Document & Multimodal Reader Agent' spesialis dalam membaca dan menganalisis dokumen (PDF, Word .docx), gambar visual (diagram, grafik, flowchart, tulisan tangan), dan video.

Tugas Anda:
1. Membaca file dokumen yang dilampirkan oleh pengguna.
2. Menganalisis gambar atau video yang dilampirkan di chat secara visual.
3. Rangkum dan sajikan isi materi, temuan visual, dan soal/instruksi secara komprehensif untuk tim agen lainnya.
4. ATURAN ANTI-LOOPING: Dilarang mengulang kata atau frasa yang sama secara berulang kali. Berikan rangkuman yang jelas, padat, dan terstruktur.
"""

def create_doc_reader_node(llm: Any):
    """Factory untuk membuat node Document & Multimodal Reader yang cepat dan adaptif."""

    def doc_reader_node(state: AgentState) -> dict:
        uploaded_files = state.get("uploaded_files", {})
        attached_media = state.get("attached_media", [])
        
        file_list_str = "\n".join([f"- {fname}: {fpath}" for fname, fpath in uploaded_files.items()]) or "Tidak ada file dokumen terdaftar."
        system_msg = SystemMessage(content=f"{DOC_READER_SYSTEM_PROMPT}\n\nDaftar file dokumen:\n{file_list_str}")
        
        extracted_sections = []
        for fname, fpath in uploaded_files.items():
            real_path = resolve_file(fpath)
            if fname.lower().endswith('.pdf'):
                content = read_pdf.invoke({"file_path": real_path})
                extracted_sections.append(f"=== Dokumen PDF: {fname} ===\n{content}")
            elif fname.lower().endswith(('.docx', '.doc')):
                content = read_word.invoke({"file_path": real_path})
                extracted_sections.append(f"=== Dokumen Word: {fname} ===\n{content}")

        extracted_context = "\n\n".join(extracted_sections) if extracted_sections else ""

        user_query_str = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    user_query_str = m.content
                elif isinstance(m.content, list):
                    for block in m.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_query_str = block.get("text", "")

        prompt_text = f"""Informasi Dokumen Teks:
{extracted_context if extracted_context else "(Tidak ada dokumen teks terlampir)"}

Pertanyaan/Instruksi Pengguna:
{user_query_str}

Tolong berikan analisis menyeluruh dari dokumen teks dan media visual (gambar/video) yang dilampirkan."""

        # Jika ada lampiran gambar/video, gunakan format multimodal
        if attached_media:
            multimodal_blocks = build_multimodal_content_blocks(prompt_text, attached_media)
            human_msg = HumanMessage(content=multimodal_blocks)
        else:
            human_msg = HumanMessage(content=prompt_text)

        summary_response = llm.invoke([system_msg, human_msg])
        cleaned_content = clean_and_format_output(summary_response.content)
        
        new_messages = [
            AIMessage(content=cleaned_content, name="doc_reader")
        ]

        logs = state.get("activity_logs", [])
        media_count = len(attached_media)
        doc_count = len(extracted_sections)
        msg_detail = f"Berhasil memproses {doc_count} dokumen"
        if media_count > 0:
            msg_detail += f" dan {media_count} media visual (gambar/video)"
            
        logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": "📄 Doc & Vision Reader",
            "message": msg_detail + "."
        })

        return {
            "messages": new_messages,
            "activity_logs": logs
        }

    return doc_reader_node

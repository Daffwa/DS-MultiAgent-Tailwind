"""State Definition untuk Hierarchical Multi-Agent System (LangGraph).
Mendefinisikan skema state bersama yang dialirkan antar agen selama siklus hidup eksekusi data science.
Mendukung 3-Tier Adaptive Hybrid Orchestration (Fast-Path, Single-Dispatch, Multi-Stage Pipeline).
"""

from typing import Annotated, Sequence, TypedDict, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Skema data state sentral yang dialirkan ke seluruh simpul graf (Nodes)."""
    
    # Riwayat pesan obrolan dalam graf
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Nama simpul (agen) berikutnya yang harus dieksekusi
    next: str
    
    # Kategori tingkatan eksekusi adaptif: 'fast_path' | 'single_dispatch' | 'multi_stage'
    execution_tier: Optional[str]
    
    # Rencana urutan eksekusi agen yang direncanakan untuk tugas multi-tahap
    execution_plan: Optional[List[str]]
    
    # Kamus berkas yang terunggah {nama_file: path_file_lengkap}
    uploaded_files: Dict[str, str]
    
    # Daftar berkas media visual/gambar yang dilampirkan
    attached_media: List[str]
    
    # Log kronologis aktivitas setiap agen untuk live stepper di UI
    activity_logs: Annotated[List[Dict[str, Any]], operator.add]
    
    # Teks respon final dari perangkum (final_writer)
    final_response: str
    
    # Metrik hasil evaluasi Machine Learning (Akurasi, F1, R², Silhouette, Confusion Matrix)
    ml_metrics: Optional[Dict[str, Any]]
    
    # Hasil uji hipotesis statistik inferensial (t-test, ANOVA, p-value, normalitas)
    hypothesis_results: Optional[Dict[str, Any]]
    
    # Profil kesehatan dataset (missing value %, deteksi outlier IQR)
    data_profile: Optional[Dict[str, Any]]

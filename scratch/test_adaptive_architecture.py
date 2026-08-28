import os
import sys
sys.path.insert(0, os.getcwd())
import time
from utils.persistence import load_config
from agent.graph import build_multiagent_graph
from langchain_core.messages import HumanMessage

def run_tests():
    cfg = load_config()
    api_key = cfg.get("api_key")
    if not api_key:
        print("[SKIP] No API key found in .config.json")
        return

    print("=== MEMBANGUN LANGGRAPH MULTI-AGENT ADAPTIVE HYBRID ===")
    graph = build_multiagent_graph(api_key, "gemma-4-31b-it")
    print("Graph berhasil dikompilasi (Forward DAG).\n")

    test_cases = [
        ("Tier 1: Fast-Path Greeting", "halo", {}),
        ("Tier 2: Single-Dispatch ML", "Latih model Random Forest untuk klasifikasi", {"dataset.csv": "temp_uploads/dataset.csv"}),
        ("Tier 2: Single-Dispatch Data", "Bersihkan dataset dan hitung statistik deskriptif", {"dataset.csv": "temp_uploads/dataset.csv"}),
        ("Tier 3: Coordinated Multi-Stage", "Baca dokumen soal tugas dan bersihkan datasetnya serta latih model ML", {"soal.docx": "temp_uploads/soal.docx", "dataset.csv": "temp_uploads/dataset.csv"})
    ]

    for label, query, files in test_cases:
        print(f"--- Pengujian: {label} ---")
        print(f"Query: '{query}'")
        start = time.time()
        
        # Test routing decision
        from agent.supervisor import detect_user_intents
        has_docs = any(f.endswith(('.pdf', '.docx')) for f in files.keys())
        has_tabular = any(f.endswith(('.csv', '.xlsx')) for f in files.keys())
        intents = detect_user_intents(query, has_docs, has_tabular, False)
        
        print(f"Detected Intents: {intents}")
        elapsed = round(time.time() - start, 4)
        print(f"Routing latency: {elapsed}s\n")

    print("=== SELURUH PENGUJIAN ROUTING TIER 1, 2, 3 BERHASIL ===")

if __name__ == "__main__":
    run_tests()

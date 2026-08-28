"""Storage and Persistence Manager for Data Science Multi-Agent Assistant.
Menangani penyimpanan persisten lokal untuk API Key, preferensi model, dan riwayat percakapan.
"""

import os
import json
import glob
from typing import List, Dict, Any, Tuple
from utils.formatters import clean_and_format_output

CONFIG_FILE = os.path.join(os.getcwd(), ".config.json")
HISTORY_FILE = os.path.join(os.getcwd(), "chat_history.json")
GENERATED_PLOTS_DIR = os.path.join(os.getcwd(), "generated_plots")
TEMP_UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")

# Pastikan folder penyimpanan tersedia
os.makedirs(GENERATED_PLOTS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Memuat konfigurasi tersimpan (API Key & Pilihan Model) dari disk."""
    default_config = {
        "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        "selected_model": "gemma-4-31b-it"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception:
            pass
            
    return default_config


def save_config(api_key: str, selected_model: str) -> None:
    """Menyimpan API Key dan pilihan model ke file .config.json agar tetap ada saat di-refresh."""
    data = {
        "api_key": api_key,
        "selected_model": selected_model
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan config: {e}")


def load_chat_history() -> List[Dict[str, Any]]:
    """Memuat seluruh riwayat pesan dari file lokal dan membersihkan jika ada token loop."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    # Sanitasi setiap pesan historis agar tidak ada loop kata lama
                    for msg in history:
                        if "content" in msg and isinstance(msg["content"], str):
                            msg["content"] = clean_and_format_output(msg["content"])
                    return history
        except Exception:
            pass
    return []


def save_chat_history(messages: List[Dict[str, Any]]) -> None:
    """Menyimpan riwayat chat ke file chat_history.json setelah disanitasi."""
    try:
        sanitized = []
        for msg in messages:
            msg_copy = dict(msg)
            if "content" in msg_copy and isinstance(msg_copy["content"], str):
                msg_copy["content"] = clean_and_format_output(msg_copy["content"])
            sanitized.append(msg_copy)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan riwayat chat: {e}")


def clear_all_history() -> None:
    """Menghapus riwayat percakapan dan file plot sementara saat tombol reset ditekan."""
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass
            
    for plot_path in glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")):
        try:
            os.remove(plot_path)
        except Exception:
            pass

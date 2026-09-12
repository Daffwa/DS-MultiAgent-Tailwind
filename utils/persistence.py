"""Storage and Persistence Manager for Data Science Multi-Agent Assistant.
Menangani penyimpanan persisten lokal untuk:
1. Konfigurasi (API Key, Provider, preferensi model Cloud & Local).
2. Multi-Session Chat ala ChatGPT / Antigravity (folder chat_sessions/).
3. Indeks sesi obrolan (sessions_index.json) dan migrasi otomatis riwayat lama.
"""

import os
import json
import glob
import time
import uuid
from typing import List, Dict, Any, Optional
from utils.formatters import clean_and_format_output

CONFIG_FILE = os.path.join(os.getcwd(), ".config.json")
HISTORY_FILE = os.path.join(os.getcwd(), "chat_history.json")
GENERATED_PLOTS_DIR = os.path.join(os.getcwd(), "generated_plots")
TEMP_UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")
SESSIONS_DIR = os.path.join(os.getcwd(), "chat_sessions")
SESSIONS_INDEX_FILE = os.path.join(SESSIONS_DIR, "sessions_index.json")

# Pastikan seluruh folder penyimpanan tersedia
os.makedirs(GENERATED_PLOTS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)


# =============================================================================
# 1. Konfigurasi Model (Cloud Gemini & Local LLM)
# =============================================================================
def load_config() -> Dict[str, Any]:
    """Memuat konfigurasi tersimpan (Provider, API Key, Local LLM & Pilihan Model) dari disk."""
    default_config = {
        "provider": "gemini",
        "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        "selected_model": "gemma-4-31b-it",
        "local_base_url": os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:11434/v1",
        "local_model_name": os.getenv("LOCAL_LLM_MODEL") or "qwen2.5-coder:7b",
        "local_api_key": os.getenv("LOCAL_LLM_API_KEY") or "ollama"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception:
            pass
            
    return default_config


def save_config(
    api_key: str,
    selected_model: str,
    provider: str = "gemini",
    local_base_url: str = "http://localhost:11434/v1",
    local_model_name: str = "qwen2.5-coder:7b",
    local_api_key: str = "ollama"
) -> None:
    """Menyimpan seluruh konfigurasi (Cloud & Local) ke file .config.json."""
    data = {
        "provider": provider,
        "api_key": api_key,
        "selected_model": selected_model,
        "local_base_url": local_base_url,
        "local_model_name": local_model_name,
        "local_api_key": local_api_key
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan config: {e}")


# =============================================================================
# 2. Multi-Session Chat Persistence (ChatGPT / Antigravity Style)
# =============================================================================
def _load_sessions_index() -> List[Dict[str, Any]]:
    """Membaca daftar sesi dari sessions_index.json."""
    if os.path.exists(SESSIONS_INDEX_FILE):
        try:
            with open(SESSIONS_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[Warning] Gagal membaca sessions_index.json: {e}")
    return []


def _save_sessions_index(index_data: List[Dict[str, Any]]) -> None:
    """Menyimpan daftar sesi ke sessions_index.json."""
    try:
        with open(SESSIONS_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan sessions_index.json: {e}")


def _ensure_initial_migration():
    """Migrasi otomatis dari chat_history.json lama ke format multi-session jika belum ada sesi."""
    index_data = _load_sessions_index()
    if index_data:
        return

    # Cek apakah ada riwayat lama di chat_history.json
    legacy_messages = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    legacy_messages = data
        except Exception:
            pass

    now_str = time.strftime("%d/%m/%Y %H:%M")
    default_session_id = "session_default"
    default_title = "Sesi Utama"

    # Jika ada pesan di chat_history.json, coba cari judul dari prompt pertama
    if legacy_messages:
        for m in legacy_messages:
            if m.get("role") == "user" and m.get("content"):
                first_line = m["content"].strip().split("\n")[0]
                words = first_line.split()[:6]
                if words:
                    default_title = " ".join(words)[:40]
                break

    session_obj = {
        "id": default_session_id,
        "title": default_title,
        "created_at": now_str,
        "updated_at": now_str,
        "message_count": len(legacy_messages),
        "messages": legacy_messages
    }

    # Simpan session file
    session_file = os.path.join(SESSIONS_DIR, f"{default_session_id}.json")
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan default session file: {e}")

    # Simpan index
    _save_sessions_index([{
        "id": default_session_id,
        "title": default_title,
        "created_at": now_str,
        "updated_at": now_str,
        "message_count": len(legacy_messages)
    }])


def list_all_sessions() -> List[Dict[str, Any]]:
    """Mengambil daftar seluruh sesi terdaftar, diurutkan dari yang paling baru diupdate."""
    _ensure_initial_migration()
    sessions = _load_sessions_index()
    return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)


def create_new_session(title: Optional[str] = None) -> Dict[str, Any]:
    """Membuat sesi percakapan baru yang kosong dan menyimpannya ke disk."""
    _ensure_initial_migration()
    session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_str = time.strftime("%d/%m/%Y %H:%M")
    session_title = title.strip() if title and title.strip() else "Sesi Baru"

    new_session_data = {
        "id": session_id,
        "title": session_title,
        "created_at": now_str,
        "updated_at": now_str,
        "message_count": 0,
        "messages": []
    }

    # 1. Tulis file JSON sesi
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(new_session_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Gagal membuat file sesi {session_id}: {e}")

    # 2. Update index
    index_data = _load_sessions_index()
    index_data.insert(0, {
        "id": session_id,
        "title": session_title,
        "created_at": now_str,
        "updated_at": now_str,
        "message_count": 0
    })
    _save_sessions_index(index_data)

    return new_session_data


def get_session_details(session_id: str) -> Optional[Dict[str, Any]]:
    """Membaca isi lengkap sesi (termasuk riwayat pesan tersanitasi) berdasarkan session_id."""
    _ensure_initial_migration()
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(session_file):
        if session_id == "session_default" and os.path.exists(HISTORY_FILE):
            legacy_msgs = load_chat_history()
            return {
                "id": "session_default",
                "title": "Sesi Utama",
                "created_at": time.strftime("%d/%m/%Y %H:%M"),
                "updated_at": time.strftime("%d/%m/%Y %H:%M"),
                "message_count": len(legacy_msgs),
                "messages": legacy_msgs
            }
        return None

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "messages" in data and isinstance(data["messages"], list):
                    for msg in data["messages"]:
                        if "content" in msg and isinstance(msg["content"], str):
                            msg["content"] = clean_and_format_output(msg["content"])
                return data
    except Exception as e:
        print(f"[Warning] Gagal membaca file sesi {session_id}: {e}")

    return None


def save_session_history(session_id: str, messages: List[Dict[str, Any]], custom_title: Optional[str] = None) -> None:
    """Menyimpan riwayat pesan ke file sesi tertentu dan memperbarui sessions_index.json."""
    _ensure_initial_migration()
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    now_str = time.strftime("%d/%m/%Y %H:%M")

    sanitized_messages = []
    for msg in messages:
        msg_copy = dict(msg)
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            msg_copy["content"] = clean_and_format_output(msg_copy["content"])
        sanitized_messages.append(msg_copy)

    current_data = get_session_details(session_id) or {
        "id": session_id,
        "title": "Sesi Baru",
        "created_at": now_str,
        "updated_at": now_str,
        "message_count": 0,
        "messages": []
    }

    current_title = current_data.get("title", "Sesi Baru")
    if custom_title and custom_title.strip():
        current_title = custom_title.strip()
    elif current_title in ["Sesi Baru", "New Chat", ""]:
        for m in sanitized_messages:
            if m.get("role") == "user" and m.get("content"):
                first_line = m["content"].strip().split("\n")[0]
                words = first_line.split()[:7]
                if words:
                    current_title = " ".join(words)[:40]
                break

    current_data["title"] = current_title
    current_data["updated_at"] = now_str
    current_data["message_count"] = len(sanitized_messages)
    current_data["messages"] = sanitized_messages

    # 1. Tulis ke file sesi
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Gagal menyimpan file sesi {session_id}: {e}")

    # 2. Update index
    index_data = _load_sessions_index()
    found = False
    for s in index_data:
        if s.get("id") == session_id:
            s["title"] = current_title
            s["updated_at"] = now_str
            s["message_count"] = len(sanitized_messages)
            found = True
            break

    if not found:
        index_data.insert(0, {
            "id": session_id,
            "title": current_title,
            "created_at": current_data.get("created_at", now_str),
            "updated_at": now_str,
            "message_count": len(sanitized_messages)
        })

    _save_sessions_index(index_data)

    # 3. Tetap sinkronkan chat_history.json untuk kompatibilitas ke belakang
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitized_messages, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def rename_session(session_id: str, new_title: str) -> bool:
    """Mengubah nama/judul sesi obrolan."""
    _ensure_initial_migration()
    new_title_clean = new_title.strip()
    if not new_title_clean:
        return False

    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    now_str = time.strftime("%d/%m/%Y %H:%M")

    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["title"] = new_title_clean
            data["updated_at"] = now_str
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Gagal rename file sesi {session_id}: {e}")
            return False

    index_data = _load_sessions_index()
    for s in index_data:
        if s.get("id") == session_id:
            s["title"] = new_title_clean
            s["updated_at"] = now_str
            break
    _save_sessions_index(index_data)
    return True


def delete_session(session_id: str) -> bool:
    """Menghapus sesi tertentu dari disk lokal."""
    _ensure_initial_migration()
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if os.path.exists(session_file):
        try:
            os.remove(session_file)
        except Exception as e:
            print(f"[Warning] Gagal menghapus file sesi {session_id}: {e}")

    index_data = _load_sessions_index()
    index_data = [s for s in index_data if s.get("id") != session_id]

    if not index_data:
        _save_sessions_index([])
        create_new_session("Sesi Utama")
    else:
        _save_sessions_index(index_data)

    return True


def clear_session_messages(session_id: str) -> bool:
    """Mengosongkan pesan dalam satu sesi tanpa menghapus sesi itu sendiri."""
    save_session_history(session_id, [], custom_title="Sesi Bersih")
    return True


# =============================================================================
# 3. Fungsi Kompatibilitas ke Belakang (Legacy Helpers)
# =============================================================================
def load_chat_history() -> List[Dict[str, Any]]:
    """Memuat riwayat pesan (kompatibilitas backward ke sesi teratas / default)."""
    _ensure_initial_migration()
    sessions = list_all_sessions()
    if sessions:
        top_session = get_session_details(sessions[0]["id"])
        if top_session and "messages" in top_session:
            return top_session["messages"]

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    for msg in history:
                        if "content" in msg and isinstance(msg["content"], str):
                            msg["content"] = clean_and_format_output(msg["content"])
                    return history
        except Exception:
            pass
    return []


def save_chat_history(messages: List[Dict[str, Any]]) -> None:
    """Menyimpan riwayat chat (kompatibilitas backward ke sesi teratas / default)."""
    _ensure_initial_migration()
    sessions = list_all_sessions()
    target_session_id = sessions[0]["id"] if sessions else "session_default"
    save_session_history(target_session_id, messages)


def clear_all_history() -> None:
    """Menghapus seluruh file plot dan mengosongkan riwayat sesi obrolan."""
    _ensure_initial_migration()
    sessions = list_all_sessions()
    for s in sessions:
        delete_session(s["id"])

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

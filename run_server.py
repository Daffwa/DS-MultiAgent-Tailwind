"""Bootstrap Runner untuk Menjalankan Backend FastAPI + Frontend Tailwind CSS secara Otomatis.
"""

import os
import sys
import time
import socket
import threading
import webbrowser
import uvicorn

def find_free_port(start_port: int = 8000, max_attempts: int = 50) -> int:
    """Mencari port kosong agar tidak bentrok dengan server lain."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start_port

def launch_browser(port: int):
    """Membuka browser default ke URL aplikasi."""
    time.sleep(1.8)
    webbrowser.open(f"http://localhost:{port}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_free_port(8000)
    print(f"[*] Menjalankan Data Science Multi-Agent (Tailwind + FastAPI) di http://localhost:{port}")

    threading.Thread(target=launch_browser, args=(port,), daemon=True).start()

    from server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()

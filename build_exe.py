"""Script otomatis untuk mem-build aplikasi Multi-Agent + Tailwind CSS menjadi file .exe mandiri menggunakan PyInstaller.
"""

import os
import sys
import subprocess
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_install_pyinstaller():
    """Memastikan PyInstaller terpasang."""
    try:
        import PyInstaller
        print("[1/3] PyInstaller sudah terpasang.")
    except ImportError:
        print("[1/3] Memasang PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def run_build():
    """Menjalankan proses kompilasi .exe."""
    os.chdir(PROJECT_DIR)
    check_and_install_pyinstaller()

    print("\n[2/3] Memulai proses kompilasi .exe (FastAPI + Tailwind CSS)...")

    # Command PyInstaller lengkap
    build_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "DS-MultiAgent-Tailwind",
        
        # Metadata dan modul penting
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "starlette",
        "--collect-all", "pydantic",
        "--collect-all", "langchain",
        "--collect-all", "langgraph",
        "--collect-all", "langchain_google_genai",
        "--collect-all", "langchain_core",
        "--collect-all", "langchain_experimental",
        "--collect-all", "pypdf",
        "--collect-all", "docx",
        "--collect-all", "pandas",
        "--collect-all", "openpyxl",
        "--collect-all", "matplotlib",
        "--collect-all", "seaborn",
        "--collect-all", "plotly",
        "--collect-all", "dotenv",
        
        # Hidden imports eksplisit
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "python_multipart",
        
        # Folder kode lokal & static assets
        "--add-data", f"{os.path.join(PROJECT_DIR, 'server.py')};.",
        "--add-data", f"{os.path.join(PROJECT_DIR, 'static')};static",
        "--add-data", f"{os.path.join(PROJECT_DIR, 'agent')};agent",
        "--add-data", f"{os.path.join(PROJECT_DIR, 'tools')};tools",
        "--add-data", f"{os.path.join(PROJECT_DIR, 'utils')};utils",
        
        "run_server.py"
    ]

    try:
        subprocess.check_call(build_cmd)

        dist_app_dir = os.path.join(PROJECT_DIR, "dist", "DS-MultiAgent-Tailwind")
        os.makedirs(os.path.join(dist_app_dir, "temp_uploads"), exist_ok=True)
        os.makedirs(os.path.join(dist_app_dir, "generated_files"), exist_ok=True)
        os.makedirs(os.path.join(dist_app_dir, "generated_plots"), exist_ok=True)

        env_ex = os.path.join(PROJECT_DIR, ".env.example")
        if os.path.exists(env_ex):
            shutil.copy(env_ex, dist_app_dir)

        print("\n" + "="*65)
        print("[3/3] SUKSES! Program .exe Tailwind CSS berhasil dibuat.")
        print(f"Lokasi File .exe: {os.path.join(dist_app_dir, 'DS-MultiAgent-Tailwind.exe')}")
        print("="*65)

    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Proses build gagal: {e}")

if __name__ == "__main__":
    run_build()

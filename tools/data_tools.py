"""Tools Data Science & Python Sandbox Execution Engine.
Dilengkapi Full-Stack Machine Learning Suite (Supervised & Unsupervised Learning),
Self-Healing Error Recovery, Inspeksi Dataset Otomatis, dan Manajemen Visualisasi.
"""

import os
import sys
import io
import glob
import time
import traceback
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

# =========================================================================
# Pustaka Machine Learning Lengkap (Supervised & Unsupervised)
# =========================================================================
try:
    import sklearn
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        IsolationForest, ExtraTreesClassifier
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score,
        r2_score, mean_squared_error, mean_absolute_error,
        silhouette_score, davies_bouldin_score
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
    )
except ImportError:
    sklearn = None

# Pustaka Statistik Inferensial
try:
    import scipy
    from scipy import stats
except ImportError:
    scipy = None

# Pustaka High-Performance Data Processing (Big Data & Streaming)
try:
    import polars as pl
except ImportError:
    pl = None

try:
    import duckdb
except ImportError:
    duckdb = None


def resolve_file(file_path: str) -> str:
    """Mencari lokasi file yang valid di direktori lokal atau temp_uploads."""
    if os.path.exists(file_path):
        return file_path
    
    temp_dir = os.path.join(os.getcwd(), "temp_uploads")
    if os.path.exists(temp_dir):
        candidate = os.path.join(temp_dir, os.path.basename(file_path))
        if os.path.exists(candidate):
            return candidate
        for f in os.listdir(temp_dir):
            if os.path.basename(file_path).lower() in f.lower() or f.lower() in os.path.basename(file_path).lower():
                return os.path.join(temp_dir, f)
    return file_path


OUTPUT_PLOT_DIR = os.path.join(os.getcwd(), "generated_plots")
OUTPUT_FILES_DIR = os.path.join(os.getcwd(), "generated_files")
TEMP_UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


@tool
def inspect_tabular_data(file_path: str, sheet_name: str = None) -> str:
    """Membaca dan memeriksa struktur file dataset tabular (CSV, Excel .xlsx, atau Parquet).
    Mengembalikan ringkasan dimensi baris x kolom, tipe data, missing values, statistik, dan 5 baris pertama.
    Mendukung Smart Lazy Scanning untuk file besar (>50MB).
    """
    real_path = resolve_file(file_path)
    if not os.path.exists(real_path):
        return f"Error: Berkas '{file_path}' tidak ditemukan."

    file_size_mb = os.path.getsize(real_path) / (1024 * 1024)
    is_large_file = file_size_mb > 50.0

    try:
        sheet_info = ""
        large_file_note = ""

        if real_path.lower().endswith('.csv'):
            if is_large_file:
                # Mode Cepat & Hemat Memori untuk File Besar
                large_file_note = f"⚠️ [Big Data Mode]: Ukuran file {file_size_mb:.1f} MB (>50MB). Menampilkan profil ringkas 100 baris pertama untuk efisiensi RAM.\n\n"
                df = pd.read_csv(real_path, nrows=100)
                dim_str = f"Dimensi File: {file_size_mb:.1f} MB (Dataset Besar - Disarankan Polars/Streaming Pipeline)"
            else:
                df = pd.read_csv(real_path)
                dim_str = f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom"

        elif real_path.lower().endswith('.parquet'):
            if pl is not None:
                p_df = pl.read_parquet(real_path, n_rows=100 if is_large_file else None)
                df = p_df.to_pandas()
            else:
                df = pd.read_parquet(real_path)
            dim_str = f"Dimensi Dataset (Parquet): {df.shape[0]} baris x {df.shape[1]} kolom"

        elif real_path.lower().endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(real_path)
            sheet_names = excel_file.sheet_names
            target_sheet = sheet_name if sheet_name in sheet_names else sheet_names[0]
            df = pd.read_excel(real_path, sheet_name=target_sheet, nrows=100 if is_large_file else None)
            sheet_info = f"Daftar Sheet: {sheet_names}\nSheet aktif: '{target_sheet}'\n\n"
            dim_str = f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom"

        else:
            return f"Format berkas tidak didukung: '{file_path}'. Gunakan CSV, Excel (.xlsx), atau Parquet (.parquet)."

        summary = [
            sheet_info + large_file_note,
            dim_str,
            "\n--- Struktur Kolom & Missing Values ---",
            pd.DataFrame({
                "Kolom": df.columns,
                "Tipe Data": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum()
            }).to_string(index=False),
            "\n--- 5 Baris Pertama ---",
            df.head(5).to_string()
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary.append("\n--- Statistik Deskriptif ---")
            summary.append(df[numeric_cols].describe().to_string())

        return "\n".join(summary)
    except Exception as e:
        return f"Error saat membaca dataset: {str(e)}"


def attempt_self_healing_code(code: str, error_msg: str, exec_globals: dict) -> str:
    """Mesin Self-Healing: Menganalisis error traceback dan mengoreksi kode secara otomatis."""
    healed_code = code

    # 1. Error KeyError (Salah eja nama kolom)
    if "KeyError:" in error_msg:
        for var_name, val in exec_globals.items():
            if isinstance(val, pd.DataFrame):
                real_cols = list(val.columns)
                key_match = re.search(r"KeyError:\s*['\"](.*?)['\"]", error_msg)
                if key_match:
                    wrong_key = key_match.group(1)
                    for col in real_cols:
                        if wrong_key.lower() == col.lower() or wrong_key.lower() in col.lower() or col.lower() in wrong_key.lower():
                            healed_code = healed_code.replace(f"'{wrong_key}'", f"'{col}'").replace(f'"{wrong_key}"', f'"{col}"')
                            break

    # 2. Error FileNotFoundError
    if "FileNotFoundError" in error_msg:
        healed_code = re.sub(
            r"pd\.read_(csv|excel)\(\s*['\"](.*?)['\"]",
            r"pd.read_\1(resolve_file(r'\2')",
            healed_code
        )

    # 3. Missing imports fallback
    auto_imports = [
        ("train_test_split", "from sklearn.model_selection import train_test_split\n"),
        ("RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier\n"),
        ("RandomForestRegressor", "from sklearn.ensemble import RandomForestRegressor\n"),
        ("LogisticRegression", "from sklearn.linear_model import LogisticRegression\n"),
        ("LinearRegression", "from sklearn.linear_model import LinearRegression\n"),
        ("KMeans", "from sklearn.cluster import KMeans\n"),
        ("DBSCAN", "from sklearn.cluster import DBSCAN\n"),
        ("PCA", "from sklearn.decomposition import PCA\n"),
        ("IsolationForest", "from sklearn.ensemble import IsolationForest\n"),
        ("stats", "from scipy import stats\n")
    ]
    for symbol, imp_stmt in auto_imports:
        if f"NameError: name '{symbol}' is not defined" in error_msg:
            healed_code = imp_stmt + healed_code

    return healed_code


@tool
def execute_python_code(code: str) -> str:
    """Mengeksekusi kode Python di sandbox terisolasi dengan Full-Stack ML Suite & proteksi Self-Healing."""
    cleaned_code = code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[9:]
    elif cleaned_code.startswith("```"):
        cleaned_code = cleaned_code[3:]
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3]
    cleaned_code = cleaned_code.strip()

    # Enforce Agg headless backend to permanently eliminate Windows GUI popups
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
    except Exception:
        pass

    plots_before = set(glob.glob(os.path.join(OUTPUT_PLOT_DIR, "*")))
    files_before = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))

    # Hook custom plt.show agar setiap kali LLM menulis plt.show(), gambar otomatis disimpan ke disk tanpa popup
    def smart_plt_show(*args, **kwargs):
        if plt.get_fignums():
            plot_name = f"plot_{int(time.time() * 1000)}.png"
            auto_plot_path = os.path.join(OUTPUT_PLOT_DIR, plot_name)
            try:
                plt.savefig(auto_plot_path, dpi=200, bbox_inches='tight')
                plt.close('all')
                print(f"[Visualisasi Grafik Disimpan]: {plot_name}")
            except Exception:
                pass

    try:
        import matplotlib.pyplot as plt
        matplotlib.pyplot.show = smart_plt_show
        plt.show = smart_plt_show
    except Exception:
        pass

    # Prepend headless backend setup directly to the executed code
    headless_prefix = "import matplotlib\nmatplotlib.use('Agg', force=True)\nimport matplotlib.pyplot as plt\nplt.switch_backend('Agg')\n\n"
    cleaned_code = headless_prefix + cleaned_code

    # Environment Eksekusi Terisolasi & Terbuka untuk Seluruh Pustaka ML & Big Data
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
        "sklearn": sklearn,
        "scipy": scipy,
        "stats": stats if scipy else None,
        "pl": pl,
        "duckdb": duckdb,
        "OUTPUT_PLOT_DIR": OUTPUT_PLOT_DIR,
        "OUTPUT_FILES_DIR": OUTPUT_FILES_DIR,
        "TEMP_UPLOAD_DIR": TEMP_UPLOAD_DIR,
        "resolve_file": resolve_file
    }

    # Auto-preload active dataset DataFrame jika tersedia agar kode tidak error 'df is not defined'
    tabular_files = glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*.csv")) + glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*.xlsx"))
    if not tabular_files:
        tabular_files = glob.glob(r"D:\Capstone\Dataset\*.csv") + glob.glob(r"D:\Capstone\Dataset\*.xlsx")
    if tabular_files:
        try:
            target_f = tabular_files[0]
            if target_f.lower().endswith(".csv"):
                exec_globals["df"] = pd.read_csv(target_f)
            else:
                exec_globals["df"] = pd.read_excel(target_f)
        except Exception:
            pass

    if sklearn:
        exec_globals.update({
            # Model Validation & Selection
            "train_test_split": train_test_split,
            "cross_val_score": cross_val_score,
            "GridSearchCV": GridSearchCV,
            # Supervised Classification
            "RandomForestClassifier": RandomForestClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "GaussianNB": GaussianNB,
            # Supervised Regression
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            # Unsupervised Clustering & Reduction & Anomaly
            "KMeans": KMeans,
            "DBSCAN": DBSCAN,
            "AgglomerativeClustering": AgglomerativeClustering,
            "PCA": PCA,
            "IsolationForest": IsolationForest,
            # Metrics
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "roc_auc_score": roc_auc_score,
            "r2_score": r2_score,
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "silhouette_score": silhouette_score,
            "davies_bouldin_score": davies_bouldin_score,
            # Preprocessing
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "LabelEncoder": LabelEncoder,
            "OneHotEncoder": OneHotEncoder
        })

    stdout_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buffer

    execution_success = False
    self_healed = False
    error_detail = ""

    def run_sandbox_isolated(target_code: str):
        nonlocal execution_success, self_healed, error_detail
        try:
            exec(target_code, exec_globals)
            execution_success = True
        except Exception as e:
            error_detail = traceback.format_exc()
            # Percobaan Self-Healing Recovery
            try:
                healed_code = attempt_self_healing_code(target_code, error_detail, exec_globals)
                if healed_code != target_code:
                    exec(healed_code, exec_globals)
                    execution_success = True
                    self_healed = True
            except Exception:
                pass

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_sandbox_isolated, cleaned_code)
            future.result(timeout=30.0)
    except concurrent.futures.TimeoutError:
        execution_success = False
        error_detail = "⚠️ [Security Timeout Guard]: Eksekusi Python dibatalkan karena melebihi batas waktu maksimal (30 detik). Periksa apakah terdapat perulangan tak terbatas (infinite loop)."
    finally:
        sys.stdout = old_stdout

    # Simpan figure Matplotlib yang aktif jika ada
    if plt.get_fignums():
        try:
            auto_plot_path = os.path.join(OUTPUT_PLOT_DIR, f"plot_{int(time.time())}.png")
            plt.savefig(auto_plot_path, dpi=200, bbox_inches='tight')
            plt.close('all')
        except Exception:
            pass

    # Otomatis deteksi objek fig Plotly
    if "fig" in exec_globals and hasattr(exec_globals["fig"], "write_html"):
        try:
            plotly_html_path = os.path.join(OUTPUT_PLOT_DIR, f"plotly_chart_{int(time.time())}.html")
            exec_globals["fig"].write_html(plotly_html_path)
        except Exception:
            pass

    plots_after = set(glob.glob(os.path.join(OUTPUT_PLOT_DIR, "*")))
    files_after = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))

    new_plots = list(plots_after - plots_before)
    new_files = list(files_after - files_before)

    output_text = stdout_buffer.getvalue().strip()

    results = []
    if self_healed:
        results.append("🔄 [Self-Healing Engine]: Kode otomatis diperbaiki dan dieksekusi dengan sukses!")

    if execution_success:
        if output_text:
            results.append(f"Output Print:\n{output_text}")
        else:
            results.append("Eksekusi Python Sandbox selesai dengan sukses tanpa error.")
    else:
        results.append(f"Error eksekusi Python:\n{error_detail}")

    if new_files:
        results.append(f"\n[Berhasil Mengekspor Berkas]: {', '.join([os.path.basename(f) for f in new_files])}")

    if new_plots:
        results.append(f"\n[Berhasil Menyimpan Grafik]: {', '.join([os.path.basename(f) for f in new_plots])}")

    return "\n".join(results)

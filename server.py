"""FastAPI Server Backend untuk Hierarchical Multi-Agent System + Tailwind CSS Frontend.
Mendukung eksekusi 4 Agen Spesialis (Supervisor, Doc Reader, Data & Stats, ML Specialist),
streaming progress real-time (SSE), Self-Healing Sandbox, ML Model Evaluator, dan Standalone Executable.
"""

import os
import sys
import glob
import json
import shutil
import time
import asyncio
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Helper untuk path resolusi saat dibungkus ke .exe (PyInstaller)
def get_bundle_dir() -> str:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_runtime_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BUNDLE_DIR = get_bundle_dir()
RUNTIME_DIR = get_runtime_dir()

STATIC_DIR = os.path.join(BUNDLE_DIR, "static")
OUTPUT_FILES_DIR = os.path.join(RUNTIME_DIR, "generated_files")
TEMP_UPLOAD_DIR = os.path.join(RUNTIME_DIR, "temp_uploads")
GENERATED_PLOTS_DIR = os.path.join(RUNTIME_DIR, "generated_plots")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_PLOTS_DIR, exist_ok=True)

# Import modul internal multi-agent
from agent.graph import build_multiagent_graph
from utils.formatters import clean_and_format_output
from tools.data_tools import execute_python_code
from tools.doc_tools import read_pdf, read_word
from tools.report_tools import generate_academic_report
from utils.persistence import (
    load_config,
    save_config,
    load_chat_history,
    save_chat_history,
    clear_all_history,
    list_all_sessions,
    create_new_session,
    get_session_details,
    save_session_history,
    rename_session,
    delete_session,
    clear_session_messages
)

app = FastAPI(title="Data Science Multi-Agent Solver API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------
class SessionCreateRequest(BaseModel):
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    local_base_url: Optional[str] = None
    local_model_name: Optional[str] = None
    local_api_key: Optional[str] = None
    attached_media: Optional[List[str]] = []


class ConfigRequest(BaseModel):
    api_key: Optional[str] = ""
    selected_model: Optional[str] = "gemma-4-31b-it"
    provider: Optional[str] = "gemini"
    local_base_url: Optional[str] = "http://localhost:11434/v1"
    local_model_name: Optional[str] = "qwen2.5-coder:7b"
    local_api_key: Optional[str] = "ollama"


class PythonSandboxRequest(BaseModel):
    code: str


class DocInspectRequest(BaseModel):
    filename: str


class RoutingTestRequest(BaseModel):
    query: str


class MLSandboxTrainRequest(BaseModel):
    model_type: Optional[str] = "random_forest"
    test_size: Optional[float] = 0.2
    target_col: Optional[str] = None


# ---------------------------------------------------------
# API Endpoints: Konfigurasi & Riwayat
# ---------------------------------------------------------
@app.get("/api/config")
def get_configuration():
    cfg = load_config()
    return {
        "provider": cfg.get("provider", "gemini"),
        "api_key": cfg.get("api_key", ""),
        "selected_model": cfg.get("selected_model", "gemma-4-31b-it"),
        "local_base_url": cfg.get("local_base_url", "http://localhost:11434/v1"),
        "local_model_name": cfg.get("local_model_name", "qwen2.5-coder:7b"),
        "local_api_key": cfg.get("local_api_key", "ollama")
    }


@app.post("/api/config")
def update_configuration(req: ConfigRequest):
    save_config(
        api_key=req.api_key or "",
        selected_model=req.selected_model or "gemma-4-31b-it",
        provider=req.provider or "gemini",
        local_base_url=req.local_base_url or "http://localhost:11434/v1",
        local_model_name=req.local_model_name or "qwen2.5-coder:7b",
        local_api_key=req.local_api_key or "ollama"
    )
    return {"status": "success", "message": "Konfigurasi berhasil disimpan."}


@app.get("/api/local-models")
def get_local_models(base_url: Optional[str] = None):
    """Mengecek ketersediaan server Local LLM (Ollama / FreeToken / LM Studio) dan mengambil daftar model."""
    import urllib.request
    cfg = load_config()
    target_url = (base_url or cfg.get("local_base_url", "http://localhost:11434/v1")).strip()
    clean_url = target_url.rstrip("/")
    models = []
    status = "offline"

    # 1. Cek endpoint standar OpenAI /v1/models
    try:
        models_endpoint = clean_url if clean_url.endswith("/v1") else f"{clean_url}/v1"
        req = urllib.request.Request(f"{models_endpoint}/models", headers={"User-Agent": "DataScience-MultiAgent/3.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                if "data" in data and isinstance(data["data"], list):
                    models = [m.get("id") for m in data["data"] if m.get("id")]
                status = "online"
    except Exception:
        # 2. Cek endpoint bawaan Ollama /api/tags jika /v1/models gagal
        try:
            ollama_host = clean_url.replace("/v1", "")
            req = urllib.request.Request(f"{ollama_host}/api/tags", headers={"User-Agent": "DataScience-MultiAgent/3.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if "models" in data and isinstance(data["models"], list):
                        models = [m.get("name") for m in data["models"] if m.get("name")]
                    status = "online"
        except Exception:
            status = "offline"

    return {
        "status": status,
        "base_url": target_url,
        "models": models
    }


# ---------------------------------------------------------
# API Endpoints: Multi-Session Chat Management (ChatGPT / Antigravity Style)
# ---------------------------------------------------------
@app.get("/api/sessions")
def get_sessions_list():
    sessions = list_all_sessions()
    return {"sessions": sessions}


@app.post("/api/sessions")
def create_session_endpoint(req: Optional[SessionCreateRequest] = None):
    title = req.title if req else None
    session = create_new_session(title)
    return {"status": "success", "session": session}


@app.get("/api/sessions/{session_id}")
def get_session_endpoint(session_id: str):
    session = get_session_details(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesi percakapan tidak ditemukan.")
    return {"session": session}


@app.patch("/api/sessions/{session_id}")
def rename_session_endpoint(session_id: str, req: SessionRenameRequest):
    success = rename_session(session_id, req.title)
    if not success:
        raise HTTPException(status_code=400, detail="Gagal mengubah nama sesi.")
    return {"status": "success", "message": "Judul sesi berhasil diperbarui."}


@app.delete("/api/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    success = delete_session(session_id)
    return {"status": "success", "message": "Sesi berhasil dihapus."}


@app.post("/api/sessions/{session_id}/clear")
def clear_session_endpoint(session_id: str):
    clear_session_messages(session_id)
    return {"status": "success", "message": "Riwayat sesi berhasil dikosongkan."}


@app.get("/api/history")
def get_chat_history():
    history = load_chat_history()
    return {"history": history}


@app.post("/api/history/clear")
def clear_history():
    clear_all_history()
    return {"status": "success", "message": "Riwayat percakapan berhasil dibersihkan."}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_info = []
    for file in files:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f_out:
            content = await file.read()
            f_out.write(content)
        
        file_size_kb = round(len(content) / 1024, 2)
        uploaded_info.append({
            "filename": file.filename,
            "filepath": file_path,
            "size_kb": file_size_kb,
            "extension": os.path.splitext(file.filename)[1].lower()
        })
    return {"status": "success", "files": uploaded_info}


@app.get("/api/dataset/preview")
def get_dataset_preview():
    uploaded_files = glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*"))
    tabular_files = [f for f in uploaded_files if f.lower().endswith(('.csv', '.xlsx', '.xls'))]

    cleaned_files = sorted(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")), key=os.path.getmtime, reverse=True)
    tabular_cleaned = [f for f in cleaned_files if f.lower().endswith(('.csv', '.xlsx', '.xls'))]

    target_file = None
    if tabular_cleaned:
        target_file = tabular_cleaned[0]
    elif tabular_files:
        target_file = tabular_files[0]

    if not target_file or not os.path.exists(target_file):
        return {"status": "empty", "message": "Belum ada dataset yang diunggah."}

    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file)
        else:
            df = pd.read_excel(target_file)

        df_clean = df.fillna("")
        
        # Profiling Ringkas
        missing_count = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = round((missing_count / total_cells * 100), 2) if total_cells > 0 else 0.0
        # Smart EDA-to-ML Recommendation Engine
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        last_col = df.columns[-1] if len(df.columns) > 0 else ""
        unique_last = df[last_col].nunique() if last_col else 0

        if unique_last in [2, 3, 4, 5] or (last_col and df[last_col].dtype == 'object'):
            rec_task = "Supervised Klasifikasi"
            rec_algo = "Random Forest Classifier / Logistic Regression"
            rec_reason = f"Kolom '{last_col}' memiliki {unique_last} kategori kelas diskrit."
            rec_key = "random_forest"
        elif last_col in numeric_cols and unique_last > 10:
            rec_task = "Supervised Regresi"
            rec_algo = "Linear Regression / Random Forest Regressor"
            rec_reason = f"Kolom '{last_col}' bertipe numerik kontinu ({unique_last} nilai unik)."
            rec_key = "linear_regression"
        else:
            rec_task = "Unsupervised Clustering & PCA"
            rec_algo = "K-Means Clustering / PCA 2D"
            rec_reason = f"Dataset memiliki {len(numeric_cols)} fitur numerik tanpa target label eksplisit."
            rec_key = "kmeans"

        ml_rec = {
            "task": rec_task,
            "algorithm": rec_algo,
            "reason": rec_reason,
            "recommended_key": rec_key,
            "numeric_count": len(numeric_cols),
            "categorical_count": len(cat_cols)
        }

        return {
            "status": "success",
            "filename": os.path.basename(target_file),
            "total_rows": int(len(df)),
            "total_cols": int(len(df.columns)),
            "missing_count": missing_count,
            "missing_pct": missing_percentage,
            "ml_recommendation": ml_rec,
            "columns": list(df.columns),
            "rows": df_clean.head(50).to_dict(orient="records")
        }
    except Exception as e:
        return {"status": "error", "message": f"Gagal membaca dataset: {str(e)}"}


@app.get("/api/plots/list")
def get_all_plots():
    """Mengambil daftar seluruh plot grafik yang telah dihasilkan untuk tab Interactive Chart."""
    plot_files = sorted(
        glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")),
        key=os.path.getmtime,
        reverse=True
    )
    plots_data = []
    for p in plot_files:
        fname = os.path.basename(p)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.html')):
            plots_data.append({
                "filename": fname,
                "url": f"/api/plots/{fname}",
                "is_html": fname.lower().endswith('.html'),
                "created_at": time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(p)))
            })
    return {"plots": plots_data}


# ---------------------------------------------------------
# MACHINE LEARNING ARENA ENDPOINTS
# ---------------------------------------------------------
# MACHINE LEARNING ARENA & SUITE ENDPOINTS
# ---------------------------------------------------------
@app.get("/api/ml/metrics")
def get_ml_evaluation_metrics():
    """Mengambil seluruh artefak evaluasi Machine Learning (Supervised & Unsupervised)."""
    cm_path = os.path.join(GENERATED_PLOTS_DIR, "confusion_matrix.png")
    feat_path = os.path.join(GENERATED_PLOTS_DIR, "feature_importance.png")
    reg_path = os.path.join(GENERATED_PLOTS_DIR, "actual_vs_predicted.png")
    cluster_path = os.path.join(GENERATED_PLOTS_DIR, "cluster_scatter.png")
    pca_path = os.path.join(GENERATED_PLOTS_DIR, "pca_projection.png")
    anomaly_path = os.path.join(GENERATED_PLOTS_DIR, "anomaly_plot.png")

    has_cm = os.path.exists(cm_path)
    has_feat = os.path.exists(feat_path)
    has_reg = os.path.exists(reg_path)
    has_cluster = os.path.exists(cluster_path)
    has_pca = os.path.exists(pca_path)
    has_anomaly = os.path.exists(anomaly_path)

    any_active = any([has_cm, has_feat, has_reg, has_cluster, has_pca, has_anomaly])

    return {
        "status": "success" if any_active else "empty",
        "has_confusion_matrix": has_cm,
        "confusion_matrix_url": "/api/plots/confusion_matrix.png" if has_cm else None,
        "has_feature_importance": has_feat,
        "feature_importance_url": "/api/plots/feature_importance.png" if has_feat else None,
        "has_regression_plot": has_reg,
        "regression_plot_url": "/api/plots/actual_vs_predicted.png" if has_reg else None,
        "has_cluster_plot": has_cluster,
        "cluster_plot_url": "/api/plots/cluster_scatter.png" if has_cluster else None,
        "has_pca_plot": has_pca,
        "pca_plot_url": "/api/plots/pca_projection.png" if has_pca else None,
        "has_anomaly_plot": has_anomaly,
        "anomaly_plot_url": "/api/plots/anomaly_plot.png" if has_anomaly else None
    }


# ---------------------------------------------------------
# MULTI-AGENT INTERACTIVE SANDBOX ENDPOINTS
# ---------------------------------------------------------
@app.post("/api/sandbox/run-python")
def run_python_in_sandbox(req: PythonSandboxRequest):
    """Menjalankan kode Python kustom di sandbox dengan perlindungan Self-Healing."""
    start = time.time()
    plots_before = set(glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")))
    files_before = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))

    try:
        result_str = execute_python_code.invoke({"code": req.code})
        elapsed = round(time.time() - start, 2)

        plots_after = set(glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")))
        new_plots = [os.path.basename(p) for p in (plots_after - plots_before)]

        files_after = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))
        new_files = [os.path.basename(f) for f in (files_after - files_before)]

        return {
            "status": "success",
            "output": result_str,
            "elapsed_seconds": elapsed,
            "new_plots": new_plots,
            "new_files": new_files
        }
    except Exception as e:
        return {
            "status": "error",
            "output": f"Eksekusi Sandbox Error: {str(e)}",
            "elapsed_seconds": round(time.time() - start, 2)
        }


@app.post("/api/sandbox/train-ml-model")
def train_ml_model_sandbox(req: MLSandboxTrainRequest):
    """Melatih dan mengevaluasi model Machine Learning Supervised / Unsupervised di Sandbox."""
    start = time.time()
    uploaded_files = glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*"))
    tabular_files = [f for f in uploaded_files if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
    
    if not tabular_files:
        raise HTTPException(status_code=400, detail="Belum ada berkas dataset (CSV/Excel) yang diunggah.")

    target_file = tabular_files[0]
    read_func = f"pd.read_csv(r'{target_file}')" if target_file.lower().endswith('.csv') else f"pd.read_excel(r'{target_file}')"
    model_choice = req.model_type or "random_forest"
    test_size = float(req.test_size or 0.2)

    # 1. UNSUPERVISED: K-Means Clustering
    if model_choice in ["kmeans", "kmeans_clustering"]:
        ml_script = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, df['Cluster'])
    db = davies_bouldin_score(X_scaled, df['Cluster'])
    
    print("=== HASIL UNSUPERVISED CLUSTERING (K-MEANS) ===")
    print(f"Jumlah Klaster (K): {{k}}")
    print(f"Silhouette Score: {{sil:.4f}} (Kualitas Pemisahan Klaster)")
    print(f"Davies-Bouldin Index: {{db:.4f}}")
    print("\\nDistribusi Data per Klaster:\\n", df['Cluster'].value_counts())
    
    plt.figure(figsize=(7, 5))
    col1, col2 = numeric_df.columns[0], numeric_df.columns[1]
    sns.scatterplot(data=df, x=col1, y=col2, hue='Cluster', palette='viridis', s=80, edgecolor='black')
    plt.title(f'K-Means Clustering Scatter Plot (K={{k}})', fontsize=12)
    plt.tight_layout()
    plt.savefig('generated_plots/cluster_scatter.png', dpi=200)
    plt.close()
    print("Grafik klaster disimpan ke generated_plots/cluster_scatter.png")
else:
    print("Jumlah fitur numerik tidak mencukupi untuk clustering.")
"""

    # 2. UNSUPERVISED: PCA (Reduksi Dimensi)
    elif model_choice in ["pca", "pca_reduction"]:
        ml_script = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    print("=== HASIL UNSUPERVISED DIMENSIONALITY REDUCTION (PCA) ===")
    print(f"Explained Variance Ratio PC1: {{var_ratio[0]*100:.2f}}%")
    print(f"Explained Variance Ratio PC2: {{var_ratio[1]*100:.2f}}%")
    print(f"Total Variansi Tertangkap: {{sum(var_ratio)*100:.2f}}%")
    
    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='#06b6d4', edgecolors='black', alpha=0.8)
    plt.title('Principal Component Analysis (PCA) 2D Projection', fontsize=12)
    plt.xlabel(f'PC1 ({{var_ratio[0]*100:.1f}}%)')
    plt.ylabel(f'PC2 ({{var_ratio[1]*100:.1f}}%)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('generated_plots/pca_projection.png', dpi=200)
    plt.close()
    print("Grafik proyeksi PCA disimpan ke generated_plots/pca_projection.png")
else:
    print("Jumlah fitur numerik tidak mencukupi untuk PCA.")
"""

    # 3. UNSUPERVISED: Isolation Forest (Deteksi Anomali)
    elif model_choice in ["isolation_forest", "isolation_forest_anomaly"]:
        ml_script = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso.fit_predict(numeric_df)
    
    outliers_count = (df['Anomaly'] == -1).sum()
    print("=== HASIL UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST) ===")
    print(f"Jumlah Baris Terdeteksi Outlier/Anomali: {{outliers_count}} dari {{len(df)}} baris ({{outliers_count/len(df)*100:.1f}}%)")
    
    plt.figure(figsize=(7, 5))
    col1, col2 = numeric_df.columns[0], numeric_df.columns[1]
    plt.scatter(df[df['Anomaly']==1][col1], df[df['Anomaly']==1][col2], c='#10b981', label='Normal (Inlier)', alpha=0.7)
    plt.scatter(df[df['Anomaly']==-1][col1], df[df['Anomaly']==-1][col2], c='#ef4444', label='Anomali (Outlier)', s=60, edgecolors='black')
    plt.title('Deteksi Anomali Isolation Forest', fontsize=12)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('generated_plots/anomaly_plot.png', dpi=200)
    plt.close()
    print("Grafik anomali disimpan ke generated_plots/anomaly_plot.png")
else:
    print("Fitur tidak mencukupi untuk deteksi anomali.")
"""

    # 4. SUPERVISED: Regresi (Linear, Ridge, RF Regressor)
    elif model_choice in ["linear_regression", "ridge_regression", "random_forest_reg"]:
        reg_model = "LinearRegression()" if model_choice == "linear_regression" else ("Ridge(alpha=1.0)" if model_choice == "ridge_regression" else "RandomForestRegressor(n_estimators=100, random_state=42)")
        ml_script = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
    
    model = {reg_model}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("=== HASIL SUPERVISED REGRESSION ===")
    print(f"Algoritma: {model_choice.upper()}")
    print(f"R-Squared (R² Score): {{r2:.4f}}")
    print(f"Root Mean Squared Error (RMSE): {{rmse:.4f}}")
    print(f"Mean Absolute Error (MAE): {{mae:.4f}}")
    
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, color='#38bdf8', edgecolors='black', alpha=0.8, label='Prediksi Sampel')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal 1:1')
    plt.title(f'Regresi: Aktual vs Prediksi (R²={{r2:.2f}})', fontsize=12)
    plt.xlabel('Nilai Aktual')
    plt.ylabel('Nilai Prediksi')
    plt.legend()
    plt.tight_layout()
    plt.savefig('generated_plots/actual_vs_predicted.png', dpi=200)
    plt.close()
    print("Grafik regresi disimpan ke generated_plots/actual_vs_predicted.png")
else:
    print("Fitur tidak mencukupi untuk regresi.")
"""

    # 5. SUPERVISED: Klasifikasi (Random Forest, Logistic, SVC, Decision Tree, KNN)
    else:
        clf_map = {
            "logistic_regression": "LogisticRegression(max_iter=1000, random_state=42)",
            "svm_clf": "SVC(random_state=42)",
            "knn_clf": "KNeighborsClassifier(n_neighbors=5)",
            "decision_tree_clf": "DecisionTreeClassifier(random_state=42)",
            "random_forest": "RandomForestClassifier(n_estimators=100, random_state=42)"
        }
        clf_model = clf_map.get(model_choice, "RandomForestClassifier(n_estimators=100, random_state=42)")

        ml_script = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    if y.nunique() > 10:
        y = (y > y.median()).astype(int)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
    
    # Deteksi Imbalance Data
    class_counts = y.value_counts(normalize=True)
    is_imbalanced = (class_counts.min() < 0.20)
    
    model = {clf_model}
    if is_imbalanced and hasattr(model, 'class_weight'):
        model.class_weight = 'balanced'

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("=== HASIL SUPERVISED CLASSIFICATION (5-FOLD CV VALIDATED) ===")
    print(f"Algoritma Model: {model_choice.upper()}")
    print(f"Akurasi Pengujian (Test Accuracy): {{acc * 100:.2f}}%")
    print(f"Validasi Silang (5-Fold CV Score): {{cv_scores.mean() * 100:.2f}}% (+/- {{cv_scores.std() * 100:.2f}}%)")
    if is_imbalanced:
        print("Proteksi Data Tidak Seimbang: class_weight='balanced' diaktifkan.")
    print(f"Jumlah Sampel Train: {{len(X_train)}} | Test: {{len(X_test)}}")
    print("\\nLaporan Klasifikasi Lengkap:\\n", classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (5-Fold CV: {{cv_scores.mean()*100:.1f}}%)', fontsize=12)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Nilai Aktual')
    plt.tight_layout()
    plt.savefig('generated_plots/confusion_matrix.png', dpi=200)
    plt.close()
    
    # Plot Feature Importance jika didukung
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(8, 4))
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        feat_imp.plot(kind='barh', color='#a855f7')
        plt.title('Peringkat Pengaruh Fitur (Feature Importance)', fontsize=12)
        plt.xlabel('Tingkat Kepentingan (Importance Score)')
        plt.tight_layout()
        plt.savefig('generated_plots/feature_importance.png', dpi=200)
        plt.close()
        
    print("Grafik evaluasi model klasifikasi berhasil disimpan!")
else:
    print("Jumlah kolom numerik tidak mencukupi untuk pemodelan klasifikasi.")
"""

    try:
        output_str = execute_python_code.invoke({"code": ml_script})
        elapsed = round(time.time() - start, 2)
        return {
            "status": "success",
            "model_type": model_choice,
            "output": output_str,
            "elapsed_seconds": elapsed
        }
    except Exception as e:
        return {
            "status": "error",
            "output": f"Pelatihan ML Error: {str(e)}",
            "elapsed_seconds": round(time.time() - start, 2)
        }


@app.get("/api/sandbox/files")
def get_sandbox_uploaded_files():
    all_files = glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*"))
    doc_files = []
    for f in all_files:
        fname = os.path.basename(f)
        ext = os.path.splitext(fname)[1].lower()
        doc_files.append({
            "filename": fname,
            "type": "document" if ext in ['.pdf', '.docx', '.doc'] else ("image" if ext in ['.png', '.jpg', '.jpeg'] else "tabular"),
            "size_kb": round(os.path.getsize(f) / 1024, 1)
        })
    return {"files": doc_files}


@app.post("/api/sandbox/inspect-doc")
def inspect_document_sandbox(req: DocInspectRequest):
    file_path = os.path.join(TEMP_UPLOAD_DIR, req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Berkas tidak ditemukan.")

    ext = os.path.splitext(req.filename)[1].lower()
    start = time.time()
    try:
        if ext == '.pdf':
            extracted = read_pdf.invoke({"file_path": file_path, "max_pages": 20})
        elif ext in ['.docx', '.doc']:
            extracted = read_word.invoke({"file_path": file_path})
        elif ext in ['.png', '.jpg', '.jpeg']:
            extracted = f"[Media Gambar Terdeteksi]: '{req.filename}' siap diproses oleh Multimodal Vision AI."
        else:
            extracted = f"Format '{ext}' bukan dokumen teks."

        return {
            "status": "success",
            "filename": req.filename,
            "extracted_text": extracted,
            "elapsed_seconds": round(time.time() - start, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/sandbox/test-routing")
def test_supervisor_routing(req: RoutingTestRequest):
    q_lower = req.query.lower()
    uploaded = glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*"))
    has_docs = any(f.lower().endswith(('.pdf', '.docx', '.doc')) for f in uploaded)
    has_tabular = any(f.lower().endswith(('.csv', '.xlsx', '.xls')) for f in uploaded)

    selected_node = "final_writer"
    agent_name = "🏁 Lead Synthesizer"
    reason = "Pertanyaan percakapan umum tanpa eksekusi dataset."

    if any(k in q_lower for k in ["pdf", "word", "baca", "dokumen", "soal", "ekstrak", "gambar", "foto"]) and has_docs:
        selected_node = "doc_reader"
        agent_name = "📄 Doc & Vision Reader"
        reason = "Terdeteksi instruksi ekstraksi soal/dokumen dan berkas PDF/Word terunggah."
    elif any(k in q_lower for k in ["ml", "machine learning", "prediksi", "klasifikasi", "regresi", "random forest", "clustering", "model", "akurasi", "confusion matrix"]) and has_tabular:
        selected_node = "ml_specialist"
        agent_name = "🤖 Machine Learning Specialist"
        reason = "Terdeteksi instruksi pelatihan model prediktif Machine Learning atau kalkulasi Confusion Matrix."
    elif any(k in q_lower for k in ["clean", "bersih", "hitung", "statistik", "analisis", "grafik", "plot", "tren", "antrian", "uji hipotesis", "t-test", "anova"]) or has_tabular:
        selected_node = "data_analyst"
        agent_name = "📊 Data & Stats Engineer"
        reason = "Terdeteksi instruksi audit dataset, kalkulasi statistik inferensial, atau visualisasi tren."

    return {
        "query": req.query,
        "selected_agent": selected_node,
        "agent_name": agent_name,
        "reason": reason,
        "has_documents": has_docs,
        "has_tabular_data": has_tabular
    }


# ---------------------------------------------------------
# CHAT STREAMING & REPORT GENERATION (4 AGENTS PIPELINE)
# ---------------------------------------------------------
@app.post("/api/chat")
async def handle_chat_stream(req: ChatRequest):
    """Menjalankan alur 4 Agen Spesialis dengan streaming event status real-time (SSE)."""
    cfg = load_config()
    provider = (req.provider or cfg.get("provider", "gemini")).lower()

    if provider == "gemini":
        api_key = req.api_key or cfg.get("api_key", "")
        model_name = req.model_name or cfg.get("selected_model", "gemma-4-31b-it")
        base_url = None
        if not api_key:
            raise HTTPException(status_code=400, detail="API Key Google Gemini belum diatur.")
    else:
        api_key = req.local_api_key or cfg.get("local_api_key", "ollama")
        model_name = req.local_model_name or req.model_name or cfg.get("local_model_name", "qwen2.5-coder:7b")
        base_url = req.local_base_url or cfg.get("local_base_url", "http://localhost:11434/v1")

    uploaded_files_map = {}
    for f in glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*")):
        uploaded_files_map[os.path.basename(f)] = f

    # Tentukan ID sesi aktif
    target_session_id = req.session_id
    if not target_session_id:
        sessions = list_all_sessions()
        target_session_id = sessions[0]["id"] if sessions else create_new_session()["id"]

    session_data = get_session_details(target_session_id)
    history = session_data.get("messages", []) if session_data else []

    user_entry = {
        "role": "user",
        "content": req.query,
        "attached_media": req.attached_media
    }
    history.append(user_entry)
    save_session_history(target_session_id, history)

    try:
        agent_graph = build_multiagent_graph(
            gemini_api_key=api_key,
            model_name=model_name,
            provider=provider,
            base_url=base_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menginisialisasi LangGraph: {str(e)}")

    initial_state = {
        "messages": [HumanMessage(content=req.query)],
        "next": "supervisor",
        "execution_tier": None,
        "execution_plan": None,
        "uploaded_files": uploaded_files_map,
        "attached_media": req.attached_media or [],
        "activity_logs": [],
        "final_response": "",
        "ml_metrics": None,
        "hypothesis_results": None,
        "data_profile": None
    }

    start_time = time.time()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'agent': '👑 Supervisor Agent', 'message': 'Menganalisis instruksi pengguna dan merencanakan alur kerja 4 agen...', 'timestamp': time.strftime('%H:%M:%S'), 'elapsed': 0.1})}\n\n"
        await asyncio.sleep(0.05)

        plots_before = set(glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")))
        files_before = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))

        logs_list = []
        final_output_text = ""

        try:
            # Jalankan LangGraph stream
            for step_output in agent_graph.stream(initial_state, {"recursion_limit": 20}):
                step_elapsed = round(time.time() - start_time, 2)
                for node_name, node_state in step_output.items():
                    if "activity_logs" in node_state and node_state["activity_logs"]:
                        logs_list = node_state["activity_logs"]
                        latest_log = logs_list[-1]
                        latest_log["duration_str"] = f"{step_elapsed}s"
                        yield f"data: {json.dumps({'type': 'status', 'agent': latest_log.get('agent', node_name), 'message': latest_log.get('message', ''), 'timestamp': latest_log.get('timestamp', time.strftime('%H:%M:%S')), 'elapsed': step_elapsed})}\n\n"
                        await asyncio.sleep(0.05)

                    if "final_response" in node_state and node_state["final_response"]:
                        final_output_text = node_state["final_response"]
                    elif "messages" in node_state and node_state["messages"]:
                        last_msg = node_state["messages"][-1]
                        if hasattr(last_msg, "content") and last_msg.content:
                            final_output_text = last_msg.content

            # Pindai file output
            for ext in ("*.csv", "*.xlsx", "*.xls"):
                for root_f in glob.glob(os.path.join(RUNTIME_DIR, ext)):
                    dest_f = os.path.join(OUTPUT_FILES_DIR, os.path.basename(root_f))
                    if os.path.abspath(root_f) != os.path.abspath(dest_f):
                        try:
                            shutil.move(root_f, dest_f)
                        except Exception:
                            pass

            plots_after = set(glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*")))
            new_plots = [os.path.basename(p) for p in (plots_after - plots_before)]

            files_after = set(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")))
            new_files = [os.path.basename(f) for f in (files_after - files_before)]

            if not new_files and any(kw in req.query.lower() for kw in ["clean", "bersih", "edit", "csv", "excel", "download", "unduh"]):
                all_gen = sorted(glob.glob(os.path.join(OUTPUT_FILES_DIR, "*")), key=os.path.getmtime, reverse=True)
                if all_gen:
                    new_files = [os.path.basename(all_gen[0])]

            formatted_response = clean_and_format_output(final_output_text)
            if not formatted_response:
                formatted_response = "Instruksi data science berhasil diselesaikan oleh Tim Multi-Agent."

            total_duration = round(time.time() - start_time, 2)

            ai_entry = {
                "role": "assistant",
                "content": formatted_response,
                "plots": [os.path.join(GENERATED_PLOTS_DIR, p) for p in new_plots],
                "exported_files": [os.path.join(OUTPUT_FILES_DIR, f) for f in new_files],
                "activity_logs": logs_list,
                "total_duration": total_duration
            }
            history.append(ai_entry)
            save_session_history(target_session_id, history)

            yield f"data: {json.dumps({'type': 'complete', 'status': 'success', 'session_id': target_session_id, 'response': formatted_response, 'plots': new_plots, 'exported_files': new_files, 'activity_logs': logs_list, 'total_duration': total_duration})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/report/generate")
def create_report_endpoint(session_id: Optional[str] = None):
    cfg = load_config()
    if session_id:
        session_data = get_session_details(session_id)
        history = session_data.get("messages", []) if session_data else load_chat_history()
    else:
        history = load_chat_history()
    
    uploaded_files_map = {}
    for f in glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*")):
        uploaded_files_map[os.path.basename(f)] = f

    all_plots = glob.glob(os.path.join(GENERATED_PLOTS_DIR, "*"))

    try:
        report_path = generate_academic_report(
            chat_history=history,
            uploaded_files=uploaded_files_map,
            plot_files=all_plots,
            model_name=cfg.get("selected_model", "Gemma 4")
        )
        return {
            "status": "success",
            "filename": os.path.basename(report_path),
            "download_url": f"/api/files/download/{os.path.basename(report_path)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat laporan: {str(e)}")


@app.get("/api/files/download/{filename}")
def download_file(filename: str):
    candidate = os.path.join(OUTPUT_FILES_DIR, filename)
    if os.path.exists(candidate):
        return FileResponse(candidate, filename=filename)
    
    candidate_temp = os.path.join(TEMP_UPLOAD_DIR, filename)
    if os.path.exists(candidate_temp):
        return FileResponse(candidate_temp, filename=filename)
        
    raise HTTPException(status_code=404, detail="Berkas tidak ditemukan.")


@app.get("/api/plots/{filename}")
def get_plot_image(filename: str):
    plot_path = os.path.join(GENERATED_PLOTS_DIR, filename)
    if os.path.exists(plot_path):
        return FileResponse(plot_path)
    raise HTTPException(status_code=404, detail="Gambar plot tidak ditemukan.")


# Mount Static Frontend
css_path = os.path.join(STATIC_DIR, "css")
js_path = os.path.join(STATIC_DIR, "js")
if os.path.exists(css_path):
    app.mount("/css", StaticFiles(directory=css_path), name="css")
if os.path.exists(js_path):
    app.mount("/js", StaticFiles(directory=js_path), name="js")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

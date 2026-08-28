"""Worker Node: Machine Learning & Predictive Modeling Specialist Agent.
Spesialis Full-Stack Auto-ML untuk:
1. Supervised Learning - Klasifikasi (Random Forest, Logistic Regression, SVC, Decision Tree, Gradient Boosting)
2. Supervised Learning - Regresi (Linear Regression, Ridge, Lasso, SVR, Random Forest Regressor)
3. Unsupervised Learning - Pengelompokan (K-Means Clustering, DBSCAN, Agglomerative Clustering)
4. Unsupervised Learning - Reduksi Dimensi (PCA - Principal Component Analysis)
5. Unsupervised Learning - Deteksi Anomali (Isolation Forest Outlier Detection)
"""

import os
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.data_tools import inspect_tabular_data, execute_python_code, resolve_file
from utils.formatters import clean_and_format_output, extract_python_code
from agent.state import AgentState

ML_SPECIALIST_SYSTEM_PROMPT = """Anda adalah 'Machine Learning & Predictive Modeling Specialist Agent' tingkat lanjut.
Anda menguasai seluruh spektrum Supervised Learning (Klasifikasi & Regresi) dan Unsupervised Learning (Clustering, PCA, Deteksi Anomali).

TUGAS & STANDAR PIPELINE PYTHON:
1. Tulis kode Python LENGKAP di dalam blok:
```python
# Kode pipeline Machine Learning Anda di sini
```

2. CABANG ALGORITMA SESUAI KEBUTUHAN PENGGUNA:
   A. SUPERVISED KLASIFIKASI (Target: Kategori/Diskrit):
      - Algoritma: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, SVC, GradientBoostingClassifier.
      - Metrik: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
      - Visualisasi: Simpan Heatmap ke `generated_plots/confusion_matrix.png` dan `generated_plots/feature_importance.png`.
   
   B. SUPERVISED REGRESI (Target: Angka Kontinu/Prediksi Nilai):
      - Algoritma: LinearRegression, Ridge, Lasso, RandomForestRegressor, SVR.
      - Metrik: R-Squared (R²), RMSE, MAE.
      - Visualisasi: Simpan scatter plot Aktual vs Prediksi ke `generated_plots/actual_vs_predicted.png`.
   
   C. UNSUPERVISED CLUSTERING (Tanpa Label / Segmentasi):
      - Algoritma: KMeans, DBSCAN, AgglomerativeClustering.
      - Metrik: Silhouette Score, Davies-Bouldin Index.
      - Visualisasi: Simpan scatter plot klaster 2D ke `generated_plots/cluster_scatter.png`.
   
   D. UNSUPERVISED REDUKSI DIMENSI (Kompresi Fitur):
      - Algoritma: PCA (Principal Component Analysis).
      - Metrik: Explained Variance Ratio.
      - Visualisasi: Simpan proyeksi 2D ke `generated_plots/pca_projection.png`.

   E. UNSUPERVISED DETEKSI ANOMALI:
      - Algoritma: IsolationForest.
      - Visualisasi: Simpan sebaran outlier ke `generated_plots/anomaly_plot.png`.

3. SINKRONISASI DENGAN DATA ANALYST (EDA ➔ ML COOPERATION):
   - Selalu periksa pesan dan analisis dari 'data_analyst'. Jika Data Analyst telah mengaudit kolom dan memberikan 'Rekomendasi Machine Learning', jadikan rekomendasi tersebut sebagai acuan utama dalam pemilihan algoritma, target, dan metode preprocessing.

4. ATURAN ANTI-LOOPING: Dilarang keras membuat garis pembatas komentar berulang seperti '/////' atau '====='. Langsung tulis kode dan jelaskan kesimpulan akademisnya.
"""

def create_ml_specialist_node(llm: ChatGoogleGenerativeAI):
    """Factory untuk membuat node Machine Learning Specialist dengan dukungan Supervised & Unsupervised mutakhir."""

    def ml_specialist_node(state: AgentState) -> dict:
        uploaded_files = state.get("uploaded_files", {})
        tabular_files = [f for f in uploaded_files.keys() if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
        
        dataset_context = ""
        for fname in tabular_files:
            try:
                summary = inspect_tabular_data.invoke({"file_path": resolve_file(uploaded_files[fname])})
                dataset_context += f"\n\nRingkasan Struktur Data [{fname}]:\n{summary}\n"
            except Exception:
                pass

        system_msg = SystemMessage(
            content=f"{ML_SPECIALIST_SYSTEM_PROMPT}{dataset_context}"
        )

        user_query_str = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    user_query_str = m.content
                elif isinstance(m.content, list):
                    for block in m.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_query_str = block.get("text", "")

        q_lower = user_query_str.lower()
        prompt = f"Instruksi Pengguna: {user_query_str}\n\nSilakan rancang pipeline Machine Learning (Supervised/Unsupervised) yang tepat, latih modelnya, buat visualisasi evaluasinya, dan sajikan metrik performanya secara rinci."

        response = llm.invoke([system_msg, HumanMessage(content=prompt)])
        response_text = str(response.content)

        extracted_code = extract_python_code(response_text)

        # Fallback Cerdas Multi-Branch jika LLM tidak menyertakan blok kode
        if not extracted_code and tabular_files:
            target_f = tabular_files[0]
            target_path = resolve_file(uploaded_files[target_f])
            read_func = f"pd.read_csv(r'{target_path}')" if target_f.lower().endswith('.csv') else f"pd.read_excel(r'{target_path}')"

            # Branch 1: Unsupervised Clustering (K-Means)
            if any(k in q_lower for k in ["cluster", "klaster", "segmentasi", "kmeans", "k-means", "dbscan"]):
                extracted_code = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, df['Cluster'])
    print(f"=== HASIL UNSUPERVISED CLUSTERING (K-MEANS) ===")
    print(f"Jumlah Klaster (K): {{k}}")
    print(f"Silhouette Score: {{sil_score:.4f}} (Kualitas Pemisahan Klaster)")
    print("\\nDistribusi Jumlah Anggota per Klaster:\\n", df['Cluster'].value_counts())
    
    # Plot Scatter Klaster 2D
    plt.figure(figsize=(7, 5))
    col1, col2 = numeric_df.columns[0], numeric_df.columns[1]
    sns.scatterplot(data=df, x=col1, y=col2, hue='Cluster', palette='viridis', s=70, edgecolor='black')
    plt.title(f'Visualisasi Klaster K-Means (K={{k}})', fontsize=12)
    plt.tight_layout()
    plt.savefig('generated_plots/cluster_scatter.png', dpi=200)
    plt.close()
    print("Grafik klaster berhasil disimpan ke generated_plots/cluster_scatter.png")
else:
    print("Fitur numerik tidak mencukupi untuk clustering.")
"""

            # Branch 2: Unsupervised PCA (Reduksi Dimensi)
            elif any(k in q_lower for k in ["pca", "reduksi dimensi", "principal component"]):
                extracted_code = f"""import pandas as pd
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
    print(f"=== HASIL UNSUPERVISED REDUKSI DIMENSI (PCA) ===")
    print(f"Explained Variance Ratio PC1: {{var_ratio[0]*100:.2f}}%")
    print(f"Explained Variance Ratio PC2: {{var_ratio[1]*100:.2f}}%")
    print(f"Total Variansi Tertangkap: {{sum(var_ratio)*100:.2f}}%")
    
    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='#06b6d4', edgecolors='black', alpha=0.8)
    plt.title('Proyeksi 2D Principal Component Analysis (PCA)', fontsize=12)
    plt.xlabel(f'PC1 ({{var_ratio[0]*100:.1f}}%)')
    plt.ylabel(f'PC2 ({{var_ratio[1]*100:.1f}}%)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('generated_plots/pca_projection.png', dpi=200)
    plt.close()
    print("Grafik proyeksi PCA berhasil disimpan ke generated_plots/pca_projection.png")
else:
    print("Fitur numerik tidak mencukupi untuk PCA.")
"""

            # Branch 3: Supervised Regresi (Prediksi Nilai Kontinu)
            elif any(k in q_lower for k in ["regresi", "regression", "linear regression", "harga", "biaya", "lama", "durasi", "nilai kontinu"]):
                extracted_code = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"=== HASIL SUPERVISED REGRESSION (RANDOM FOREST) ===")
    print(f"Koefisien Determinasi (R² Score): {{r2:.4f}}")
    print(f"Root Mean Squared Error (RMSE): {{rmse:.4f}}")
    print(f"Mean Absolute Error (MAE): {{mae:.4f}}")
    
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, color='#38bdf8', edgecolors='black', alpha=0.8, label='Prediksi Sampel')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1')
    plt.title(f'Aktual vs Prediksi Regresi (R²={{r2:.2f}})', fontsize=12)
    plt.xlabel('Nilai Aktual')
    plt.ylabel('Nilai Prediksi')
    plt.legend()
    plt.tight_layout()
    plt.savefig('generated_plots/actual_vs_predicted.png', dpi=200)
    plt.close()
    print("Grafik evaluasi regresi berhasil disimpan ke generated_plots/actual_vs_predicted.png")
else:
    print("Fitur numerik tidak mencukupi untuk regresi.")
"""

            # Branch 4: Supervised Klasifikasi (Default)
            else:
                extracted_code = f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = {read_func}.dropna()
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) >= 2:
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    if y.nunique() > 10:
        y = (y > y.median()).astype(int)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Deteksi Ketidakseimbangan Kelas (Class Imbalance)
    class_counts = y.value_counts(normalize=True)
    is_imbalanced = (class_counts.min() < 0.20)
    cw = 'balanced' if is_imbalanced else None
    
    model = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    # 5-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"=== HASIL SUPERVISED CLASSIFICATION (5-FOLD CV VALIDATED) ===")
    print(f"Akurasi Pengujian (Test Accuracy): {acc * 100:.2f}%")
    print(f"Validasi Silang (5-Fold CV Accuracy): {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    if is_imbalanced:
        print("Proteksi Data Tidak Seimbang: class_weight='balanced' diaktifkan otomatis.")
    print("\nLaporan Klasifikasi Lengkap:\n", classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Heatmap (5-Fold Validated)', fontsize=12)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Nilai Aktual')
    plt.tight_layout()
    plt.savefig('generated_plots/confusion_matrix.png', dpi=200)
    plt.close()
    
    # Plot Feature Importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(8, 4))
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        feat_imp.plot(kind='barh', color='#a855f7')
        plt.title('Feature Importance Ranking', fontsize=12)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('generated_plots/feature_importance.png', dpi=200)
        plt.close()
        
    print("Visualisasi evaluasi model berhasil disimpan di generated_plots/")
else:
    print("Jumlah kolom numerik tidak mencukupi untuk pemodelan prediktif.")
"""

        execution_output = ""
        if extracted_code:
            try:
                execution_output = execute_python_code.invoke({"code": extracted_code})
            except Exception as e:
                execution_output = f"Gagal mengeksekusi pipeline Machine Learning: {str(e)}"

        combined_response = response_text
        if extracted_code and not extract_python_code(response_text):
            combined_response = f"Berikut adalah script Machine Learning yang dieksekusi:\n```python\n{extracted_code}\n```"

        if execution_output:
            combined_response += f"\n\n**Hasil Evaluasi Model Machine Learning (Output Terminal):**\n```text\n{execution_output}\n```"

        cleaned_content = clean_and_format_output(combined_response)

        new_messages = [
            AIMessage(content=cleaned_content, name="ml_specialist")
        ]

        logs = state.get("activity_logs", [])
        logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": "🤖 Machine Learning Specialist",
            "message": "Menyelesaikan pelatihan model Auto-ML (Supervised/Unsupervised) & sinkronisasi metrik."
        })

        return {
            "messages": new_messages,
            "activity_logs": logs
        }

    return ml_specialist_node

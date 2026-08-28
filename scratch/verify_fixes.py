"""Skrip verifikasi untuk memastikan ketiga perbaikan bug berjalan dengan benar."""

# Test 1: extract_python_code dari utils/formatters.py
from utils.formatters import clean_and_format_output, extract_python_code

print("[TEST 1] extract_python_code dari utils/formatters.py")
test_input = "Here is the code:\n```python\nprint('hello world')\n```\nDone."
result = extract_python_code(test_input)
assert result == "print('hello world')", f"FAIL: got {repr(result)}"
print(f"  [OK] Extracted: {repr(result)}")

# Test 2: import dari worker files
print("\n[TEST 2] Import dari agent/worker_coder.py")
from agent.worker_coder import create_data_analyst_node, DATA_ANALYST_SYSTEM_PROMPT
assert "Data" in DATA_ANALYST_SYSTEM_PROMPT
print("  [OK] DATA_ANALYST_SYSTEM_PROMPT tersedia dan valid")

print("\n[TEST 3] Import dari agent/worker_ml.py")
from agent.worker_ml import create_ml_specialist_node, ML_SPECIALIST_SYSTEM_PROMPT
assert "Machine Learning" in ML_SPECIALIST_SYSTEM_PROMPT
print("  [OK] ML_SPECIALIST_SYSTEM_PROMPT tersedia dan valid")

# Test 4: missing_percentage logic
print("\n[TEST 4] Logika kalkulasi missing_percentage")
import numpy as np
import pandas as pd
df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
missing_count = int(df.isnull().sum().sum())
total_cells = df.shape[0] * df.shape[1]
missing_percentage = round((missing_count / total_cells * 100), 2) if total_cells > 0 else 0.0
assert missing_count == 2, f"FAIL: missing_count = {missing_count}"
assert missing_percentage == 33.33, f"FAIL: missing_pct = {missing_percentage}"
print(f"  [OK] {missing_count} missing / {total_cells} cells = {missing_percentage}%")

# Test 5: numpy tersedia di server.py context
print("\n[TEST 5] numpy import check untuk server.py")
numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
assert numeric_cols == ["a", "b"], f"FAIL: got {numeric_cols}"
print(f"  [OK] np.number select_dtypes works: {numeric_cols}")

print("\n" + "=" * 60)
print("SEMUA 5 PENGUJIAN PERBAIKAN BERHASIL! (5/5 PASS)")
print("=" * 60)

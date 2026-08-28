import os
import sys
sys.path.insert(0, os.getcwd())
import time
from utils.persistence import load_config
from agent.graph import build_multiagent_graph
from langchain_core.messages import HumanMessage

cfg = load_config()
api_key = cfg.get("api_key")
if not api_key:
    print("No API key in config")
    exit(0)

start = time.time()
graph = build_multiagent_graph(api_key, "gemma-4-31b-it")
res = graph.invoke({
    "messages": [HumanMessage(content="halo")],
    "uploaded_files": {"dataset.csv": "temp_uploads/dataset.csv"},
    "attached_media": [],
    "activity_logs": []
})
elapsed = round(time.time() - start, 2)
print(f"=== GREETING RESPONSE TIME (WITH ACTIVE DATASET): {elapsed} seconds ===")
for log in res.get("activity_logs", []):
    print(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
print("\nFinal Response:\n", res.get("final_response"))

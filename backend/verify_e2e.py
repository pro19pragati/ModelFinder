"""End-to-end verification: run a regression experiment on the House Price dataset via the API."""
import time
import requests
import json

BASE = "http://127.0.0.1:8000"

# Start the server if not running
proc = None
try:
    import subprocess
    import sys
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="D:/AI_PRAGATI_ANNANT/backend",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
except Exception:
    pass

try:
    # Check health
    health = requests.get(f"{BASE}/api/health", timeout=10)
    print("Health check:", health.json())

    # Upload the House Price dataset
    path = "D:/AI_PRAGATI_ANNANT/backend/storage/uploads/3a0d7a5c-0caf-46b1-8059-50a39920b7fc_modelfinder_house_prices_regression.csv"
    with open(path, "rb") as f:
        upload = requests.post(f"{BASE}/api/datasets/upload", files={"file": ("house_prices.csv", f, "text/csv")}, timeout=30)
    print("Upload:", upload.status_code)
    dataset = upload.json()
    dataset_id = dataset["dataset"]["id"]
    print("Dataset ID:", dataset_id)

    # Start a regression session
    start = requests.post(f"{BASE}/api/experiments/start", json={
        "dataset_id": dataset_id,
        "target_column": "house_price_inr",
        "problem_type": "regression",
        "priority": "speed",
        "max_experiments": 3,
        "interactive": False,
        "auto_start": True,
        "name": "House Price Regression Fix Test",
    }, timeout=30)
    print("Start session:", start.status_code)
    session = start.json()
    session_id = session["id"]
    print("Session ID:", session_id)

    # Poll until completed
    for _ in range(90):
        prog = requests.get(f"{BASE}/api/experiments/sessions/{session_id}/progress", timeout=10).json()
        print(f"  Status: {prog['status']} | Stage: {prog['stage']} | Message: {prog.get('message', '')}")
        if prog["status"] in {"completed", "stopped", "failed"}:
            break
        time.sleep(1)

    # Get full session details
    status = requests.get(f"{BASE}/api/experiments/sessions/{session_id}", timeout=10).json()
    print()
    print("Session status:", status["session"]["status"])
    print("Stop reason:", status["session"].get("stop_reason"))
    print("Completed experiments:", status["session"]["completed_experiments"])
    print("Recommendation:", json.dumps(status["recommendation"], indent=2)[:500])
    print()
    for exp in status["experiments"]:
        print(f"  #{exp['experiment_number']}: {exp['model_name']} ({exp['status']}) -> R2={exp.get('metrics', {}).get('r2')}")

    # Cleanup
    requests.delete(f"{BASE}/api/experiments/sessions/{session_id}", timeout=10)
    requests.delete(f"{BASE}/api/datasets/{dataset_id}", timeout=10)
    print("\nCleanup done.")
finally:
    if proc:
        proc.terminate()
        proc.wait()
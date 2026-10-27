"""Integration and unit tests for the Autonomous ML Agent FastAPI application."""

import time
import pytest
from fastapi.testclient import TestClient

from main import app
from database import init_db
from config import settings

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    settings.ensure_directories()
    init_db()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Autonomous ML" in data["app"]

def test_settings(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert "max_upload_size_mb" in data
    assert "default_priority" in data
    assert len(data["priorities"]) == 3

def test_system_resources(client):
    response = client.get("/api/system/resources")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_count" in data
    assert "memory_total_mb" in data

def test_load_sample_dataset(client):
    response = client.post("/api/datasets/sample/linear_regression/load")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Sample Linear Regression"
    assert data["problem_type"] == "regression"
    assert data["is_analyzed"] is True
    assert "columns" in data
    assert "price" in data["columns"]
    dataset_id = data["id"]

    # Preview endpoint
    preview_res = client.get(f"/api/datasets/{dataset_id}/preview?limit=10")
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert len(preview_data["rows"]) == 10
    assert preview_data["total_rows"] > 0

    # Dataset detail
    detail_res = client.get(f"/api/datasets/{dataset_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == dataset_id

def test_dataset_reanalyze(client):
    # Load nonlinear regression sample
    load_res = client.post("/api/datasets/sample/nonlinear_regression/load")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["id"]

    # Trigger reanalysis
    analyze_res = client.post(
        f"/api/datasets/{dataset_id}/analyze",
        json={"target_column": "energy_demand_mw", "problem_type": "regression"},
    )
    assert analyze_res.status_code == 200
    analysis = analyze_res.json()["analysis"]
    assert analysis["problem_type"] == "regression"
    assert "linearity" in analysis
    assert analysis["linearity"]["verdict"] == "non-linear structure present"

def test_experiment_session_full_lifecycle(client):
    # 1. Load classification sample
    load_res = client.post("/api/datasets/sample/classification/load")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["id"]

    # 2. Start session with budget of 2 experiments, speed priority
    start_res = client.post(
        "/api/experiments/start",
        json={
            "dataset_id": dataset_id,
            "target_column": "churn",
            "problem_type": "classification",
            "priority": "speed",
            "max_experiments": 2,
            "interactive": False,
            "auto_start": True,
            "name": "API Test Classification Session",
        },
    )
    assert start_res.status_code == 200
    session_data = start_res.json()
    session_id = session_data["id"]
    assert session_data["problem_type"] == "classification"

    # 3. Poll for completion (up to 30 seconds)
    completed = False
    for _ in range(30):
        prog_res = client.get(f"/api/experiments/sessions/{session_id}/progress")
        assert prog_res.status_code == 200
        prog = prog_res.json()
        if prog["status"] in {"completed", "stopped", "failed"}:
            completed = True
            break
        time.sleep(1)

    assert completed, f"Session did not complete in time, status: {prog['status']}"

    # 4. Check full session status
    status_res = client.get(f"/api/experiments/sessions/{session_id}")
    assert status_res.status_code == 200
    full_status = status_res.json()
    assert full_status["session"]["status"] in {"completed", "stopped"}
    assert len(full_status["experiments"]) >= 1
    assert len(full_status["decisions"]) >= 1
    assert full_status["recommendation"] is not None

    # 5. Check comparison endpoint
    comp_res = client.get(f"/api/experiments/sessions/{session_id}/comparison")
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert "rows" in comp_data
    assert "charts" in comp_data
    assert len(comp_data["rows"]) >= 1

    # 6. Test prediction using the recommended champion model artifact
    # NOTE: The artifact is always saved on the priority-ranked winner (recommendation.experiment_id),
    # which may differ from session.best_experiment_id (raw best-by-metric).
    rec = full_status.get("recommendation") or {}
    recommended_exp_id = rec.get("experiment_id") or full_status["session"].get("best_experiment_id")
    if recommended_exp_id:
        exp_res = client.get(f"/api/experiments/{recommended_exp_id}")
        assert exp_res.status_code == 200
        exp_detail = exp_res.json()

        # Verify that the initial experiment has CV metrics recorded
        initial_exps = [e for e in full_status["experiments"] if e.get("kind") == "initial"]
        if initial_exps:
            assert initial_exps[0].get("cv_metrics") is not None, "Initial experiment must have cv_metrics"

        # Test predict endpoint with sample input row
        preview_res = client.get(f"/api/datasets/{dataset_id}/preview?limit=1")
        sample_row = preview_res.json()["rows"][0]
        # remove target column
        sample_row.pop("churn", None)

        pred_res = client.post(
            f"/api/experiments/{recommended_exp_id}/predict",
            json={"inputs": [sample_row]},
        )
        # Only assert 200 if the artifact exists on this experiment
        if exp_detail.get("artifact_path"):
            assert pred_res.status_code == 200
            pred_data = pred_res.json()
            assert "predictions" in pred_data
            assert len(pred_data["predictions"]) == 1

    # 7. Test CSV export of session results
    csv_res = client.get(f"/api/experiments/sessions/{session_id}/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type", "")
    assert "experiment_number" in csv_res.text
    assert "is_recommended" in csv_res.text

    # 8. Test session deletion
    del_sess_res = client.delete(f"/api/experiments/sessions/{session_id}")
    assert del_sess_res.status_code == 200
    assert del_sess_res.json()["ok"] is True

    # 9. Test dataset deletion
    del_ds_res = client.delete(f"/api/datasets/{dataset_id}")
    assert del_ds_res.status_code == 200
    assert del_ds_res.json()["ok"] is True


def test_not_found_handling(client):
    res = client.get("/api/datasets/non_existent_id")
    assert res.status_code == 404
    data = res.json()
    assert data["error"] == "not_found"


def test_regression_session_lifecycle(client):
    """Regression session end-to-end: load → start → complete → validate recommendation."""
    # 1. Load regression sample (linear) and verify dataset
    load_res = client.post("/api/datasets/sample/linear_regression/load")
    assert load_res.status_code == 200
    dataset = load_res.json()
    dataset_id = dataset["id"]
    assert dataset["problem_type"] == "regression"
    assert dataset["is_analyzed"] is True

    # 2. Start session: 3 experiments, performance priority
    start_res = client.post(
        "/api/experiments/start",
        json={
            "dataset_id": dataset_id,
            "target_column": "price",
            "problem_type": "regression",
            "priority": "performance",
            "max_experiments": 3,
            "interactive": False,
            "auto_start": True,
            "name": "API Test Regression Session",
        },
    )
    assert start_res.status_code == 200
    session_id = start_res.json()["id"]

    # 3. Poll until completed (up to 60 seconds for 3 experiments)
    completed = False
    for _ in range(60):
        prog = client.get(f"/api/experiments/sessions/{session_id}/progress").json()
        if prog["status"] in {"completed", "stopped", "failed"}:
            completed = True
            break
        time.sleep(1)
    assert completed, f"Regression session did not complete in time, status: {prog['status']}"

    # 4. Validate session details
    status_res = client.get(f"/api/experiments/sessions/{session_id}")
    assert status_res.status_code == 200
    full_status = status_res.json()
    assert full_status["session"]["status"] in {"completed", "stopped"}
    assert len(full_status["experiments"]) >= 1
    assert full_status["recommendation"] is not None

    rec = full_status["recommendation"]
    assert rec.get("available") is True
    assert rec.get("model_name") is not None
    assert rec.get("primary_metric") == "r2"
    assert rec.get("reason") is not None and len(rec["reason"]) > 20

    # 5. Verify decisions were recorded
    assert len(full_status["decisions"]) >= 1
    first_decision = full_status["decisions"][0]
    assert first_decision["decision"] in {"initial_model", "stop", "optimize", "try_another_model"}

    # 6. Verify the recommended experiment artifact is accessible
    rec_exp_id = rec.get("experiment_id")
    if rec_exp_id:
        exp_res = client.get(f"/api/experiments/{rec_exp_id}")
        assert exp_res.status_code == 200

    # 7. Cleanup
    client.delete(f"/api/experiments/sessions/{session_id}")
    client.delete(f"/api/datasets/{dataset_id}")


def test_plateau_requires_min_experiments(client):
    """Verify that the decision engine does not plateau-stop after just 2 experiments.

    We run 4 experiments and check that the session ran at least 3 before stopping
    (i.e., the plateau guard required >= 3 scored runs).
    """
    load_res = client.post("/api/datasets/sample/classification/load")
    dataset_id = load_res.json()["id"]

    start_res = client.post(
        "/api/experiments/start",
        json={
            "dataset_id": dataset_id,
            "target_column": "churn",
            "problem_type": "classification",
            "priority": "balanced",
            "max_experiments": 4,
            "interactive": False,
            "auto_start": True,
            "name": "Plateau Guard Test",
        },
    )
    assert start_res.status_code == 200
    session_id = start_res.json()["id"]

    for _ in range(60):
        prog = client.get(f"/api/experiments/sessions/{session_id}/progress").json()
        if prog["status"] in {"completed", "stopped", "failed"}:
            break
        time.sleep(1)

    status = client.get(f"/api/experiments/sessions/{session_id}").json()
    experiments_run = len(status["experiments"])
    stop_code = status["session"].get("stop_reason", "")

    # If stopped by plateau, must have run at least 3 experiments first
    if "plateau" in stop_code.lower():
        assert experiments_run >= 3, (
            f"Plateau fired after only {experiments_run} experiment(s); minimum is 3"
        )

    client.delete(f"/api/experiments/sessions/{session_id}")
    client.delete(f"/api/datasets/{dataset_id}")

"""FastAPI entrypoint for the Autonomous ML Experimentation Agent.

Exposes REST APIs for:
* Dataset management (upload, analysis, sample datasets)
* Autonomous experiment sessions (start, live progress, pause/continue, stop, cancel)
* Model comparison & leaderboards
* Model artifact inspection, download, and interactive live prediction
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from config import settings
from database import get_db, init_db, session_scope
from errors import (
    AgentError,
    ConflictError,
    DatasetValidationError,
    NotFoundError,
    ResourceLimitError,
    TrainingError,
)
from logging_config import get_logger
from models import AgentDecision, Dataset, Experiment, ExperimentSession
from schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    ColumnInfo,
    ComparisonResponse,
    DecisionOut,
    DatasetDetail,
    DatasetPreviewResponse,
    DatasetSummary,
    DatasetUploadResponse,
    ExperimentOut,
    LiveProgress,
    MessageResponse,
    SessionDetail,
    SessionStatusResponse,
    SessionSummary,
    SettingsResponse,
    StartExperimentRequest,
    StopRequest,
)
from services import dataset_analyzer as analyzer
from services import evaluator
from services import experiment_manager as em
from services import model_selector as selector
from services import resource_monitor as rm

logger = get_logger(__name__)


# ------------------------------------------------------------------ lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_db()
    logger.info("Database schema initialized at startup.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Autonomous ML Experimentation Agent API",
    lifespan=lifespan,
)

# ------------------------------------------------------------------ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev and vite previews
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------- error handlers
@app.exception_handler(AgentError)
async def handle_agent_error(request, exc: AgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# --------------------------------------------------------- system endpoints
@app.get("/api/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/settings", response_model=SettingsResponse, tags=["System"])
def get_system_settings() -> SettingsResponse:
    snapshot = rm.resource_snapshot()
    return SettingsResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        max_upload_size_mb=settings.max_upload_size_mb,
        max_rows=settings.max_rows,
        max_columns=settings.max_columns,
        default_max_experiments=settings.default_max_experiments,
        hard_max_experiments=settings.hard_max_experiments,
        max_experiment_seconds=settings.max_experiment_seconds,
        max_session_seconds=settings.max_session_seconds,
        default_priority=settings.default_priority,
        priorities=list(settings.priorities),
        supported_extensions=list(settings.supported_extensions),
        random_state=settings.random_state,
        cv_folds=settings.cv_folds,
        optimization_trials=settings.optimization_trials,
        plateau_patience=settings.plateau_patience,
        targets={
            "target_r2": settings.target_r2,
            "target_f1": settings.target_f1,
            "target_accuracy": settings.target_accuracy,
        },
        resource_snapshot=snapshot,
    )


@app.get("/api/system/resources", tags=["System"])
def get_live_resources() -> Dict[str, Any]:
    return rm.resource_snapshot()


# -------------------------------------------------------- dataset endpoints
@app.post("/api/datasets/upload", response_model=DatasetUploadResponse, tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    original_name = file.filename or "uploaded_dataset.csv"
    ext = Path(original_name).suffix.lower()
    if ext not in settings.supported_extensions:
        raise DatasetValidationError(
            f"Unsupported file extension '{ext}'. Supported formats: {', '.join(settings.supported_extensions)}"
        )

    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id}_{original_name}"
    stored_path = settings.upload_dir / stored_filename

    total_bytes = 0
    with open(stored_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > settings.max_upload_size_mb * 1024 * 1024:
                stored_path.unlink(missing_ok=True)
                raise ResourceLimitError(
                    f"File size exceeds maximum allowed upload limit of {settings.max_upload_size_mb} MB"
                )
            buffer.write(chunk)

    try:
        df = analyzer.load_dataframe(stored_path)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise DatasetValidationError(f"Failed to parse dataset: {exc}")

    violations = rm.check_hard_limits(
        n_rows=len(df),
        n_columns=len(df.columns),
        file_size_mb=total_bytes / (1024 * 1024),
    )
    if violations:
        stored_path.unlink(missing_ok=True)
        raise ResourceLimitError(
            f"Dataset violates hard system resource limits", violations=violations
        )

    columns = list(df.columns)
    column_dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    candidates = analyzer.target_candidates(df)
    suggested_target = candidates[0] if candidates else (columns[-1] if columns else None)

    dataset = Dataset(
        name=Path(original_name).stem.replace("_", " ").title(),
        original_filename=original_name,
        stored_path=str(stored_path.resolve()),
        file_extension=ext,
        file_size_bytes=total_bytes,
        n_rows=len(df),
        n_columns=len(df.columns),
        target_column=suggested_target,
        columns=columns,
        column_dtypes=column_dtypes,
        is_analyzed=False,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    preview_rows = df.head(15).fillna("").to_dict(orient="records")
    preview = DatasetPreviewResponse(
        dataset_id=dataset.id,
        columns=columns,
        dtypes=column_dtypes,
        rows=preview_rows,
        total_rows=len(df),
        preview_rows=len(preview_rows),
        target_column=suggested_target,
        target_candidates=candidates,
    )

    warnings = []
    if len(df) < 50:
        warnings.append("Dataset has fewer than 50 rows; generalization metrics may be volatile.")

    return DatasetUploadResponse(
        dataset=DatasetSummary.model_validate(dataset),
        preview=preview,
        warnings=warnings,
    )


@app.get("/api/datasets", response_model=List[DatasetSummary], tags=["Datasets"])
def list_datasets(db: Session = Depends(get_db)) -> List[DatasetSummary]:
    datasets = db.execute(select(Dataset).order_by(desc(Dataset.created_at))).scalars().all()
    return [DatasetSummary.model_validate(d) for d in datasets]


@app.get("/api/datasets/{dataset_id}", response_model=DatasetDetail, tags=["Datasets"])
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> DatasetDetail:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' not found.")
    session_count = (
        db.query(ExperimentSession).filter(ExperimentSession.dataset_id == dataset_id).count()
    )
    detail = DatasetDetail.model_validate(dataset)
    detail.session_count = session_count
    return detail


@app.get(
    "/api/datasets/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    tags=["Datasets"],
)
def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
) -> DatasetPreviewResponse:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' not found.")

    df = analyzer.load_dataframe(dataset.stored_path)
    preview_df = df.head(limit).fillna("")
    candidates = analyzer.target_candidates(df)

    return DatasetPreviewResponse(
        dataset_id=dataset.id,
        columns=list(df.columns),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        rows=preview_df.to_dict(orient="records"),
        total_rows=len(df),
        preview_rows=len(preview_df),
        target_column=dataset.target_column,
        target_candidates=candidates,
    )


@app.delete("/api/datasets/{dataset_id}", response_model=MessageResponse, tags=["Datasets"])
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)) -> MessageResponse:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' not found.")

    if dataset.stored_path:
        try:
            p = Path(dataset.stored_path)
            if p.exists():
                p.unlink()
        except Exception as exc:
            logger.warning("Failed to delete dataset file %s: %s", dataset.stored_path, exc)

    db.delete(dataset)
    db.commit()
    return MessageResponse(ok=True, message=f"Dataset '{dataset_id}' deleted successfully.")


@app.post(
    "/api/datasets/{dataset_id}/analyze",
    response_model=AnalysisResponse,
    tags=["Datasets"],
)
def analyze_dataset(
    dataset_id: str,
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' not found.")

    if request.target_column not in (dataset.columns or []):
        raise DatasetValidationError(
            f"Target column '{request.target_column}' does not exist in dataset columns."
        )

    df = analyzer.load_dataframe(dataset.stored_path)
    file_size_mb = (dataset.file_size_bytes or 0) / (1024 * 1024)

    analysis = analyzer.analyze_dataframe(
        df=df,
        target=request.target_column,
        problem_type_override=request.problem_type,
        file_size_mb=file_size_mb,
    )

    dataset.target_column = request.target_column
    dataset.problem_type = analysis["problem_type"]
    dataset.problem_type_source = analysis["problem_type_source"]
    dataset.analysis = analysis
    dataset.is_analyzed = True
    dataset.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(dataset)

    return AnalysisResponse(
        dataset=DatasetSummary.model_validate(dataset),
        analysis=analysis,
    )


@app.post(
    "/api/datasets/sample/{sample_name}/load",
    response_model=DatasetDetail,
    tags=["Datasets"],
)
def load_sample_dataset(sample_name: str, db: Session = Depends(get_db)) -> DatasetDetail:
    """Load and auto-analyze one of the pre-bundled sample datasets."""
    sample_mapping = {
        "linear_regression": ("sample_linear_regression.csv", "price", "Sample Linear Regression"),
        "nonlinear_regression": (
            "sample_nonlinear_regression.csv",
            "energy_demand_mw",
            "Sample Nonlinear Regression",
        ),
        "classification": ("sample_classification.csv", "churn", "Sample Churn Classification"),
    }
    if sample_name not in sample_mapping:
        raise NotFoundError(
            f"Unknown sample '{sample_name}'. Choose from: {', '.join(sample_mapping.keys())}"
        )

    filename, target, display_name = sample_mapping[sample_name]
    sample_path = settings.base_dir / "sample_data" / filename
    if not sample_path.exists():
        raise NotFoundError(f"Sample file '{filename}' was not found in sample_data directory.")

    # Copy to uploads directory to maintain consistent management
    copy_id = str(uuid.uuid4())
    dest_path = settings.upload_dir / f"{copy_id}_{filename}"
    shutil.copyfile(sample_path, dest_path)

    df = analyzer.load_dataframe(dest_path)
    file_size = dest_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    analysis = analyzer.analyze_dataframe(
        df=df,
        target=target,
        file_size_mb=file_size_mb,
    )

    dataset = Dataset(
        name=display_name,
        original_filename=filename,
        stored_path=str(dest_path.resolve()),
        file_extension=".csv",
        file_size_bytes=file_size,
        n_rows=len(df),
        n_columns=len(df.columns),
        target_column=target,
        problem_type=analysis["problem_type"],
        problem_type_source="agent",
        columns=list(df.columns),
        column_dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        analysis=analysis,
        is_analyzed=True,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    detail = DatasetDetail.model_validate(dataset)
    detail.session_count = 0
    return detail


# --------------------------------------------------- experiment session endpoints
@app.post(
    "/api/experiments/start",
    response_model=SessionDetail,
    tags=["Experiments"],
)
def start_experiment_session(
    request: StartExperimentRequest,
    db: Session = Depends(get_db),
) -> SessionDetail:
    session = em.create_session(
        db=db,
        dataset_id=request.dataset_id,
        target_column=request.target_column,
        problem_type=request.problem_type,
        priority=request.priority,
        max_experiments=request.max_experiments,
        interactive=request.interactive,
        auto_start=request.auto_start,
        name=request.name,
    )
    db.commit()
    db.refresh(session)

    if request.auto_start:
        em.start_session_thread(session.id)

    return _session_to_detail(session, db)


@app.get(
    "/api/experiments/sessions",
    response_model=List[SessionSummary],
    tags=["Experiments"],
)
def list_sessions(db: Session = Depends(get_db)) -> List[SessionSummary]:
    sessions = (
        db.execute(select(ExperimentSession).order_by(desc(ExperimentSession.created_at)))
        .scalars()
        .all()
    )
    results = []
    for s in sessions:
        results.append(_session_to_summary(s, db))
    return results


@app.get(
    "/api/experiments/sessions/{session_id}",
    response_model=SessionStatusResponse,
    tags=["Experiments"],
)
def get_session_status(session_id: str, db: Session = Depends(get_db)) -> SessionStatusResponse:
    session = db.get(ExperimentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session '{session_id}' not found.")

    progress_raw = em.get_progress(session_id)
    progress = LiveProgress(
        status=progress_raw.get("status", session.status),
        stage=progress_raw.get("stage", "idle"),
        progress=float(progress_raw.get("progress") or 0.0),
        current_experiment_number=progress_raw.get("current_experiment_number"),
        current_model=progress_raw.get("current_model"),
        current_experiment_id=progress_raw.get("current_experiment_id"),
        message=progress_raw.get("message"),
        eta_seconds=progress_raw.get("eta_seconds"),
        elapsed_seconds=progress_raw.get("elapsed_seconds"),
        remaining_budget=progress_raw.get("remaining_budget"),
        updated_at=progress_raw.get("updated_at"),
        log=list(progress_raw.get("log") or []),
    )

    experiments = (
        db.execute(
            select(Experiment)
            .where(Experiment.session_id == session_id)
            .order_by(Experiment.experiment_number)
        )
        .scalars()
        .all()
    )

    decisions = (
        db.execute(
            select(AgentDecision)
            .where(AgentDecision.session_id == session_id)
            .order_by(AgentDecision.step)
        )
        .scalars()
        .all()
    )

    exp_rows = [
        {
            "id": e.id,
            "experiment_number": e.experiment_number,
            "model_key": e.model_key,
            "model_name": e.model_name,
            "kind": e.kind,
            "status": e.status,
            "hyperparameters": e.hyperparameters or {},
            "metrics": e.metrics,
            "test_metrics": e.test_metrics,
            "cv_metrics": e.cv_metrics,
            "training_duration_seconds": e.training_duration_seconds,
            "estimated_duration_seconds": e.estimated_duration_seconds,
            "resource_usage": e.resource_usage,
            "primary_metric": evaluator.primary_metric_value(e.metrics, session.problem_type),
        }
        for e in experiments
    ]

    comparison = None
    if experiments:
        comparison = em.build_comparison(session, exp_rows, session.recommendation)

    return SessionStatusResponse(
        session=_session_to_detail(session, db),
        progress=progress,
        experiments=[ExperimentOut.model_validate(e) for e in experiments],
        decisions=[DecisionOut.model_validate(d) for d in decisions],
        comparison=comparison,
        recommendation=session.recommendation,
    )


@app.get(
    "/api/experiments/sessions/{session_id}/progress",
    response_model=LiveProgress,
    tags=["Experiments"],
)
def get_session_live_progress(session_id: str) -> LiveProgress:
    p = em.get_progress(session_id)
    return LiveProgress(
        status=p.get("status", "unknown"),
        stage=p.get("stage", "idle"),
        progress=float(p.get("progress") or 0.0),
        current_experiment_number=p.get("current_experiment_number"),
        current_model=p.get("current_model"),
        current_experiment_id=p.get("current_experiment_id"),
        message=p.get("message"),
        eta_seconds=p.get("eta_seconds"),
        elapsed_seconds=p.get("elapsed_seconds"),
        remaining_budget=p.get("remaining_budget"),
        updated_at=p.get("updated_at"),
        log=list(p.get("log") or []),
    )


@app.post(
    "/api/experiments/sessions/{session_id}/continue",
    response_model=MessageResponse,
    tags=["Experiments"],
)
def continue_session(session_id: str) -> MessageResponse:
    em.request_continue(session_id)
    return MessageResponse(ok=True, message=f"Session '{session_id}' resume signal sent.")


@app.post(
    "/api/experiments/sessions/{session_id}/stop",
    response_model=MessageResponse,
    tags=["Experiments"],
)
def stop_session(session_id: str, req: StopRequest) -> MessageResponse:
    em.request_stop(session_id, mode=req.mode)
    return MessageResponse(
        ok=True, message=f"Session '{session_id}' stop signal sent with mode '{req.mode}'."
    )


@app.post(
    "/api/experiments/sessions/{session_id}/cancel",
    response_model=MessageResponse,
    tags=["Experiments"],
)
def cancel_current_experiment(session_id: str) -> MessageResponse:
    em.request_cancel_current(session_id)
    return MessageResponse(
        ok=True, message=f"Cancel signal sent for current running experiment in session '{session_id}'."
    )


@app.post(
    "/api/experiments/sessions/{session_id}/resume",
    response_model=MessageResponse,
    tags=["Experiments"],
)
def resume_session(session_id: str, db: Session = Depends(get_db)) -> MessageResponse:
    session = db.get(ExperimentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session '{session_id}' not found.")

    if session.status == "running":
        raise ConflictError(f"Session '{session_id}' is already running.")

    session.status = "running"
    db.commit()

    em.request_continue(session_id)
    return MessageResponse(ok=True, message=f"Session '{session_id}' resumed.")


@app.delete(
    "/api/experiments/sessions/{session_id}",
    response_model=MessageResponse,
    tags=["Experiments"],
)
def delete_session(session_id: str, db: Session = Depends(get_db)) -> MessageResponse:
    session = db.get(ExperimentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session '{session_id}' not found.")

    try:
        em.request_stop(session_id, mode="save")
    except Exception:
        pass

    experiments = (
        db.execute(select(Experiment).where(Experiment.session_id == session_id)).scalars().all()
    )
    for exp in experiments:
        if exp.artifact_path:
            try:
                p = Path(exp.artifact_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    db.delete(session)
    db.commit()
    return MessageResponse(ok=True, message=f"Session '{session_id}' deleted successfully.")


@app.get(
    "/api/experiments/sessions/{session_id}/export/csv",
    tags=["Experiments"],
)
def export_session_results_csv(session_id: str, db: Session = Depends(get_db)):
    """Export all completed experiment results and leaderboard for a session as CSV."""
    session = db.get(ExperimentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session '{session_id}' not found.")

    experiments = (
        db.execute(
            select(Experiment)
            .where(Experiment.session_id == session_id)
            .order_by(Experiment.experiment_number)
        )
        .scalars()
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "experiment_number",
        "model_name",
        "model_key",
        "kind",
        "status",
        "problem_type",
        "primary_metric_name",
        "primary_metric_value",
        "test_metrics",
        "validation_metrics",
        "training_duration_seconds",
        "peak_memory_mb",
        "cpu_seconds",
        "is_recommended",
        "hyperparameters",
        "created_at",
    ])

    rec_id = session.recommendation.get("experiment_id") if session.recommendation else None
    primary_metric_name = "accuracy" if session.problem_type == "classification" else "r2"

    for e in experiments:
        primary_val = evaluator.primary_metric_value(e.metrics, session.problem_type)
        is_rec = "YES" if (rec_id and e.id == rec_id) else "NO"
        res = e.resource_usage or {}
        writer.writerow([
            e.experiment_number,
            e.model_name,
            e.model_key,
            e.kind,
            e.status,
            session.problem_type,
            primary_metric_name,
            round(primary_val, 4) if primary_val is not None else "",
            json.dumps(e.test_metrics or {}),
            json.dumps(e.metrics or {}),
            round(e.training_duration_seconds, 3) if e.training_duration_seconds else "",
            res.get("peak_memory_mb", ""),
            res.get("cpu_seconds", ""),
            is_rec,
            json.dumps(e.hyperparameters or {}),
            e.created_at.isoformat() if e.created_at else "",
        ])

    csv_data = output.getvalue()
    safe_name = (session.name or "session").replace(" ", "_").lower()
    filename = f"{safe_name}_{session_id[:8]}_results.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get(
    "/api/experiments/sessions/{session_id}/comparison",
    response_model=ComparisonResponse,
    tags=["Experiments"],
)
def get_session_comparison(session_id: str, db: Session = Depends(get_db)) -> ComparisonResponse:
    session = db.get(ExperimentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session '{session_id}' not found.")

    experiments = (
        db.execute(
            select(Experiment)
            .where(Experiment.session_id == session_id)
            .order_by(Experiment.experiment_number)
        )
        .scalars()
        .all()
    )

    exp_rows = [
        {
            "id": e.id,
            "experiment_number": e.experiment_number,
            "model_key": e.model_key,
            "model_name": e.model_name,
            "kind": e.kind,
            "status": e.status,
            "hyperparameters": e.hyperparameters or {},
            "metrics": e.metrics,
            "test_metrics": e.test_metrics,
            "cv_metrics": e.cv_metrics,
            "training_duration_seconds": e.training_duration_seconds,
            "estimated_duration_seconds": e.estimated_duration_seconds,
            "resource_usage": e.resource_usage,
            "error_message": e.error_message,
            "primary_metric": evaluator.primary_metric_value(e.metrics, session.problem_type),
        }
        for e in experiments
    ]

    comparison_data = em.build_comparison(session, exp_rows, session.recommendation)
    return ComparisonResponse(
        session=_session_to_detail(session, db),
        problem_type=session.problem_type,
        primary_metric=comparison_data["primary_metric"],
        rows=comparison_data["rows"],
        charts=comparison_data["charts"],
        recommendation=session.recommendation,
        best_experiment_id=comparison_data.get("best_experiment_id"),
    )


# --------------------------------------------- single experiment inspection & inference
@app.get(
    "/api/experiments/{experiment_id}",
    response_model=ExperimentOut,
    tags=["Experiments"],
)
def get_experiment_detail(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentOut:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")
    return ExperimentOut.model_validate(exp)


@app.get("/api/experiments/{experiment_id}/download", tags=["Experiments"])
def download_model_artifact(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")
    if not exp.artifact_path or not Path(exp.artifact_path).exists():
        raise NotFoundError(f"Model artifact not found for experiment '{experiment_id}'.")

    filename = Path(exp.artifact_path).name
    return FileResponse(
        path=exp.artifact_path,
        media_type="application/octet-stream",
        filename=filename,
    )


class PredictRequest(BaseModel):
    inputs: List[Dict[str, Any]] = Field(
        ..., description="List of rows to predict with feature key-value pairs"
    )


@app.post("/api/experiments/{experiment_id}/predict", tags=["Experiments"])
def predict_with_model(
    experiment_id: str,
    req: PredictRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Execute live model inference using the trained pipeline artifact."""
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")
    if not exp.artifact_path or not Path(exp.artifact_path).exists():
        raise NotFoundError(
            f"No trained pipeline artifact is persisted for experiment '{experiment_id}'."
        )

    try:
        artifact = joblib.load(exp.artifact_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            pipeline = artifact["pipeline"]
        else:
            pipeline = artifact
    except Exception as exc:
        raise TrainingError(f"Failed to load serialized pipeline: {exc}")

    if not req.inputs:
        raise DatasetValidationError("inputs list must not be empty.")

    try:
        df = pd.DataFrame(req.inputs)
        preds = pipeline.predict(df)
        preds_list = preds.tolist()

        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            try:
                prob = pipeline.predict_proba(df)
                probabilities = prob.tolist()
            except Exception:
                probabilities = None

        return {
            "experiment_id": experiment_id,
            "model_name": exp.model_name,
            "n_predictions": len(preds_list),
            "predictions": preds_list,
            "probabilities": probabilities,
        }
    except Exception as exc:
        raise TrainingError(f"Prediction failed: {exc}")


# ------------------------------------------------------------- internal helpers
def _session_to_summary(session: ExperimentSession, db: Session) -> SessionSummary:
    dataset = db.get(Dataset, session.dataset_id)
    best_name = None
    best_metric = None
    if session.best_experiment_id:
        best_exp = db.get(Experiment, session.best_experiment_id)
        if best_exp:
            best_name = best_exp.model_name
            best_metric = evaluator.primary_metric_value(best_exp.metrics, session.problem_type)

    summary = SessionSummary.model_validate(session)
    summary.dataset_name = dataset.name if dataset else None
    summary.best_model_name = best_name
    summary.best_primary_metric = best_metric
    return summary


def _session_to_detail(session: ExperimentSession, db: Session) -> SessionDetail:
    dataset = db.get(Dataset, session.dataset_id)
    best_name = None
    best_metric = None
    if session.best_experiment_id:
        best_exp = db.get(Experiment, session.best_experiment_id)
        if best_exp:
            best_name = best_exp.model_name
            best_metric = evaluator.primary_metric_value(best_exp.metrics, session.problem_type)

    detail = SessionDetail.model_validate(session)
    detail.dataset_name = dataset.name if dataset else None
    detail.best_model_name = best_name
    detail.best_primary_metric = best_metric
    return detail


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

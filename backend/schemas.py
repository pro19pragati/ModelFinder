"""Pydantic v2 schemas used for request validation and API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------- datasets
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    kind: str  # numeric | categorical | datetime | boolean | text
    missing: int = 0
    unique: int = 0


class DatasetSummary(ORMModel):
    id: str
    name: str
    original_filename: str
    file_extension: str
    file_size_bytes: int = 0
    n_rows: int = 0
    n_columns: int = 0
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    problem_type_source: Optional[str] = None
    columns: List[Any] = Field(default_factory=list)
    is_analyzed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetDetail(DatasetSummary):
    analysis: Optional[Dict[str, Any]] = None
    session_count: int = 0


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: List[str]
    dtypes: Dict[str, str]
    rows: List[Dict[str, Any]]
    total_rows: int
    preview_rows: int
    target_column: Optional[str] = None
    target_candidates: List[str] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    dataset: DatasetSummary
    preview: DatasetPreviewResponse
    warnings: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    target_column: str
    problem_type: Optional[str] = None  # optional user correction
    force: bool = False

    @field_validator("target_column")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("target_column must not be empty")
        return value

    @field_validator("problem_type")
    @classmethod
    def _valid_problem(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().lower()
        if value not in settings.supported_problem_types:
            raise ValueError(
                "problem_type must be one of " + ", ".join(settings.supported_problem_types)
            )
        return value


class AnalysisResponse(BaseModel):
    dataset: DatasetSummary
    analysis: Dict[str, Any]


# ------------------------------------------------------------------ experiments
class StartExperimentRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    priority: str = settings.default_priority
    max_experiments: Optional[int] = None
    interactive: Optional[bool] = None
    auto_start: bool = True
    name: Optional[str] = None

    @field_validator("priority")
    @classmethod
    def _valid_priority(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if value not in settings.priorities:
            raise ValueError("priority must be one of " + ", ".join(settings.priorities))
        return value

    @field_validator("max_experiments")
    @classmethod
    def _valid_budget(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 1:
            raise ValueError("max_experiments must be >= 1")
        if value > settings.hard_max_experiments:
            raise ValueError(
                f"max_experiments cannot exceed the hard limit of {settings.hard_max_experiments}"
            )
        return value


class StopRequest(BaseModel):
    mode: str = "compare"  # compare | save

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if value not in {"compare", "save"}:
            raise ValueError("mode must be 'compare' or 'save'")
        return value


class ExperimentOut(ORMModel):
    id: str
    session_id: str
    dataset_id: str
    experiment_number: int
    model_key: str
    model_name: str
    kind: str
    status: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    cv_metrics: Optional[Dict[str, Any]] = None
    trials: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    training_duration_seconds: Optional[float] = None
    estimated_duration_seconds: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None
    dataset_rows: Optional[int] = None
    dataset_features: Optional[int] = None
    notes: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class DecisionOut(ORMModel):
    id: str
    session_id: str
    experiment_id: Optional[str] = None
    step: int
    decision: str
    headline: Optional[str] = None
    reasoning: str
    details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class SessionSummary(ORMModel):
    id: str
    dataset_id: str
    name: Optional[str] = None
    target_column: str
    problem_type: str
    priority: str
    status: str
    interactive: bool = False
    saved: bool = False
    max_experiments: int = 10
    completed_experiments: int = 0
    current_experiment_number: int = 0
    dataset_name: Optional[str] = None
    best_model_name: Optional[str] = None
    best_primary_metric: Optional[float] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    stop_reason: Optional[str] = None


class SessionDetail(SessionSummary):
    plan: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    resource_snapshot: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    final_metrics: Optional[Dict[str, Any]] = None
    best_experiment_id: Optional[str] = None


class LiveProgress(BaseModel):
    status: str
    stage: str
    progress: float = 0.0
    current_experiment_number: Optional[int] = None
    current_model: Optional[str] = None
    current_experiment_id: Optional[str] = None
    message: Optional[str] = None
    eta_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    remaining_budget: Optional[int] = None
    updated_at: Optional[str] = None
    log: List[Dict[str, Any]] = Field(default_factory=list)


class SessionStatusResponse(BaseModel):
    session: SessionDetail
    progress: LiveProgress
    experiments: List[ExperimentOut] = Field(default_factory=list)
    decisions: List[DecisionOut] = Field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None


class ComparisonResponse(BaseModel):
    session: SessionDetail
    problem_type: str
    primary_metric: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    charts: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[Dict[str, Any]] = None
    best_experiment_id: Optional[str] = None


class SettingsResponse(BaseModel):
    app_name: str
    app_version: str
    max_upload_size_mb: float
    max_rows: int
    max_columns: int
    default_max_experiments: int
    hard_max_experiments: int
    max_experiment_seconds: float
    max_session_seconds: float
    default_priority: str
    priorities: List[str]
    supported_extensions: List[str]
    random_state: int
    cv_folds: int
    optimization_trials: int
    plateau_patience: int
    targets: Dict[str, float]
    resource_snapshot: Dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    ok: bool = True
    message: str

"""SQLAlchemy ORM models: datasets, sessions (runs), experiments, decisions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    """An uploaded dataset plus the cached analysis produced by the agent."""

    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(Text, nullable=False)
    file_extension = Column(String(16), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    n_rows = Column(Integer, default=0)
    n_columns = Column(Integer, default=0)
    target_column = Column(String(255), nullable=True)
    problem_type = Column(String(32), nullable=True)
    problem_type_source = Column(String(32), nullable=True)  # agent | user
    columns = Column(JSON, default=list)
    column_dtypes = Column(JSON, default=dict)
    analysis = Column(JSON, nullable=True)
    is_analyzed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    sessions = relationship(
        "ExperimentSession", back_populates="dataset", cascade="all, delete-orphan"
    )
    experiments = relationship(
        "Experiment", back_populates="dataset", cascade="all, delete-orphan"
    )


class ExperimentSession(Base):
    """A single autonomous experimentation run over one dataset."""

    __tablename__ = "experiment_sessions"

    id = Column(String(36), primary_key=True, default=new_id)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    name = Column(String(255), nullable=True)

    target_column = Column(String(255), nullable=False)
    problem_type = Column(String(32), nullable=False)
    priority = Column(String(32), default="balanced", nullable=False)
    status = Column(String(32), default="created", nullable=False)
    # created | running | waiting | paused | stopped | completed | failed

    interactive = Column(Boolean, default=False)
    auto_start = Column(Boolean, default=True)
    saved = Column(Boolean, default=False)
    resumed_from = Column(String(36), nullable=True)

    max_experiments = Column(Integer, default=10)
    completed_experiments = Column(Integer, default=0)
    current_experiment_number = Column(Integer, default=0)

    plan = Column(JSON, nullable=True)
    budget = Column(JSON, nullable=True)
    resource_snapshot = Column(JSON, nullable=True)
    recommendation = Column(JSON, nullable=True)
    final_metrics = Column(JSON, nullable=True)
    best_experiment_id = Column(String(36), nullable=True)
    stop_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    dataset = relationship("Dataset", back_populates="sessions")
    experiments = relationship(
        "Experiment",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Experiment.experiment_number",
    )
    decisions = relationship(
        "AgentDecision",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AgentDecision.step",
    )


class Experiment(Base):
    """One training + evaluation run (also the unit of the experiment budget)."""

    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=new_id)
    session_id = Column(String(36), ForeignKey("experiment_sessions.id"), nullable=False)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    experiment_number = Column(Integer, nullable=False)

    model_key = Column(String(64), nullable=False)
    model_name = Column(String(128), nullable=False)
    kind = Column(String(32), default="initial")  # initial|optimization|alternative
    status = Column(String(32), default="running")  # running|completed|failed|cancelled

    hyperparameters = Column(JSON, default=dict)
    metrics = Column(JSON, nullable=True)        # validation metrics (drive decisions)
    test_metrics = Column(JSON, nullable=True)   # held-out test metrics (final model)
    cv_metrics = Column(JSON, nullable=True)
    trials = Column(JSON, nullable=True)         # optuna trial summary
    feature_importance = Column(JSON, nullable=True)

    training_duration_seconds = Column(Float, nullable=True)
    estimated_duration_seconds = Column(Float, nullable=True)
    resource_usage = Column(JSON, nullable=True)  # cpu_seconds / peak_memory_mb
    dataset_rows = Column(Integer, nullable=True)
    dataset_features = Column(Integer, nullable=True)
    artifact_path = Column(Text, nullable=True)

    notes = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("ExperimentSession", back_populates="experiments")
    dataset = relationship("Dataset", back_populates="experiments")
    decisions = relationship("AgentDecision", back_populates="experiment")


class AgentDecision(Base):
    """The agent's reasoning, recorded after every completed experiment."""

    __tablename__ = "agent_decisions"

    id = Column(String(36), primary_key=True, default=new_id)
    session_id = Column(String(36), ForeignKey("experiment_sessions.id"), nullable=False)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=True)
    step = Column(Integer, default=0)
    decision = Column(String(32), nullable=False)  # stop | optimize | try_another_model
    headline = Column(String(255), nullable=True)
    reasoning = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    session = relationship("ExperimentSession", back_populates="decisions")
    experiment = relationship("Experiment", back_populates="decisions")

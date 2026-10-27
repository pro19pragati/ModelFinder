"""Central configuration for the Autonomous ML Experimentation Agent backend.

All tunables live here so that the resource limits, experimentation budget and
stopping criteria are configurable in one place (env vars override defaults).
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("AML_STORAGE_DIR", str(BASE_DIR / "storage")))
UPLOAD_DIR = STORAGE_DIR / "uploads"
ARTIFACT_DIR = STORAGE_DIR / "artifacts"


class Settings:
    """Runtime settings (instantiated once as ``settings``)."""

    app_name = "Autonomous ML Experimentation Agent"
    app_version = "1.0.0"
    api_prefix = "/api"

    # ---------------------------------------------------------------- storage
    base_dir = BASE_DIR
    storage_dir = STORAGE_DIR
    upload_dir = UPLOAD_DIR
    artifact_dir = ARTIFACT_DIR
    database_url = os.getenv(
        "AML_DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'experiments.db').as_posix()}"
    )

    # --------------------------------------------------------- hard limits
    # Upload / dataset guards. These are *hard* resource limits: the agent
    # refuses datasets that exceed them instead of silently degrading.
    max_upload_size_mb = _env_float("AML_MAX_UPLOAD_MB", 60.0)
    max_rows = _env_int("AML_MAX_ROWS", 200_000)
    max_columns = _env_int("AML_MAX_COLUMNS", 300)
    max_categorical_cardinality = _env_int("AML_MAX_CARDINALITY", 200)
    max_onehot_categories = _env_int("AML_MAX_ONEHOT", 25)
    supported_extensions = (".csv", ".xlsx", ".xlsm", ".xls")
    supported_problem_types = ("classification", "regression")

    # ------------------------------------------------------- experiment budget
    default_max_experiments = _env_int("AML_MAX_EXPERIMENTS", 10)
    hard_max_experiments = _env_int("AML_HARD_MAX_EXPERIMENTS", 25)
    default_priority = os.getenv("AML_DEFAULT_PRIORITY", "balanced")
    priorities = ("performance", "speed", "balanced")

    # Wall-clock guards for a single experiment / a whole session.
    max_experiment_seconds = _env_float("AML_MAX_EXPERIMENT_SECONDS", 300.0)
    max_session_seconds = _env_float("AML_MAX_SESSION_SECONDS", 1800.0)
    min_seconds_per_experiment = 0.20

    # ------------------------------------------------------------ training
    random_state = 42
    test_size = 0.2           # held out until the very end
    validation_size = 0.2     # of the remaining data, used for model selection
    cv_folds = _env_int("AML_CV_FOLDS", 3)
    analysis_sample_rows = 3000
    linearity_sample_rows = 2000
    pilot_sample_rows = 400
    optimization_trials = _env_int("AML_OPTIMIZATION_TRIALS", 12)
    optimization_min_trials = _env_int("AML_MIN_OPTIMIZATION_TRIALS", 6)
    dense_row_threshold = 60_000   # above this one-hot output stays sparse

    # ----------------------------------------------- agent stopping criteria
    # Minimum relative improvement that counts as "real progress".
    min_improvement_epsilon = 0.01
    # Plateau: this many consecutive experiments without progress -> stop.
    plateau_patience = 2
    # "Good enough" targets used to decide that more work is unnecessary.
    target_r2 = 0.92
    target_f1 = 0.90
    target_accuracy = 0.92
    # Below this the agent keeps looking for a different model family.
    weak_r2 = 0.55
    weak_f1 = 0.55
    weak_accuracy = 0.70

    # ------------------------------------------------------------- runtime
    poll_interval_seconds = 1.0
    interactive_default = _env_bool("AML_INTERACTIVE", False)
    cors_origins = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )

    def ensure_directories(self) -> None:
        for path in (self.storage_dir, self.upload_dir, self.artifact_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

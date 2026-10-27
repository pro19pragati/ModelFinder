"""Training-time estimation and experiment-budget allocation.

The user never has to specify a training duration or an experiment count. The
agent derives both from:

  * dataset size (rows / encoded width) and missing-value ratio,
  * the model complexity of the candidate algorithms,
  * *measured* machine resources (CPU count, available RAM),
  * a small **pilot fit** on a sub-sample that calibrates the estimate,
  * actual training times observed during the session (refined after every
    completed experiment).

Every value produced here is explicitly an *estimate*; only the values produced
by ``ResourceTracker`` are measured.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from logging_config import get_logger
from services import model_selector as selector
from services import preprocessing

logger = get_logger(__name__)


@dataclass
class TimeEstimate:
    """Estimated wall-clock duration of a single experiment."""

    expected_seconds: float
    low_seconds: float
    high_seconds: float
    method: str
    basis: Dict[str, Any] = field(default_factory=dict)
    measured_pilot_seconds: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expected_seconds": round(self.expected_seconds, 3),
            "low_seconds": round(self.low_seconds, 3),
            "high_seconds": round(self.high_seconds, 3),
            "range_label": f"{self.low_seconds:.1f}s - {self.high_seconds:.1f}s",
            "method": self.method,
            "basis": self.basis,
            "measured_pilot_seconds": (
                round(self.measured_pilot_seconds, 4)
                if self.measured_pilot_seconds is not None
                else None
            ),
            "is_estimate": True,
        }


class TimeEstimator:
    """Calibrating estimator for training duration (seconds per 1000 rows).

    The rate is initialised from a pilot fit and then refined with the actual
    durations of completed experiments using an exponential moving average.
    """

    def __init__(self, n_rows: int, effective_width: int, cpu_count: int) -> None:
        self.n_rows = max(1, int(n_rows))
        self.effective_width = max(1, int(effective_width))
        self.cpu_count = max(1, int(cpu_count))
        self.seconds_per_1000_rows: Optional[float] = None
        self.observations: List[Dict[str, Any]] = []
        self.pilot_seconds: Optional[float] = None

    # ------------------------------------------------------------------ pilot
    def run_pilot(self, split: Any, model_key: str) -> Optional[float]:
        """Fit a cheap linear model on a small sub-sample to calibrate timings."""
        try:
            sample_rows = min(settings.pilot_sample_rows, len(split.X_train))
            if sample_rows < 40:
                return None
            task = selector.get_spec(model_key).task
            pilot_key = "ridge_regression" if task == "regression" else "logistic_regression"
            plan = preprocessing.plan_preprocessing(
                numeric_features=split.numeric_features,
                categorical_features=split.categorical_features,
                model_needs_scaling=True,
                n_rows=sample_rows,
            )
            pipeline = preprocessing.build_pipeline(
                selector.get_spec(pilot_key).builder({}), plan
            )
            started = time.perf_counter()
            pipeline.fit(split.X_train.iloc[:sample_rows], split.y_train[:sample_rows])
            elapsed = time.perf_counter() - started
            self.pilot_seconds = elapsed
            self.seconds_per_1000_rows = max(1e-4, elapsed / sample_rows * 1000.0)
            logger.info(
                "Pilot fit on %s rows took %.4fs (%.5fs per 1000 rows)",
                sample_rows,
                elapsed,
                self.seconds_per_1000_rows,
            )
            return elapsed
        except Exception as exc:  # pragma: no cover - pilot must never break a session
            logger.warning("Pilot fit failed, falling back to heuristics: %s", exc)
            return None

    # ------------------------------------------------------- observations
    def record(self, model_key: str, seconds: float, rows: int) -> None:
        """Learn from a finished experiment to refine future estimates."""
        if seconds <= 0 or rows <= 0:
            return
        spec = selector.get_spec(model_key)
        observed_rate = seconds / rows * 1000.0 / max(0.2, spec.complexity)
        self.observations.append(
            {
                "model_key": model_key,
                "seconds": round(seconds, 4),
                "rows": int(rows),
                "rate_per_1000_rows": round(observed_rate, 5),
                "complexity": spec.complexity,
            }
        )
        if self.seconds_per_1000_rows is None:
            self.seconds_per_1000_rows = observed_rate
        else:
            self.seconds_per_1000_rows = (
                0.6 * self.seconds_per_1000_rows + 0.4 * observed_rate
            )

    # ------------------------------------------------------------ estimates
    def estimate(self, model_key: str, extra_fits: int = 1) -> TimeEstimate:
        """Estimated duration of one experiment with the given model."""
        spec = selector.get_spec(model_key)
        rows = self.n_rows
        width_factor = 1.0 + 0.5 * math.log10(1 + self.effective_width / 10.0)
        cpu_factor = 1.0
        if spec.family == "ensemble" and self.cpu_count > 1:
            cpu_factor = 1.0 / max(1.0, min(4.0, self.cpu_count / 4.0))

        if self.seconds_per_1000_rows is not None:
            base = self.seconds_per_1000_rows * (rows / 1000.0) * spec.complexity
            base *= width_factor * cpu_factor
            method = "pilot_calibrated" if not self.observations else "pilot_plus_measured"
        else:
            heuristic_rate = 0.0018  # seconds per 1000 rows for a plain linear model
            base = heuristic_rate * (rows / 1000.0) * spec.complexity * width_factor * cpu_factor
            method = "heuristic"

        base *= max(1, extra_fits)
        expected = max(settings.min_seconds_per_experiment, base)
        low = max(settings.min_seconds_per_experiment, base * 0.55)
        high = max(low * 1.4, base * 1.8)
        return TimeEstimate(
            expected_seconds=expected,
            low_seconds=low,
            high_seconds=high,
            method=method,
            measured_pilot_seconds=self.pilot_seconds,
            basis={
                "n_rows": rows,
                "effective_width": self.effective_width,
                "cpu_count": self.cpu_count,
                "model_complexity": spec.complexity,
                "seconds_per_1000_rows": (
                    round(self.seconds_per_1000_rows, 6)
                    if self.seconds_per_1000_rows is not None
                    else None
                ),
                "extra_fits": extra_fits,
                "observations": len(self.observations),
            },
        )

    def refine_from_history(self, experiments: List[Dict[str, Any]]) -> None:
        """Re-seed the rate from completed experiments (used when resuming)."""
        for item in experiments:
            seconds = item.get("training_duration_seconds")
            rows = item.get("dataset_rows")
            model_key = item.get("model_key")
            if seconds and rows and model_key:
                self.record(model_key, float(seconds), int(rows))



# --------------------------------------------------------------- budget
def allocate_budget(
    analysis: Dict[str, Any],
    estimator: TimeEstimator,
    initial_model_key: str,
    resource_snapshot: Dict[str, Any],
    priority: str = "balanced",
    max_experiments_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Decide how many experiments fit in the available budget.

    Three concepts are kept separate (as required by the design):
      * **hard resource limits** - enforced by ``resource_monitor``, never bypassed,
      * **experimentation budget** - how many experiments the agent may run,
      * **optimization priorities** - how performance and speed are weighted.
    """
    profile = analysis.get("profile") or {}
    n_rows = int(profile.get("n_rows") or 0)
    effective_width = int(profile.get("estimated_onehot_width") or 0)
    session_seconds = max(60.0, settings.max_session_seconds)

    initial_estimate = estimator.estimate(initial_model_key)
    optimization_estimate = estimator.estimate(initial_model_key, extra_fits=6)
    per_experiment_seconds = max(
        settings.min_seconds_per_experiment,
        (initial_estimate.expected_seconds + optimization_estimate.expected_seconds) / 2.0,
    )
    affordable = max(1, int(session_seconds // max(1.0, per_experiment_seconds)))
    default_budget = settings.default_max_experiments
    priority_budget = {
        "performance": default_budget,
        "balanced": max(2, int(round(default_budget * 0.8))),
        "speed": max(2, int(round(default_budget * 0.5))),
    }.get(priority, default_budget)

    reasons: List[str] = []
    if max_experiments_override:
        max_experiments = min(int(max_experiments_override), settings.hard_max_experiments)
        reasons.append(
            f"The user requested a maximum of {max_experiments} experiment(s); the hard cap is "
            f"{settings.hard_max_experiments}."
        )
    else:
        max_experiments = min(default_budget, priority_budget, affordable)
        reasons.append(
            f"The configured default budget is {default_budget} experiments and the '{priority}' "
            f"priority profile allows {priority_budget}."
        )
        if affordable < max_experiments:
            reasons.append(
                f"At an estimated {per_experiment_seconds:.1f}s per experiment only {affordable} "
                f"experiment(s) fit into the {session_seconds:.0f}s session window, so the budget "
                "was reduced accordingly."
            )
    max_experiments = max(1, min(max_experiments, settings.hard_max_experiments))

    memory_mb = float(resource_snapshot.get("memory_available_mb") or 0.0)
    reasons.append(
        f"Resources: {resource_snapshot.get('cpu_count')} CPU(s) and {memory_mb:.0f} MB of "
        f"available RAM for {n_rows:,} rows with an encoded width of about {effective_width} "
        "columns."
    )
    if memory_mb and memory_mb < 1_000:
        reasons.append(
            "Available RAM is low, so heavy ensembles may be replaced by histogram-based boosting."
        )
    if priority == "speed":
        reasons.append(
            "The 'speed' priority spends fewer experiments and prefers fast models within a small "
            "accuracy tolerance."
        )
    elif priority == "performance":
        reasons.append(
            "The 'performance' priority allows the full budget and keeps optimising while the "
            "score improves."
        )
    else:
        reasons.append(
            "The 'balanced' priority allows the full budget but stops as soon as the score "
            "plateaus."
        )

    return {
        "max_experiments": int(max_experiments),
        "affordable_experiments": int(affordable),
        "session_seconds": float(session_seconds),
        "per_experiment_seconds_estimate": round(per_experiment_seconds, 3),
        "estimated_total_seconds": round(per_experiment_seconds * max_experiments, 2),
        "estimated_total_label": f"{per_experiment_seconds * max_experiments / 60.0:.1f} min",
        "initial_experiment_estimate": initial_estimate.as_dict(),
        "optimization_experiment_estimate": optimization_estimate.as_dict(),
        "priority": priority,
        "hard_limits": {
            "max_rows": settings.max_rows,
            "max_columns": settings.max_columns,
            "max_upload_size_mb": settings.max_upload_size_mb,
            "max_experiment_seconds": settings.max_experiment_seconds,
            "max_session_seconds": settings.max_session_seconds,
            "hard_max_experiments": settings.hard_max_experiments,
        },
        "reasoning": reasons,
        "stopping_criteria": {
            "target_r2": settings.target_r2,
            "target_f1_weighted": settings.target_f1,
            "target_accuracy": settings.target_accuracy,
            "plateau_patience": settings.plateau_patience,
            "min_improvement": settings.min_improvement_epsilon,
        },
    }


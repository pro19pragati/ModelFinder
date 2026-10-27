"""Session orchestration: planning, running, stopping, resuming and comparing.

Design notes
------------
* Training runs in a **daemon background thread** so the API never blocks, while
  the API only ever polls status.
* Live progress lives in an in-memory registry (fast polling) and every state
  transition is also persisted to SQLite (durable history).
* Cancellation is cooperative: the flag is checked between pipeline stages and
  between Optuna trials, so completed results are always preserved.
* Three user controls are supported: ``continue`` (resume the loop), ``stop``
  with mode ``compare`` or ``save`` (both preserve results and produce a
  recommendation) and ``cancel`` (abort the experiment that is currently
  training and stop the session).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from config import settings
from database import session_scope
from errors import ConflictError, NotFoundError
from logging_config import get_logger
from models import AgentDecision, Dataset, Experiment, ExperimentSession
from services import dataset_analyzer as analyzer
from services import decision_engine
from services import evaluator
from services import experiment_planner
from services import model_selector as selector
from services import optimizer as optuna_optimizer
from services import resource_monitor as rm
from services import time_estimator
from services import trainer

logger = get_logger(__name__)

_runners: Dict[str, "RunnerControl"] = {}
_progress: Dict[str, Dict[str, Any]] = {}
_registry_lock = threading.RLock()
MAX_LOG_ENTRIES = 60
TERMINAL_STATUSES = {"completed", "stopped", "failed"}


@dataclass
class RunnerControl:
    """Cooperative control flags for one session worker."""

    session_id: str
    stop_requested: threading.Event = field(default_factory=threading.Event)
    cancel_current: threading.Event = field(default_factory=threading.Event)
    continue_event: threading.Event = field(default_factory=threading.Event)
    interactive: bool = False
    stop_mode: str = "compare"
    thread: Optional[threading.Thread] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- progress
def _progress_entry(session_id: str) -> Dict[str, Any]:
    if session_id not in _progress:
        _progress[session_id] = {
            "session_id": session_id,
            "status": "created",
            "stage": "queued",
            "progress": 0.0,
            "current_experiment_number": None,
            "current_model": None,
            "current_experiment_id": None,
            "message": "Session created",
            "eta_seconds": None,
            "elapsed_seconds": None,
            "started_monotonic": None,
            "updated_at": _now_iso(),
            "log": deque(maxlen=MAX_LOG_ENTRIES),
        }
    return _progress[session_id]


def update_progress(session_id: str, **fields: Any) -> None:
    """Thread-safe progress update (+ optional log line)."""
    with _registry_lock:
        entry = _progress_entry(session_id)
        message = fields.pop("message", None)
        log_line = fields.pop("log", None)
        entry.update(fields)
        entry["updated_at"] = _now_iso()
        if message:
            entry["message"] = message
        if log_line or message:
            entry["log"].append(
                {
                    "time": _now_iso(),
                    "text": log_line or message,
                    "stage": entry.get("stage"),
                    "level": fields.get("log_level", "info"),
                }
            )
        fields.pop("log_level", None)
        started = entry.get("started_monotonic")
        if started:
            entry["elapsed_seconds"] = round(time.perf_counter() - started, 2)


def get_progress(session_id: str) -> Dict[str, Any]:
    with _registry_lock:
        entry = _progress.get(session_id)
        if entry is None:
            return {
                "status": "unknown",
                "stage": "idle",
                "progress": 0.0,
                "message": None,
                "log": [],
                "remaining_budget": None,
            }
        snapshot = dict(entry)
        snapshot["log"] = list(entry["log"])
        snapshot.pop("started_monotonic", None)
        return snapshot


def is_running(session_id: str) -> bool:
    with _registry_lock:
        control = _runners.get(session_id)
        return bool(control and control.thread and control.thread.is_alive())


def active_session_ids() -> List[str]:
    with _registry_lock:
        return [
            session_id
            for session_id, control in _runners.items()
            if control.thread and control.thread.is_alive()
        ]

# ------------------------------------------------------------- session setup
def create_session(
    db: Any,
    dataset_id: str,
    target_column: Optional[str],
    problem_type: Optional[str],
    priority: str,
    max_experiments: Optional[int],
    interactive: Optional[bool],
    auto_start: bool,
    name: Optional[str] = None,
) -> ExperimentSession:
    """Create an experiment session for an already analysed dataset."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' was not found.")
    if not dataset.is_analyzed or not dataset.analysis:
        raise ConflictError(
            "The dataset has not been analysed yet. Run the analysis (with a target column) before "
            "starting an experiment."
        )
    resolved_target = target_column or dataset.target_column
    if not resolved_target:
        raise ConflictError("No target column was selected for this dataset.")
    if resolved_target not in (dataset.columns or []):
        raise ConflictError(f"Target column '{resolved_target}' does not exist in the dataset.")

    analysis = dataset.analysis or {}
    analysis_problem = analysis.get("problem_type")
    resolved_problem = problem_type or analysis_problem or dataset.problem_type or "regression"
    needs_reanalysis = (dataset.target_column != resolved_target) or (
        problem_type and problem_type != analysis_problem
    )
    if needs_reanalysis:
        frame = analyzer.load_dataframe(dataset.stored_path)
        analysis = analyzer.analyze_dataframe(
            frame,
            resolved_target,
            problem_type_override=problem_type,
            file_size_mb=(dataset.file_size_bytes or 0) / (1024 * 1024),
        )
        dataset.target_column = resolved_target
        dataset.problem_type = analysis["problem_type"]
        dataset.problem_type_source = analysis["problem_type_source"]
        dataset.analysis = analysis

    requested_budget = max_experiments or settings.default_max_experiments
    session = ExperimentSession(
        dataset_id=dataset.id,
        name=name or f"{dataset.name} - autonomous run",
        target_column=resolved_target,
        problem_type=resolved_problem,
        priority=priority,
        status="created",
        interactive=settings.interactive_default if interactive is None else bool(interactive),
        auto_start=bool(auto_start),
        max_experiments=min(requested_budget, settings.hard_max_experiments),
        completed_experiments=0,
        current_experiment_number=0,
        resource_snapshot=analysis.get("resource_snapshot") or rm.resource_snapshot(),
    )
    db.add(session)
    db.flush()
    return session


def start_session_thread(session_id: str) -> None:
    """(Re)start the background worker for a session."""
    with _registry_lock:
        existing = _runners.get(session_id)
        if existing and existing.thread and existing.thread.is_alive():
            raise ConflictError("This session is already running.")
        with session_scope() as db:
            session = db.get(ExperimentSession, session_id)
            if session is None:
                raise NotFoundError(f"Session '{session_id}' was not found.")
            interactive = bool(session.interactive)
        control = RunnerControl(session_id=session_id, interactive=interactive)
        _runners[session_id] = control
        thread = threading.Thread(
            target=_worker,
            args=(session_id,),
            name=f"session-{session_id[:8]}",
            daemon=True,
        )
        control.thread = thread
    update_progress(
        session_id,
        started_monotonic=time.perf_counter(),
        status="running",
        stage="starting",
        message="Autonomous experimentation started",
        log="Worker thread started",
        progress=0.0,
    )
    thread.start()


def request_continue(session_id: str) -> None:
    with _registry_lock:
        control = _runners.get(session_id)
    if control is not None and is_running(session_id):
        control.continue_event.set()
        update_progress(
            session_id,
            status="running",
            stage="resuming",
            message="Continue requested - the agent will run the next experiment",
            log="Continue requested by the user",
        )
        return
    start_session_thread(session_id)


def request_stop(session_id: str, mode: str = "compare") -> None:
    with _registry_lock:
        control = _runners.get(session_id)
        if control is None:
            control = RunnerControl(session_id=session_id)
            _runners[session_id] = control
        control.stop_mode = mode
        control.stop_requested.set()
        control.continue_event.set()  # release a waiting worker so it can stop
    message = (
        "Stop requested - the agent finishes the current experiment, preserves all results and "
        "builds the comparison."
        if mode == "compare"
        else "Stop & Save requested - results are preserved so the session can be resumed later."
    )
    update_progress(
        session_id,
        status="stopping",
        stage="stopping",
        message=message,
        log=f"Stop requested (mode={mode})",
    )


def request_cancel_current(session_id: str) -> None:
    """Abort the experiment that is currently training (results so far are kept)."""
    with _registry_lock:
        control = _runners.get(session_id)
        if control is None:
            control = RunnerControl(session_id=session_id)
            _runners[session_id] = control
        control.cancel_current.set()
        control.stop_requested.set()
        control.stop_mode = "compare"
        control.continue_event.set()
    update_progress(
        session_id,
        status="cancelling",
        stage="cancelling",
        message="Cancelling the running experiment - completed results are preserved.",
        log="Cancel-current requested by the user",
    )


def _cancelled(control: RunnerControl) -> bool:
    return control.cancel_current.is_set() or control.stop_requested.is_set()


def _wait_for_continue(control: RunnerControl, session_id: str) -> bool:
    """Block until the user continues; returns False when the user stopped."""
    control.continue_event.clear()
    update_progress(
        session_id,
        status="waiting",
        stage="waiting_for_user",
        message="Waiting for the user to continue (interactive mode).",
    )
    while not control.continue_event.wait(timeout=0.5):
        if control.stop_requested.is_set():
            return False
    return not control.stop_requested.is_set()


# ------------------------------------------------------------- data helpers
def _experiment_rows(db: Any, session_id: str, problem_type: str) -> List[Dict[str, Any]]:
    """Plain dicts for every experiment in a session (ordered, with metrics)."""
    records = db.execute(
        select(Experiment)
        .where(Experiment.session_id == session_id)
        .order_by(Experiment.experiment_number)
    ).scalars().all()
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "id": record.id,
                "experiment_number": record.experiment_number,
                "model_key": record.model_key,
                "model_name": record.model_name,
                "kind": record.kind,
                "status": record.status,
                "hyperparameters": record.hyperparameters or {},
                "metrics": record.metrics,
                "test_metrics": record.test_metrics,
                "cv_metrics": record.cv_metrics,
                "trials": record.trials,
                "feature_importance": record.feature_importance,
                "training_duration_seconds": record.training_duration_seconds,
                "estimated_duration_seconds": record.estimated_duration_seconds,
                "resource_usage": record.resource_usage,
                "dataset_rows": record.dataset_rows,
                "dataset_features": record.dataset_features,
                "notes": record.notes,
                "error_message": record.error_message,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "finished_at": record.finished_at.isoformat() if record.finished_at else None,
                "primary_metric": evaluator.primary_metric_value(record.metrics, problem_type),
            }
        )
    return rows


def build_comparison(
    session: ExperimentSession,
    experiments: List[Dict[str, Any]],
    recommendation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Comparison table + chart data for the comparison page."""
    problem_type = session.problem_type
    priority = session.priority or "balanced"
    primary = evaluator.primary_metric_name(problem_type)
    ranked = evaluator.rank_experiments(experiments, problem_type, priority)

    rows: List[Dict[str, Any]] = []
    for item in experiments:
        if item.get("status") != "completed":
            rows.append(
                {
                    "experiment_id": item["id"],
                    "experiment_number": item.get("experiment_number"),
                    "model_name": item.get("model_name"),
                    "model_key": item.get("model_key"),
                    "kind": item.get("kind"),
                    "status": item.get("status"),
                    "primary_metric": item.get("primary_metric"),
                    "training_seconds": item.get("training_duration_seconds"),
                    "hyperparameters": item.get("hyperparameters"),
                    "resource_usage": item.get("resource_usage"),
                    "metrics": item.get("metrics"),
                    "rank": None,
                    "is_recommended": False,
                    "error_message": item.get("error_message"),
                }
            )
            continue
        row = {
            "experiment_id": item["id"],
            "experiment_number": item.get("experiment_number"),
            "model_name": item.get("model_name"),
            "model_key": item.get("model_key"),
            "kind": item.get("kind"),
            "status": "completed",
            "primary_metric": item.get("primary_metric"),
            "training_seconds": item.get("training_duration_seconds"),
            "hyperparameters": item.get("hyperparameters"),
            "resource_usage": item.get("resource_usage"),
            "metrics": item.get("metrics"),
            "rank": None,
            "is_recommended": False,
        }
        rows.append(row)

    rank_by_id = {item["id"]: item for item in ranked}
    for row in rows:
        entry = rank_by_id.get(row["experiment_id"])
        if entry:
            row["rank"] = entry["rank"]
            row["selection_score"] = entry["selection_score"]
            row["selection_rationale"] = entry["selection_rationale"]

    recommended_model_key = (recommendation or {}).get("model_key")
    recommended_id = (recommendation or {}).get("experiment_id")
    for row in rows:
        row["is_recommended"] = bool(
            (recommended_id and row["experiment_id"] == recommended_id)
            or (not recommended_id and recommended_model_key and row["model_key"] == recommended_model_key
                and row.get("rank") == 1)
        )

    completed = [item for item in experiments if item.get("status") == "completed"]
    performance = [
        {
            "experiment_number": item.get("experiment_number"),
            "label": f"#{item.get('experiment_number')} {item.get('model_name')}",
            "model_name": item.get("model_name"),
            "model_key": item.get("model_key"),
            "kind": item.get("kind"),
            "primary_metric": item.get("primary_metric"),
        }
        for item in completed
        if item.get("primary_metric") is not None
    ]
    durations = [
        {
            "experiment_number": item.get("experiment_number"),
            "label": f"#{item.get('experiment_number')} {item.get('model_name')}",
            "training_seconds": item.get("training_duration_seconds") or 0.0,
            "estimated_seconds": (item.get("estimated_duration_seconds") or 0.0),
        }
        for item in completed
    ]
    progression: List[Dict[str, Any]] = []
    running_best: Optional[float] = None
    for item in performance:
        value = item["primary_metric"]
        running_best = value if running_best is None else max(running_best, value)
        progression.append(
            {
                "experiment_number": item["experiment_number"],
                "label": item["label"],
                "metric": value,
                "best_so_far": round(running_best, 4),
            }
        )
    resources = [
        {
            "experiment_number": item.get("experiment_number"),
            "label": f"#{item.get('experiment_number')} {item.get('model_name')}",
            "cpu_seconds": (item.get("resource_usage") or {}).get("cpu_seconds"),
            "peak_memory_mb": (item.get("resource_usage") or {}).get("peak_memory_mb"),
        }
        for item in completed
    ]

    return {
        "problem_type": problem_type,
        "primary_metric": primary,
        "secondary_metric": evaluator.SECONDARY_METRIC.get(problem_type),
        "priority": priority,
        "rows": rows,
        "charts": {
            "performance": performance,
            "durations": durations,
            "progression": progression,
            "resources": resources,
            "feature_importance": _feature_importance_chart(experiments, problem_type),
        },
        "recommendation": recommendation,
        "best_experiment_id": (recommendation or {}).get("experiment_id"),
        "chart_labels": {
            "performance": f"Primary metric ({primary}) per experiment",
            "durations": "Training duration per experiment (seconds)",
            "progression": f"Best {primary} so far",
            "resources": "Measured resource usage per experiment",
        },
    }


def _feature_importance_chart(
    experiments: List[Dict[str, Any]], problem_type: str
) -> List[Dict[str, Any]]:
    best = evaluator.best_by_metric(experiments, problem_type)
    if not best or not best.get("feature_importance"):
        return []
    return best["feature_importance"][:12]


def _headline_metrics(metrics: Dict[str, Any], problem_type: str) -> List[Tuple[str, Any]]:
    if problem_type == "classification":
        keys = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    else:
        keys = ["r2", "mae", "rmse"]
    return [(key, metrics[key]) for key in keys if metrics.get(key) is not None]


# -------------------------------------------------------- final recommendation
def build_recommendation(
    session: ExperimentSession,
    experiments: List[Dict[str, Any]],
    stop_reason: Optional[str] = None,
    stop_code: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the final recommendation, honouring the optimization priority."""
    problem_type = session.problem_type
    priority = session.priority or "balanced"
    primary = evaluator.primary_metric_name(problem_type)
    ranked = evaluator.rank_experiments(experiments, problem_type, priority)
    if not ranked:
        return {
            "available": False,
            "reason": "No experiment produced a usable score, so no model can be recommended.",
            "priority": priority,
            "primary_metric": primary,
            "experiments_completed": 0,
        }

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    completed = [item for item in experiments if item.get("status") == "completed"]
    stopped_early = bool(stop_code and stop_code not in {"target_reached", "exhausted_candidates"})
    spec = selector.get_spec(best["model_key"])

    if priority == "speed":
        priority_note = (
            "The 'speed' priority selects the fastest model whose score stays within 3% of the "
            "best score."
        )
    elif priority == "performance":
        priority_note = (
            "The 'performance' priority selects the model with the highest primary metric "
            "regardless of training time."
        )
    else:
        priority_note = (
            "The 'balanced' priority weights the normalised score at 70% and the normalised "
            "training time at 30%."
        )

    reason_parts = [
        f"{best['model_name']} achieved the best {primary} of {best['primary_metric']:.4f} across "
        f"{len(completed)} completed experiment(s).",
        priority_note,
    ]
    if runner_up:
        delta = float(best["primary_metric"]) - float(runner_up["primary_metric"])
        if runner_up["model_name"] != best["model_name"]:
            reason_parts.append(
                f"It beats the runner-up {runner_up['model_name']} by {delta:+.4f} {primary}."
            )
        else:
            reason_parts.append(
                f"It beats the previous {runner_up['model_name']} configuration by {delta:+.4f} "
                f"{primary}."
            )
    if best.get("training_duration_seconds"):
        reason_parts.append(
            f"Training took {best['training_duration_seconds']:.2f}s on "
            f"{best.get('dataset_rows') or 0:,} rows."
        )
    if best.get("test_metrics"):
        reason_parts.append(
            f"On the held-out test split it scored "
            f"{', '.join(f'{key} {value}' for key, value in _headline_metrics(best['test_metrics'], problem_type))}."
        )
    if best.get("kind") == "optimization":
        reason_parts.append(
            "The winning configuration came from an Optuna hyper-parameter search "
            f"({(best.get('trials') or {}).get('trials_completed')} trials)."
        )
    if stopped_early:
        reason_parts.append(
            f"Experimentation was stopped early ({stop_reason or 'user stop'}), so this"
            " recommendation is based only on the experiments that had already completed."
        )

    return {
        "available": True,
        "experiment_id": best["id"],
        "experiment_number": best.get("experiment_number"),
        "model_key": best["model_key"],
        "model_name": best["model_name"],
        "model_family": spec.family,
        "kind": best.get("kind"),
        "priority": priority,
        "primary_metric": primary,
        "metrics": best.get("metrics"),
        "test_metrics": best.get("test_metrics"),
        "cv_metrics": best.get("cv_metrics"),
        "hyperparameters": best.get("hyperparameters"),
        "training_duration_seconds": best.get("training_duration_seconds"),
        "resource_usage": best.get("resource_usage"),
        "feature_importance": best.get("feature_importance"),
        "experiments_completed": len(completed),
        "max_experiments": session.max_experiments,
        "experiments_used": len(experiments),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "stop_code": stop_code,
        "selection_score": best.get("selection_score"),
        "selection_rationale": best.get("selection_rationale"),
        "reason": " ".join(reason_parts),
        "ranked": [
            {
                "rank": item["rank"],
                "experiment_id": item["id"],
                "experiment_number": item.get("experiment_number"),
                "model_name": item.get("model_name"),
                "model_key": item.get("model_key"),
                "primary_metric": item.get("primary_metric"),
                "training_seconds": item.get("training_duration_seconds"),
                "selection_score": item.get("selection_score"),
                "selection_rationale": item.get("selection_rationale"),
                "is_recommended": item["rank"] == 1,
            }
            for item in ranked[:8]
        ],
        "model_details": selector.describe_model(best["model_key"]),
    }


def _session_context(session_id: str) -> Dict[str, Any]:
    """Load everything the worker needs and (re)build the plan when required."""
    with session_scope() as db:
        session = db.get(ExperimentSession, session_id)
        if session is None:
            raise NotFoundError(f"Session '{session_id}' was not found.")
        dataset = db.get(Dataset, session.dataset_id)
        if dataset is None:
            raise NotFoundError("The dataset for this session no longer exists.")
        analysis = dict(dataset.analysis or {})
        if not analysis:
            raise ConflictError("The dataset has no cached analysis; re-run the analysis first.")
        problem_type = session.problem_type
        target = session.target_column
        priority = session.priority
        interactive = bool(session.interactive)
        max_experiments = int(session.max_experiments)
        budget_override = bool(getattr(session, "budget_override", False))
        stored_plan = session.plan
        stored_budget = session.budget
        frame = analyzer.load_dataframe(dataset.stored_path)
        history = _experiment_rows(db, session_id, problem_type)
        snapshot = rm.resource_snapshot()

    split = trainer.split_dataset(frame, target, problem_type)
    profile = analysis.get("profile") or {}
    estimator = time_estimator.TimeEstimator(
        n_rows=len(split.X_train) + len(split.X_val) + len(split.X_test),
        effective_width=int(profile.get("estimated_onehot_width") or 0),
        cpu_count=int(snapshot.get("cpu_count") or 1),
    )
    estimator.refine_from_history(history)

    resume = bool(history) and bool(stored_plan)
    if resume:
        plan = dict(stored_plan)
        budget = dict(stored_budget or {})
        selection_data = plan.get("initial_model") or {}
        selection = selector.SelectionResult(
            model_key=selection_data.get("model_key") or history[0]["model_key"],
            model_name=selection_data.get("model_name") or history[0]["model_name"],
            rule=selection_data.get("rule") or "resumed",
            reason=selection_data.get("reason") or "Resumed from the saved plan.",
            evidence=selection_data.get("evidence") or {},
            alternates=list(plan.get("alternates") or []),
            candidate_pool=list(plan.get("candidate_pool") or []),
        )
        logger.info("Resuming session %s with %s existing experiment(s)", session_id, len(history))
    else:
        estimator.run_pilot(split, "linear_regression" if problem_type == "regression" else "logistic_regression")
        selection = selector.select_initial_model(analysis, snapshot)
        estimator.run_pilot(split, selection.model_key)
        budget = time_estimator.allocate_budget(
            analysis,
            estimator,
            selection.model_key,
            snapshot,
            priority,
            max_experiments if budget_override else None,
        )
        plan = experiment_planner.build_plan(
            analysis, selection, budget, split.summary(), snapshot, priority, estimator
        ).as_dict()

    return {
        "session_id": session_id,
        "problem_type": problem_type,
        "target": target,
        "priority": priority,
        "interactive": interactive,
        "max_experiments": int(budget.get("max_experiments") or max_experiments),
        "analysis": analysis,
        "split": split,
        "history": history,
        "snapshot": snapshot,
        "selection": selection,
        "estimator": estimator,
        "budget": budget,
        "plan": plan,
        "resume": resume,
    }


def _run_one_experiment(
    context: Dict[str, Any],
    control: RunnerControl,
    spec: Dict[str, Any],
    decision_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Train and persist a single experiment. Returns None when cancelled."""
    session_id = context["session_id"]
    problem_type = context["problem_type"]
    split = context["split"]
    estimator = context["estimator"]
    priority = context["priority"]
    model_key = spec.get("model_key") or context["selection"].model_key
    kind = spec.get("kind") or "initial"
    model_meta = selector.describe_model(model_key)
    number = max([item.get("experiment_number") or 0 for item in context["history"]] + [0]) + 1
    remaining_budget = max(0, context["max_experiments"] - len(context["history"]))

    estimate = estimator.estimate(model_key, extra_fits=6 if kind == "optimization" else 1)
    update_progress(
        session_id,
        status="running",
        stage="preparing",
        progress=0.05,
        current_experiment_number=number,
        current_model=model_meta["model_name"],
        eta_seconds=estimate.expected_seconds,
        message=(
            f"Experiment {number}/{context['max_experiments']}: preparing {model_meta['model_name']}"
        ),
        log=f"Experiment {number} started with {model_meta['model_name']} ({kind})",
    )

    with session_scope() as db:
        session = db.get(ExperimentSession, session_id)
        record = Experiment(
            session_id=session_id,
            dataset_id=session.dataset_id,
            experiment_number=number,
            model_key=model_key,
            model_name=model_meta["model_name"],
            kind=kind,
            status="running",
            hyperparameters=spec.get("params") or {},
            estimated_duration_seconds=estimate.expected_seconds,
            dataset_rows=len(split.X_train) + len(split.X_val),
            dataset_features=len(split.numeric_features) + len(split.categorical_features),
        )
        db.add(record)
        db.flush()
        experiment_id = record.id
        session.current_experiment_number = number
        session.status = "running"

    update_progress(session_id, current_experiment_id=experiment_id, stage="training")
    trials_info: Optional[Dict[str, Any]] = None
    params = dict(spec.get("params") or {})

    try:
        if kind == "optimization":
            per_trial_seconds = max(
                0.01, estimate.expected_seconds / max(1, settings.optimization_trials)
            )
            trials = optuna_optimizer.suggested_trials(
                remaining_budget, priority, per_trial_seconds
            )
            update_progress(
                session_id,
                stage="optimizing",
                progress=0.15,
                message=(
                    f"Optuna hyper-parameter search for {model_meta['model_name']} "
                    f"({trials} trials)"
                ),
                log=f"Optuna study started ({trials} trials)",
            )
            trials_info = optuna_optimizer.optimize(
                split=split,
                model_key=model_key,
                problem_type=problem_type,
                base_params=params,
                n_trials=trials,
                time_budget_seconds=min(settings.max_experiment_seconds, 240.0),
                cancel_check=lambda: control.cancel_current.is_set(),
                progress=lambda payload: update_progress(session_id, **payload),
            )
            if control.cancel_current.is_set():
                return _mark_cancelled(
                    experiment_id, session_id, "Optimisation cancelled by the user"
                )
            params = trials_info.get("best_params") or params
            update_progress(
                session_id,
                stage="training",
                progress=0.7,
                message=(
                    f"Refitting {model_meta['model_name']} with the best trial configuration "
                    f"({trials_info.get('trials_completed')} trials completed)"
                ),
            )

        if control.cancel_current.is_set():
            return _mark_cancelled(
                experiment_id, session_id, "Cancelled by the user before training"
            )

        result = trainer.run_experiment(
            split=split,
            model_key=model_key,
            params=params,
            problem_type=problem_type,
            with_cv=(kind == "initial"),
            time_budget_seconds=estimate.expected_seconds,
        )
    except Exception as exc:
        logger.exception("Experiment %s (%s) failed", number, model_key)
        with session_scope() as db:
            record = db.get(Experiment, experiment_id)
            record.status = "failed"
            record.error_message = str(exc)[:2000]
            record.finished_at = datetime.now(timezone.utc)
        update_progress(
            session_id,
            stage="failed",
            progress=1.0,
            message=f"Experiment {number} ({model_meta['model_name']}) failed: {exc}",
            log=f"Experiment {number} failed: {exc}",
            log_level="error",
        )
        return {"id": experiment_id, "status": "failed", "error": str(exc), "model_key": model_key}


    if control.cancel_current.is_set():
        return _mark_cancelled(experiment_id, session_id, "Cancelled by the user after training")

    metrics = result["validation_metrics"]
    primary = result["primary_metric"]
    with session_scope() as db:
        record = db.get(Experiment, experiment_id)
        record.status = "completed"
        record.hyperparameters = params
        record.metrics = metrics
        record.cv_metrics = result.get("cv_metrics")
        record.trials = trials_info
        record.feature_importance = result.get("feature_importance")
        record.training_duration_seconds = result["training_seconds"]
        record.resource_usage = result["resource_usage"]
        record.notes = " ".join(split.notes[:3])
        record.finished_at = datetime.now(timezone.utc)
        session = db.get(ExperimentSession, session_id)
        session.completed_experiments = int(session.completed_experiments or 0) + 1

    estimator.record(model_key, result["training_seconds"], len(split.X_train) + len(split.X_val))
    update_progress(
        session_id,
        status="running",
        stage="evaluated",
        progress=1.0,
        message=(
            f"Experiment {number} finished: {model_meta['model_name']} "
            f"{evaluator.primary_metric_name(problem_type)} "
            f"{primary if primary is not None else 'n/a'} in {result['training_seconds']:.2f}s"
        ),
        log=(
            f"Experiment {number} completed - {evaluator.summarize(metrics, problem_type)} "
            f"({result['training_seconds']:.2f}s)"
        ),
    )
    return {
        "id": experiment_id,
        "status": "completed",
        "model_key": model_key,
        "experiment_number": number,
        "metrics": metrics,
        "primary_metric": primary,
        "training_seconds": result["training_seconds"],
        "decision": decision_payload,
    }


def _mark_cancelled(
    experiment_id: str, session_id: str, message: str
) -> Optional[Dict[str, Any]]:
    with session_scope() as db:
        record = db.get(Experiment, experiment_id)
        if record is not None:
            record.status = "cancelled"
            record.error_message = message
            record.finished_at = datetime.now(timezone.utc)
    update_progress(
        session_id, stage="cancelled", message=message, log=message, log_level="warning"
    )
    return None


def _persist_decision(
    session_id: str,
    experiment_id: Optional[str],
    step: int,
    decision: Dict[str, Any],
) -> None:
    with session_scope() as db:
        db.add(
            AgentDecision(
                session_id=session_id,
                experiment_id=experiment_id,
                step=step,
                decision=decision.get("decision") or "info",
                headline=(decision.get("headline") or "")[:250],
                reasoning=decision.get("reasoning") or "",
                details=decision.get("details") or {},
            )
        )


# --------------------------------------------------------------- finalisation
def _finalize_session(
    context: Dict[str, Any],
    stop_reason: str,
    stop_code: str,
    cancelled: bool = False,
) -> None:
    """Pick the winner, refit it, score the test split and store the verdict."""
    session_id = context["session_id"]
    problem_type = context["problem_type"]
    with session_scope() as db:
        session = db.get(ExperimentSession, session_id)
        experiments = _experiment_rows(db, session_id, problem_type)
        saved_flag = bool(session.saved)

    completed = [item for item in experiments if item.get("status") == "completed"]
    recommendation = build_recommendation(session, experiments, stop_reason, stop_code)

    # Refit the winner on train + validation and score the untouched test split.
    # IMPORTANT: use the same priority-ranked winner that build_recommendation chose,
    # not just best_by_metric — so the recommended model and the saved artifact always match.
    if recommendation.get("available") and completed and not cancelled:
        priority = context.get("priority", "balanced")
        ranked = evaluator.rank_experiments(completed, problem_type, priority)
        best = ranked[0] if ranked else None
        if best is not None:
            update_progress(
                session_id,
                stage="finalizing",
                progress=0.9,
                message=(
                    f"Refitting {best['model_name']} on train+validation and scoring the held-out "
                    "test split"
                ),
                log=f"Final evaluation of {best['model_name']} on the held-out test set",
            )
            try:
                final = trainer.finalize_model(
                    split=context["split"],
                    model_key=best["model_key"],
                    params=best.get("hyperparameters") or {},
                    problem_type=problem_type,
                )
                artifact = trainer.persist_model(
                    final["pipeline"], best["model_key"], best["id"]
                )
                with session_scope() as db:
                    record = db.get(Experiment, best["id"])
                    record.test_metrics = final["test_metrics"]
                    record.artifact_path = artifact
                    session = db.get(ExperimentSession, session_id)
                    session.final_metrics = final["test_metrics"]
                recommendation["test_metrics"] = final["test_metrics"]
                recommendation["final_training_seconds"] = final["training_seconds"]
                recommendation["final_resource_usage"] = final["resource_usage"]
                recommendation["artifact_path"] = artifact
                recommendation["reason"] = (
                    recommendation["reason"]
                    + " The winning configuration was refit on the training + validation splits and "
                    "evaluated once on the untouched test split."
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Final evaluation failed")
                recommendation["finalization_error"] = str(exc)

    if stop_code in {"budget_exhausted", "target_reached", "exhausted_candidates"}:
        final_status = "completed"
    else:
        final_status = "stopped"
    if saved_flag:
        final_status = "stopped"

    best_row = evaluator.best_by_metric(experiments, problem_type)
    with session_scope() as db:
        session = db.get(ExperimentSession, session_id)
        session.status = final_status
        session.stop_reason = stop_reason
        session.recommendation = recommendation
        session.finished_at = datetime.now(timezone.utc)
        if best_row:
            session.best_experiment_id = best_row["id"]

    metric_key = evaluator.primary_metric_name(problem_type)
    metrics_for_display = recommendation.get("test_metrics") or recommendation.get("metrics") or {}
    best_metric = metrics_for_display.get(metric_key)
    update_progress(
        session_id,
        status=final_status,
        stage="finished",
        progress=1.0,
        eta_seconds=0.0,
        message=(
            f"Session finished - recommended model: {recommendation.get('model_name', 'n/a')}"
            + (f" ({metric_key} {best_metric})" if best_metric is not None else "")
        ),
        log=f"Session finished. {stop_reason}",
    )


# --------------------------------------------------------------- background worker
def _worker(session_id: str) -> None:
    """Daemon thread executing the autonomous ML experimentation loop."""
    logger.info("Session %s worker started", session_id)
    with _registry_lock:
        control = _runners.get(session_id)
    if control is None:
        logger.error("No control object found for session %s", session_id)
        return

    start_monotonic = time.perf_counter()

    try:
        update_progress(
            session_id,
            status="running",
            stage="initializing",
            message="Initializing session context and planning...",
        )
        context = _session_context(session_id)
        problem_type = context["problem_type"]

        with session_scope() as db:
            session = db.get(ExperimentSession, session_id)
            if session:
                session.status = "running"
                if not session.started_at:
                    session.started_at = datetime.now(timezone.utc)
                session.plan = context["plan"]
                session.budget = context["budget"]
                session.resource_snapshot = context["snapshot"]

        if not context["resume"]:
            _persist_decision(
                session_id=session_id,
                experiment_id=None,
                step=0,
                decision={
                    "decision": "initial_model",
                    "headline": f"Selected {context['selection'].model_name} as baseline",
                    "reasoning": context["selection"].reason,
                    "details": {
                        "rule": context["selection"].rule,
                        "evidence": context["selection"].evidence,
                        "candidate_pool": context["selection"].candidate_pool,
                    },
                },
            )

        # Main experimentation loop
        while True:
            if control.stop_requested.is_set():
                logger.info("Stop requested for session %s (mode=%s)", session_id, control.stop_mode)
                _finalize_session(
                    context,
                    f"Experimentation stopped early by the user ({control.stop_mode}).",
                    "user_stop",
                )
                break

            # Current history
            with session_scope() as db:
                history = _experiment_rows(db, session_id, problem_type)
            context["history"] = history

            # Check if budget exhausted
            if len(history) >= context["max_experiments"]:
                logger.info("Budget exhausted (%s experiments)", context["max_experiments"])
                _finalize_session(
                    context,
                    f"Experimentation completed: budget of {context['max_experiments']} experiments reached.",
                    "budget_exhausted",
                )
                break

            # Check if interactive review pause needed
            if control.interactive and len(history) > 0:
                update_progress(
                    session_id,
                    status="waiting",
                    stage="waiting_for_user",
                    message="Interactive mode: waiting for user to continue or stop...",
                    log="Waiting for user review before running next experiment",
                )
                with session_scope() as db:
                    s = db.get(ExperimentSession, session_id)
                    if s:
                        s.status = "waiting"

                resumed = _wait_for_continue(control, session_id)
                if not resumed or control.stop_requested.is_set():
                    _finalize_session(
                        context,
                        "Experimentation stopped by user during interactive review.",
                        "user_stop",
                    )
                    break

                with session_scope() as db:
                    s = db.get(ExperimentSession, session_id)
                    if s:
                        s.status = "running"
                update_progress(session_id, status="running")

            # Determine next experiment to run
            if len(history) == 0:
                current_spec = {
                    "model_key": context["selection"].model_key,
                    "kind": "initial",
                    "params": selector.default_params_for(
                        context["selection"].model_key,
                        context["problem_type"],
                        (context["analysis"] or {}).get("profile") or {},
                    ),
                }
                decision_payload = {
                    "decision": "initial_model",
                    "headline": f"Initial Baseline: {context['selection'].model_name}",
                    "reasoning": context["selection"].reason,
                    "details": {
                        "rule": context["selection"].rule,
                        "evidence": context["selection"].evidence,
                    },
                }
            else:
                elapsed = time.perf_counter() - start_monotonic
                d_ctx = decision_engine.DecisionContext(
                    problem_type=problem_type,
                    priority=context["priority"],
                    experiments=history,
                    budget=context["budget"],
                    elapsed_seconds=elapsed,
                    selection={
                        "model_key": context["selection"].model_key,
                        "model_name": context["selection"].model_name,
                        "alternates": context["plan"].get("alternates") or [],
                        "candidate_pool": context["selection"].candidate_pool,
                    },
                    resource_snapshot=context["snapshot"],
                    analysis=context["analysis"],
                )
                decision = decision_engine.decide(d_ctx)

                if decision.decision == decision_engine.STOP:
                    logger.info("Decision engine triggered STOP: %s", decision.reasoning)
                    _finalize_session(
                        context,
                        decision.reasoning,
                        decision.stop_code or "agent_stop",
                    )
                    break

                current_spec = decision.next_experiment or {
                    "model_key": context["selection"].model_key,
                    "kind": "alternative",
                    "params": {},
                }
                decision_payload = decision.as_dict()

            # Run one experiment
            result = _run_one_experiment(context, control, current_spec, decision_payload)

            if control.cancel_current.is_set():
                control.cancel_current.clear()
                if control.stop_requested.is_set():
                    _finalize_session(
                        context,
                        "Experiment cancelled and session stopped by user.",
                        "user_cancelled",
                        cancelled=True,
                    )
                    break

            if result is None:
                if control.stop_requested.is_set():
                    _finalize_session(
                        context,
                        "Session stopped following experiment cancellation.",
                        "user_cancelled",
                        cancelled=True,
                    )
                    break
                continue

            # Persist decision for this experiment
            step_num = len(context["history"]) + 1
            _persist_decision(
                session_id=session_id,
                experiment_id=result.get("id"),
                step=step_num,
                decision=decision_payload,
            )

    except Exception as exc:
        logger.exception("Session worker crashed for %s: %s", session_id, exc)
        update_progress(
            session_id,
            status="failed",
            stage="crashed",
            message=f"Session worker encountered an unhandled error: {exc}",
            log=f"Worker crashed: {exc}",
            log_level="error",
        )
        with session_scope() as db:
            session = db.get(ExperimentSession, session_id)
            if session:
                session.status = "failed"
                session.stop_reason = f"Unhandled worker failure: {exc}"
                session.finished_at = datetime.now(timezone.utc)



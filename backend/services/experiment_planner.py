"""Experiment planning: what to run first, what comes next, and when to stop.

The planner is deliberately deterministic and inspectable. It combines

  * the model selector's first candidate and the ordered alternatives,
  * the budget allocated from measured resources plus the pilot timing,
  * the preprocessing plan the first experiment will use,
  * the exact stopping criteria, and
  * the documented decision rule chain that drives the loop.

The plan is stored on the session so the UI can explain *why* the agent does what
it does before any model has been trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from logging_config import get_logger
from services import model_selector as selector
from services import preprocessing
from services import time_estimator

logger = get_logger(__name__)


DECISION_RULES: List[Dict[str, str]] = [
    {
        "id": "budget_exhausted",
        "title": "Experimentation budget exhausted",
        "description": "Stop when the allocated number of experiments has been used.",
    },
    {
        "id": "target_reached",
        "title": "Target quality reached",
        "description": (
            f"Stop when the primary metric reaches the 'good enough' target "
            f"(R2 >= {settings.target_r2} for regression, weighted F1 >= {settings.target_f1} for "
            "classification) and the last experiment added little."
        ),
    },
    {
        "id": "plateau",
        "title": "Score plateau",
        "description": (
            f"Stop after {settings.plateau_patience} consecutive experiments whose improvement is "
            f"below {settings.min_improvement_epsilon}."
        ),
    },
    {
        "id": "time_budget",
        "title": "Session time budget",
        "description": (
            f"Stop when elapsed time plus the estimated duration of the next experiment would "
            f"exceed {settings.max_session_seconds:.0f}s."
        ),
    },
    {
        "id": "weak_result",
        "title": "Weak result - switch model family",
        "description": (
            f"If the best metric stays below the weak threshold (R2 {settings.weak_r2} / F1 "
            f"{settings.weak_f1}), try an untried model from a different family, preferring "
            "ensembles."
        ),
    },
    {
        "id": "optimize_best_model",
        "title": "Optimise the leader",
        "description": (
            "If the best model has a searchable hyper-parameter space and has not been optimised "
            "yet, spend an experiment on a TPE search (Optuna)."
        ),
    },
    {
        "id": "try_another_model",
        "title": "Try another model",
        "description": (
            "If the leader is already optimised or cannot be searched, test the next untried "
            "candidate model instead of repeating an identical run."
        ),
    },
]


@dataclass
class ExperimentPlan:
    problem_type: str
    target_column: str
    priority: str
    initial_model: Dict[str, Any]
    alternates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_pool: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    stopping_criteria: Dict[str, Any] = field(default_factory=dict)
    decision_rules: List[Dict[str, str]] = field(default_factory=list)
    split_summary: Dict[str, Any] = field(default_factory=dict)
    resource_snapshot: Dict[str, Any] = field(default_factory=dict)
    queue: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "target_column": self.target_column,
            "priority": self.priority,
            "initial_model": self.initial_model,
            "alternates": self.alternates,
            "candidate_pool": self.candidate_pool,
            "budget": self.budget,
            "preprocessing": self.preprocessing,
            "stopping_criteria": self.stopping_criteria,
            "decision_rules": self.decision_rules,
            "split_summary": self.split_summary,
            "resource_snapshot": self.resource_snapshot,
            "queue": self.queue,
            "notes": self.notes,
        }

def build_plan(
    analysis: Dict[str, Any],
    selection: Any,
    budget: Dict[str, Any],
    split_summary: Dict[str, Any],
    resource_snapshot: Dict[str, Any],
    priority: str = "balanced",
    estimator: Optional[time_estimator.TimeEstimator] = None,
) -> ExperimentPlan:
    """Assemble the full plan for one autonomous session."""
    problem_type = analysis.get("problem_type", "regression")
    profile = analysis.get("profile") or {}
    spec = selector.get_spec(selection.model_key)

    preprocessing_plan = preprocessing.plan_preprocessing(
        numeric_features=split_summary.get("numeric_features") or [],
        categorical_features=split_summary.get("categorical_features") or [],
        model_needs_scaling=spec.needs_scaling,
        n_rows=int(split_summary.get("train_rows") or profile.get("n_rows") or 0),
    )

    notes: List[str] = []
    linearity = analysis.get("linearity")
    if problem_type == "regression" and linearity:
        notes.append(
            f"Linearity probe: linear baseline R2 {linearity.get('linear_cv_r2')} vs depth-6 tree "
            f"R2 {linearity.get('tree_cv_r2')} -> {linearity.get('verdict')}."
        )
    if (analysis.get("problem_type_detection") or {}).get("ambiguous"):
        notes.append(
            "The target column is ambiguous; the detected problem type can be overridden by the "
            "user before or during the session."
        )
    if split_summary.get("dropped_features"):
        dropped = ", ".join(
            f"{item['column']} ({item['reason']})"
            for item in split_summary["dropped_features"][:5]
        )
        notes.append(f"Features excluded from training: {dropped}.")
    notes.append(
        "The first selected model is a *candidate*: it only becomes the recommended model if it "
        "survives comparison against the alternatives."
    )

    initial_model = dict(selection.as_dict())
    initial_model["default_params"] = selector.default_params_for(
        selection.model_key, problem_type, profile
    )
    initial_model["description"] = spec.description
    initial_model["strengths"] = spec.strengths
    initial_model["weaknesses"] = spec.weaknesses

    plan = ExperimentPlan(
        problem_type=problem_type,
        target_column=str(analysis.get("target") or split_summary.get("target_column") or ""),
        priority=priority,
        initial_model=initial_model,
        alternates=list(selection.alternates),
        candidate_pool=list(selection.candidate_pool),
        budget=budget,
        preprocessing=preprocessing_plan.as_dict(),
        stopping_criteria=budget.get("stopping_criteria", {}),
        decision_rules=DECISION_RULES,
        split_summary=split_summary,
        resource_snapshot=resource_snapshot,
        notes=notes,
    )
    plan.queue = estimate_queue(plan.as_dict(), estimator)
    return plan


def describe_next_experiment(
    plan: Dict[str, Any], history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Preview of the next experiment (used by the dashboard before it starts)."""
    tried = {item.get("model_key") for item in history}
    initial = (plan.get("initial_model") or {}).get("model_key")
    if initial and initial not in tried:
        return {
            "kind": "initial",
            "model_key": initial,
            "model_name": (plan.get("initial_model") or {}).get("model_name"),
            "reason": "First candidate selected by the model selector.",
        }
    for alternate in plan.get("alternates") or []:
        if alternate.get("model_key") not in tried:
            return {
                "kind": "alternative",
                "model_key": alternate["model_key"],
                "model_name": alternate.get("model_name"),
                "reason": alternate.get("reason"),
            }
    return {
        "kind": "optimization",
        "model_key": initial,
        "model_name": (plan.get("initial_model") or {}).get("model_name"),
        "reason": "All candidates have been tried; the next experiment optimises the leader.",
    }


def estimate_queue(
    plan: Dict[str, Any], estimator: Optional[time_estimator.TimeEstimator]
) -> List[Dict[str, Any]]:
    """Rough preview of the planned queue with estimated durations."""
    if estimator is None:
        return []
    queue: List[Dict[str, Any]] = []
    initial = (plan.get("initial_model") or {}).get("model_key")
    if initial:
        estimate = estimator.estimate(initial)
        queue.append(
            {
                "position": 1,
                "kind": "initial",
                "model_key": initial,
                "model_name": (plan.get("initial_model") or {}).get("model_name"),
                "estimated_seconds": estimate.expected_seconds,
                "estimated_range": [estimate.low_seconds, estimate.high_seconds],
            }
        )
    for alternate in (plan.get("alternates") or [])[:5]:
        try:
            estimate = estimator.estimate(alternate["model_key"])
        except KeyError:
            continue
        queue.append(
            {
                "position": len(queue) + 1,
                "kind": "alternative",
                "model_key": alternate["model_key"],
                "model_name": alternate.get("model_name"),
                "estimated_seconds": estimate.expected_seconds,
                "estimated_range": [estimate.low_seconds, estimate.high_seconds],
            }
        )
    return queue


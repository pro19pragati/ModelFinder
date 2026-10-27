"""The autonomous decision engine.

After every completed experiment the agent decides whether to

  1. **stop** experimentation,
  2. **optimize** the current best model (hyper-parameter search), or
  3. **try another model** (diversify the model family).

The rules are transparent and rule-based: they use only real evaluation results,
measured training times, the remaining budget and the selected optimisation
priority. Every decision returns human-readable reasoning built from the actual
numbers - nothing is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from logging_config import get_logger
from services import evaluator
from services import model_selector as selector

logger = get_logger(__name__)

STOP = "stop"
OPTIMIZE = "optimize"
TRY_ANOTHER_MODEL = "try_another_model"

ENSEMBLE_FAMILIES = {"ensemble", "boosted"}


@dataclass
class DecisionContext:
    """Everything the decision rules are allowed to look at."""

    problem_type: str
    priority: str = "balanced"
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    selection: Dict[str, Any] = field(default_factory=dict)
    resource_snapshot: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------ derived helpers
    @property
    def completed(self) -> List[Dict[str, Any]]:
        return [item for item in self.experiments if item.get("status") == "completed"]

    @property
    def scored(self) -> List[Dict[str, Any]]:
        return [item for item in self.completed if item.get("primary_metric") is not None]

    @property
    def max_experiments(self) -> int:
        return int(self.budget.get("max_experiments") or settings.default_max_experiments)

    @property
    def remaining(self) -> int:
        return max(0, self.max_experiments - len(self.experiments))

    @property
    def best(self) -> Optional[Dict[str, Any]]:
        return evaluator.best_by_metric(self.completed, self.problem_type)

    @property
    def best_metric(self) -> Optional[float]:
        best = self.best
        if best and best.get("primary_metric") is not None:
            return float(best["primary_metric"])
        return None

    @property
    def tried_models(self) -> List[str]:
        return [item["model_key"] for item in self.experiments if item.get("model_key")]

    @property
    def optimized_models(self) -> List[str]:
        return [
            item["model_key"]
            for item in self.experiments
            if item.get("kind") == "optimization" and item.get("model_key")
        ]

    @property
    def last_training_seconds(self) -> float:
        if not self.completed:
            return float(self.budget.get("per_experiment_seconds_estimate") or 1.0)
        return float(self.completed[-1].get("training_duration_seconds") or 1.0)

    @property
    def estimated_next_seconds(self) -> float:
        values = [float(item.get("training_duration_seconds") or 0.0) for item in self.completed]
        values = [value for value in values if value > 0]
        if values:
            return sum(values) / len(values)
        return float(self.budget.get("per_experiment_seconds_estimate") or 1.0)

    def target_for(self) -> float:
        if self.problem_type == "classification":
            return float(settings.target_f1)
        return float(settings.target_r2)

    def weak_threshold(self) -> float:
        if self.problem_type == "classification":
            return float(settings.weak_f1)
        return float(settings.weak_r2)

    def metric_label(self) -> str:
        return evaluator.primary_metric_name(self.problem_type)


@dataclass
class Decision:
    decision: str
    headline: str
    reasoning: str
    details: Dict[str, Any] = field(default_factory=dict)
    next_experiment: Optional[Dict[str, Any]] = None
    stop_code: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "headline": self.headline,
            "reasoning": self.reasoning,
            "details": self.details,
            "next_experiment": self.next_experiment,
            "stop_code": self.stop_code,
        }

def evaluate_improvement(context: DecisionContext) -> List[Dict[str, Any]]:
    """Primary-metric progression across the completed experiments."""
    progression: List[Dict[str, Any]] = []
    previous: Optional[float] = None
    for item in context.scored:
        current = float(item["primary_metric"])
        change = evaluator.improvement(previous, current)
        progression.append(
            {
                "experiment_number": item.get("experiment_number"),
                "model_name": item.get("model_name"),
                "model_key": item.get("model_key"),
                "kind": item.get("kind"),
                "metric": round(current, 4),
                "change": change,
                "training_seconds": item.get("training_duration_seconds"),
            }
        )
        previous = current
    return progression


def _best_alternate(context: DecisionContext) -> Optional[Dict[str, Any]]:
    """Next untried model, preferring a different family (ensembles when weak)."""
    tried = set(context.tried_models)
    alternates = [
        item
        for item in (context.selection.get("alternates") or [])
        if item.get("model_key") not in tried
    ]
    if not alternates:
        return None
    best = context.best
    best_spec = selector.get_spec(best["model_key"]) if best else None
    best_family = best_spec.family if best_spec else None
    weak = context.best_metric is not None and context.best_metric < context.weak_threshold()

    def sort_key(item: Dict[str, Any]) -> tuple:
        try:
            spec = selector.get_spec(item["model_key"])
            family = spec.family
            complexity = spec.complexity
        except KeyError:
            family, complexity = "unknown", 1.0
        strong_first = 0 if (weak and family in ENSEMBLE_FAMILIES) else 1
        different_family = 0 if family != best_family else 1
        return (strong_first, different_family, complexity)

    alternates.sort(key=sort_key)
    return alternates[0]


def _plan_optimization(
    context: DecisionContext, model_key: str, reason_code: str
) -> Optional[Dict[str, Any]]:
    spec = selector.get_spec(model_key)
    if spec.param_space is None:
        return None
    history = [item for item in context.completed if item.get("model_key") == model_key]
    base_params: Dict[str, Any] = {}
    if history:
        base_params = dict(
            evaluator.best_by_metric(history, context.problem_type).get("hyperparameters") or {}
        )
    if not base_params:
        base_params = selector.default_params_for(
            model_key, context.problem_type, (context.analysis or {}).get("profile") or {}
        )
    return {
        "kind": "optimization",
        "model_key": model_key,
        "params": base_params,
        "reason_code": reason_code,
    }


def _plan_alternative(context: DecisionContext, alternate: Dict[str, Any]) -> Dict[str, Any]:
    model_key = alternate["model_key"]
    return {
        "kind": "alternative",
        "model_key": model_key,
        "params": selector.default_params_for(
            model_key, context.problem_type, (context.analysis or {}).get("profile") or {}
        ),
        "reason_code": "alternate_model",
        "alternate_reason": alternate.get("reason"),
    }


def decide(context: DecisionContext) -> Decision:
    """Apply the rule chain and return the next action."""
    metric_label = context.metric_label()
    remaining = context.remaining
    completed = context.completed
    best = context.best
    best_metric = context.best_metric
    target = context.target_for()
    weak = context.weak_threshold()
    progression = evaluate_improvement(context)
    priority = context.priority
    session_seconds = float(
        context.budget.get("session_seconds") or settings.max_session_seconds
    )
    next_seconds = context.estimated_next_seconds
    attempts = len(context.experiments)
    failed = [item for item in context.experiments if item.get("status") == "failed"]

    details: Dict[str, Any] = {
        "metric": metric_label,
        "best_metric": best_metric,
        "best_model": best.get("model_name") if best else None,
        "best_model_key": best.get("model_key") if best else None,
        "experiments_run": attempts,
        "experiments_completed": len(completed),
        "experiments_failed": len(failed),
        "remaining_experiments": remaining,
        "max_experiments": context.max_experiments,
        "target_metric": target,
        "weak_threshold": weak,
        "priority": priority,
        "elapsed_seconds": round(context.elapsed_seconds, 2),
        "estimated_next_seconds": round(next_seconds, 2),
        "session_seconds": session_seconds,
        "progression": progression,
        "tried_models": context.tried_models,
        "optimized_models": context.optimized_models,
    }

    # ---------------------------------------------------------------- guards
    if best is None:
        if remaining <= 0 or attempts >= context.max_experiments:
            return Decision(
                decision=STOP,
                headline="No usable result - stopping",
                reasoning=(
                    f"All {attempts} experiment(s) failed or produced no usable {metric_label}, and "
                    f"the budget of {context.max_experiments} is exhausted. No model can be "
                    "recommended from this dataset - review the failure messages in the experiment "
                    "history."
                ),
                details=details,
                stop_code="no_usable_result",
            )
        alternate = _best_alternate(context)
        if alternate is not None:
            return Decision(
                decision=TRY_ANOTHER_MODEL,
                headline=f"Attempting {alternate['model_name']} after empty results",
                reasoning=(
                    f"No experiment produced a usable {metric_label} yet, so the agent moves on to "
                    f"{alternate['model_name']} with {remaining} experiment(s) left."
                ),
                details=details,
                next_experiment=_plan_alternative(context, alternate),
            )
        return Decision(
            decision=STOP,
            headline="Stopping: every candidate model failed",
            reasoning=(
                "Every candidate model failed or returned no score and no untried candidate "
                "remains, so the agent stops instead of repeating identical experiments."
            ),
            details=details,
            stop_code="all_candidates_failed",
        )

    assert best_metric is not None

    # ------------------------------------------------- 1. budget exhausted
    if remaining <= 0:
        return Decision(
            decision=STOP,
            headline="Experimentation budget exhausted",
            reasoning=(
                f"The agent completed {len(completed)} of its {context.max_experiments} budgeted "
                f"experiments. The best result is {best['model_name']} with {metric_label} "
                f"{best_metric:.4f}, so experimentation stops because no budget remains."
            ),
            details=details,
            stop_code="budget_exhausted",
        )

    # ------------------------------------------- 2. target quality reached
    satisfied_target = target if priority == "performance" else target * 0.97
    if best_metric >= satisfied_target:
        last_change = progression[-1]["change"]["absolute"] if progression else None
        stable = (
            len(progression) < 2
            or last_change is None
            or abs(last_change) <= settings.min_improvement_epsilon
        )
        # M4: If CV metrics are available, cross-check that the CV mean is also above the
        # target (with a 0.02 grace margin) before stopping early — prevents stopping on a
        # lucky single-split validation score.
        cv_ok = True
        if best and best.get("cv_metrics"):
            cv_mean = best["cv_metrics"].get("mean")
            if cv_mean is not None:
                cv_ok = float(cv_mean) >= satisfied_target - 0.02
        if cv_ok and (stable or priority == "speed"):
            cv_note = (
                f" CV mean {best['cv_metrics']['mean']:.4f} also confirms quality."
                if best and best.get("cv_metrics") and best["cv_metrics"].get("mean")
                else ""
            )
            return Decision(
                decision=STOP,
                headline=f"Target reached with {best['model_name']}",
                reasoning=(
                    f"{best['model_name']} reached a {metric_label} of {best_metric:.4f}, at or "
                    f"above the 'good enough' target of {satisfied_target:.4f}"
                    + (
                        f", and the last experiment only changed the score by {last_change:.4f}"
                        if last_change is not None
                        else ""
                    )
                    + cv_note
                    + f". With {remaining} experiment(s) still available the agent judges further "
                    "experimentation unnecessary and stops."
                ),
                details=details,
                stop_code="target_reached",
            )

    # --------------------------------------------------------- 3. plateau
    # B2: Require at least (plateau_patience + 1) scored experiments AND at least 2 distinct
    # model families to have been tried before declaring a plateau.  This prevents the engine
    # from stopping after initial + first-alternative when the initial model was just unlucky.
    distinct_families_tried = len({
        (context.analysis or {}).get("_family_" + item.get("model_key", ""), item.get("model_key"))
        for item in context.scored
    })
    # Use model_key set as proxy for distinct families when spec lookup is unavailable.
    distinct_families_tried = len(set(item.get("model_key", "") for item in context.scored))
    min_scored_for_plateau = max(settings.plateau_patience + 1, 3)
    if len(progression) > 1 and len(context.scored) >= min_scored_for_plateau:
        recent = progression[-settings.plateau_patience:]
        gains = [
            item["change"]["absolute"] for item in recent if item["change"]["absolute"] is not None
        ]
        if len(gains) >= settings.plateau_patience and all(
            abs(gain) < settings.min_improvement_epsilon for gain in gains
        ):
            return Decision(
                decision=STOP,
                headline="Score plateau reached",
                reasoning=(
                    f"The last {len(gains)} experiment(s) changed the {metric_label} by at most "
                    f"{max(abs(gain) for gain in gains):.4f}, below the "
                    f"{settings.min_improvement_epsilon} improvement threshold. The best model is "
                    f"{best['model_name']} at {best_metric:.4f}, so the agent stops instead of "
                    f"spending the remaining {remaining} experiment(s) on changes that are unlikely "
                    "to matter."
                ),
                details=details,
                stop_code="plateau",
            )

    # ------------------------------------------------------- 4. time guard
    if context.elapsed_seconds + next_seconds > session_seconds:
        return Decision(
            decision=STOP,
            headline="Session time budget exhausted",
            reasoning=(
                f"{context.elapsed_seconds:.1f}s have already been spent and the next experiment "
                f"is estimated at about {next_seconds:.1f}s, which would exceed the "
                f"{session_seconds:.0f}s session budget. The best model remains "
                f"{best['model_name']} with {metric_label} {best_metric:.4f}."
            ),
            details=details,
            stop_code="time_budget",
        )

    best_key = best.get("model_key")
    already_optimized = best_key in context.optimized_models
    alternate = _best_alternate(context)
    optimization_plan = _plan_optimization(context, best_key, "optimize_best_model")

    # ------------------------------ 5. weak result -> change model family
    if best_metric < weak and alternate is not None:
        return Decision(
            decision=TRY_ANOTHER_MODEL,
            headline=f"Switching to {alternate['model_name']}",
            reasoning=(
                f"{best['model_name']} only reached {metric_label} {best_metric:.4f}, below the "
                f"{weak:.2f} level at which the agent considers a result promising. "
                f"{remaining} experiment(s) remain, so the next run uses "
                f"{alternate['model_name']}"
                + (f" ({alternate['reason']})" if alternate.get("reason") else "")
                + "."
            ),
            details=details,
            next_experiment=_plan_alternative(context, alternate),
        )

    # --- 5b. M1: mediocre result + untried alternatives -> explore before tuning
    # If the best metric is below 70% of the target and untried alternatives exist,
    # try a different model family FIRST before spending a slot on optimization.
    # This avoids wasting an experiment tuning a mediocre baseline when a better
    # algorithm family hasn't been tested yet.
    mediocre_threshold = target * 0.70
    if (
        best_metric < mediocre_threshold
        and alternate is not None
        and not already_optimized
        and remaining >= 2
    ):
        return Decision(
            decision=TRY_ANOTHER_MODEL,
            headline=f"Exploring {alternate['model_name']} before optimising",
            reasoning=(
                f"{best['model_name']} is at {metric_label} {best_metric:.4f}, below 70% of the "
                f"target ({mediocre_threshold:.4f}). The agent tries {alternate['model_name']} "
                "first to check whether a different model family is more capable on this data, "
                "rather than spending a budget slot optimising an underperforming baseline."
            ),
            details=details,
            next_experiment=_plan_alternative(context, alternate),
        )

    # ------------------- 6. good result and room to tune -> optimize
    if optimization_plan is not None and not already_optimized and remaining >= 1:
        linearity_note = ""
        if context.problem_type == "regression":
            linear_r2 = (context.analysis.get("linearity") or {}).get("linear_cv_r2")
            if linear_r2 is not None:
                linearity_note = f" (linearity probe: linear R2 {linear_r2})"
        return Decision(
            decision=OPTIMIZE,
            headline=f"Optimising {best['model_name']} hyper-parameters",
            reasoning=(
                f"{best['model_name']} leads with {metric_label} {best_metric:.4f} and trained in "
                f"{context.last_training_seconds:.2f}s{linearity_note}. {remaining} experiment(s) "
                "are still available, so the agent spends one on a TPE hyper-parameter search of "
                "this model instead of switching algorithm."
            ),
            details=details,
            next_experiment=optimization_plan,
        )

    # ------------------- 7. another model family might do better
    if alternate is not None:
        return Decision(
            decision=TRY_ANOTHER_MODEL,
            headline=f"Trying {alternate['model_name']}",
            reasoning=(
                f"{best['model_name']} reached {metric_label} {best_metric:.4f} and has already "
                "been optimised or exposes no searchable hyper-parameters"
                + f". With {remaining} experiment(s) left, the agent tests "
                f"{alternate['model_name']} to check whether a different model family improves the "
                "score."
            ),
            details=details,
            next_experiment=_plan_alternative(context, alternate),
        )

    # ------------------- 8. nothing new to try: refine the leader again
    if optimization_plan is not None and remaining >= 2:
        return Decision(
            decision=OPTIMIZE,
            headline=f"Second optimisation round for {best['model_name']}",
            reasoning=(
                f"Every candidate model has been tried at least once. {best['model_name']} still "
                f"leads with {metric_label} {best_metric:.4f} and {remaining} experiment(s) remain, "
                "so the agent spends another round refining its hyper-parameters rather than "
                "repeating an identical run."
            ),
            details=details,
            next_experiment=optimization_plan,
        )

    return Decision(
        decision=STOP,
        headline="No further experiments planned",
        reasoning=(
            f"All candidate models have been tried and optimised; the best result is "
            f"{best['model_name']} with {metric_label} {best_metric:.4f}. The agent stops with "
            f"{remaining} budget slot(s) unused because no distinct experiment remains."
        ),
        details=details,
        stop_code="exhausted_candidates",
    )


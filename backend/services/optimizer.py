"""Hyper-parameter optimisation with Optuna.

One *optimisation experiment* = one Optuna study counted as a single entry in the
experiment budget. Inside the study several trials are trained on the training
split and scored on the validation split; the best configuration is then
re-evaluated through the normal ``trainer`` path so the recorded metrics stay
comparable with every other experiment.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import settings
from logging_config import get_logger
from services import evaluator
from services import model_selector as selector
from services import preprocessing
from services import trainer

logger = get_logger(__name__)


def suggest_configuration(model_key: str, trial: Any) -> Dict[str, Any]:
    """Ask the model's search space for one candidate configuration."""
    space = selector.param_search_space(model_key)
    if space is None:
        return {}
    return space(trial)


def optimize(
    split: Any,
    model_key: str,
    problem_type: str,
    base_params: Dict[str, Any],
    n_trials: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run an Optuna study and return the best configuration + trial summary.

    ``cancel_check`` is polled between trials so a user-initiated stop is honoured
    without waiting for the whole study to finish.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials = max(settings.optimization_min_trials, int(n_trials or settings.optimization_trials))
    budget = float(time_budget_seconds or settings.max_experiment_seconds)
    spec = selector.get_spec(model_key)
    hist_mode = bool(base_params.get("hist"))
    plan = trainer.build_plan_for(model_key, split)

    trial_rows: List[Dict[str, Any]] = []
    best_holder: Dict[str, Any] = {"score": None, "params": dict(base_params)}

    def objective(trial: Any) -> float:
        if cancel_check is not None and cancel_check():
            raise optuna.TrialPruned("cancelled by user")
        params = dict(base_params)
        params.update(suggest_configuration(model_key, trial))
        try:
            estimator = spec.builder(params)
            pipeline = preprocessing.build_pipeline(estimator, plan)
            started = time.perf_counter()
            pipeline.fit(split.X_train, split.y_train)
            seconds = time.perf_counter() - started
            metrics = trainer.evaluate_pipeline(
                pipeline, split.X_val, split.y_val, problem_type, split.classes
            )
            score = evaluator.primary_metric_value(metrics, problem_type)
        except Exception as exc:
            trial_rows.append(
                {"trial": trial.number, "score": None, "params": params, "error": str(exc)[:200]}
            )
            logger.debug("Optuna trial %s failed: %s", trial.number, exc)
            raise optuna.TrialPruned(str(exc)) from exc

        if score is None or (isinstance(score, float) and np.isnan(score)):
            raise optuna.TrialPruned("no usable score")

        trial_rows.append(
            {
                "trial": trial.number,
                "score": float(round(score, 5)),
                "seconds": float(round(seconds, 4)),
                "params": params,
            }
        )
        if best_holder["score"] is None or score > best_holder["score"]:
            best_holder["score"] = float(score)
            best_holder["params"] = params
        if progress is not None:
            progress(
                {
                    "stage": "optimizing",
                    "message": (
                        f"Optuna trial {trial.number + 1}/{trials} - best "
                        f"{evaluator.primary_metric_name(problem_type)} "
                        f"{best_holder['score']:.4f}"
                    ),
                    "progress": min(0.95, (trial.number + 1) / max(1, trials)),
                    "trials_completed": len(
                        [row for row in trial_rows if row.get("score") is not None]
                    ),
                    "best_trial_score": best_holder["score"],
                }
            )
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_state, multivariate=True),
    )
    study.set_user_attr("hist", hist_mode)
    study.set_user_attr("model_key", model_key)
    study.set_user_attr("problem_type", problem_type)

    started = time.perf_counter()
    try:
        study.optimize(
            objective,
            n_trials=trials,
            timeout=budget,
            catch=(Exception,),
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover - safety net
        logger.warning("Optimisation study for %s stopped early: %s", model_key, exc)
    elapsed = time.perf_counter() - started

    scored = [row for row in trial_rows if row.get("score") is not None]
    completed_trials = len(scored)
    best_params = dict(best_holder["params"])
    if scored:
        best_row = max(scored, key=lambda row: row["score"])
        best_params = dict(best_row["params"])

    return {
        "model_key": model_key,
        "model_name": spec.label,
        "base_params": {k: v for k, v in base_params.items() if k != "hist"},
        "best_params": {k: v for k, v in best_params.items() if k != "hist"},
        "best_trial_score": best_holder["score"],
        "trials_requested": trials,
        "trials_completed": completed_trials,
        "trials_failed": len(trial_rows) - completed_trials,
        "optimisation_seconds": float(round(elapsed, 4)),
        "pruned": bool(completed_trials < trials),
        "trial_history": sorted(scored, key=lambda row: row["trial"])[:40],
        "sampler": "TPE (multivariate)",
        "scoring": evaluator.primary_metric_name(problem_type),
        "note": (
            "One optimisation experiment = one Optuna study; the study's inner trials are counted "
            "as trials, not as separate experiments."
        ),
    }


def suggested_trials(remaining_budget: int, priority: str, per_trial_seconds: float) -> int:
    """How many Optuna trials to spend inside a single optimisation experiment."""
    base = settings.optimization_trials
    if remaining_budget <= 1:
        base = max(settings.optimization_min_trials, base // 2)
    if priority == "speed":
        base = max(settings.optimization_min_trials, int(base * 0.6))
    elif priority == "performance":
        base = int(base * 1.5)
    if per_trial_seconds > 0:
        affordable = int(
            max(
                settings.optimization_min_trials,
                settings.max_experiment_seconds // max(0.01, per_trial_seconds),
            )
        )
        base = min(base, affordable)
    return int(max(settings.optimization_min_trials, min(60, base)))


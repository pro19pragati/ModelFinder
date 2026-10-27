"""Metric computation and priority-aware ranking of experiments.

Regression  : MAE, MSE, RMSE, R2 (+ MAPE)
Classification: accuracy, precision, recall, F1 (weighted + macro), confusion
                matrix, balanced accuracy, ROC-AUC (when probabilities exist)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)

PRIMARY_METRIC = {"regression": "r2", "classification": "f1_weighted"}
SECONDARY_METRIC = {"regression": "rmse", "classification": "accuracy"}
# For regression a *lower* value is better for these metrics.
LOWER_IS_BETTER = {"mae", "mse", "rmse", "mape", "log_loss"}


def _clean(value: Any, digits: int = 4) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[mask], prediction[mask]
    if truth.size == 0:
        return {"n_samples": 0, "error": "no valid samples"}

    mae = mean_absolute_error(truth, prediction)
    mse = mean_squared_error(truth, prediction)
    rmse = math.sqrt(mse)
    r2 = r2_score(truth, prediction) if truth.size > 1 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        nonzero = np.abs(truth) > 1e-9
        mape = (
            float(np.mean(np.abs((truth[nonzero] - prediction[nonzero]) / truth[nonzero])) * 100)
            if nonzero.any()
            else None
        )
    return {
        "n_samples": int(truth.size),
        "mae": _clean(mae),
        "mse": _clean(mse),
        "rmse": _clean(rmse),
        "r2": _clean(r2),
        "mape": _clean(mape, 2) if mape is not None else None,
        "mean_target": _clean(truth.mean()),
        "std_target": _clean(truth.std()),
        "residual_std": _clean(float((truth - prediction).std())),
    }


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.size == 0:
        return {"n_samples": 0, "error": "no valid samples"}

    resolved_labels = list(labels) if labels is not None else sorted(
        {*(np.unique(truth)).tolist(), *(np.unique(prediction)).tolist()}, key=str
    )
    matrix = confusion_matrix(truth, prediction, labels=resolved_labels)

    metrics: Dict[str, Any] = {
        "n_samples": int(truth.size),
        "n_classes": int(len(resolved_labels)),
        "labels": [str(label) for label in resolved_labels],
        "accuracy": _clean(accuracy_score(truth, prediction)),
        "precision_weighted": _clean(
            precision_score(truth, prediction, average="weighted", zero_division=0)
        ),
        "recall_weighted": _clean(
            recall_score(truth, prediction, average="weighted", zero_division=0)
        ),
        "f1_weighted": _clean(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "precision_macro": _clean(
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        "recall_macro": _clean(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
        "f1_macro": _clean(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": _clean(balanced_accuracy_score(truth, prediction)),
        "confusion_matrix": [[int(value) for value in row] for row in matrix],
    }

    per_class = []
    precision_per = precision_score(
        truth, prediction, average=None, zero_division=0, labels=resolved_labels
    )
    recall_per = recall_score(
        truth, prediction, average=None, zero_division=0, labels=resolved_labels
    )
    f1_per = f1_score(truth, prediction, average=None, zero_division=0, labels=resolved_labels)
    supports = matrix.sum(axis=1)
    for index, label in enumerate(resolved_labels):
        per_class.append(
            {
                "label": str(label),
                "precision": _clean(precision_per[index]),
                "recall": _clean(recall_per[index]),
                "f1": _clean(f1_per[index]),
                "support": int(supports[index]),
            }
        )
    metrics["per_class"] = per_class

    if y_proba is not None:
        probabilities = np.asarray(y_proba, dtype=float)
        try:
            if probabilities.shape[1] == 2:
                metrics["roc_auc"] = _clean(roc_auc_score(truth, probabilities[:, 1]))
            elif len(resolved_labels) > 2:
                metrics["roc_auc_ovr"] = _clean(
                    roc_auc_score(
                        truth, probabilities, multi_class="ovr", labels=resolved_labels
                    )
                )
        except Exception as exc:  # pragma: no cover - depends on class layout
            logger.debug("ROC-AUC could not be computed: %s", exc)
        try:
            metrics["log_loss"] = _clean(log_loss(truth, probabilities, labels=resolved_labels))
        except Exception as exc:  # pragma: no cover
            logger.debug("Log loss could not be computed: %s", exc)
    return metrics



def primary_metric_name(problem_type: str) -> str:
    return PRIMARY_METRIC.get(problem_type, "r2")


def primary_metric_value(metrics: Optional[Dict[str, Any]], problem_type: str) -> Optional[float]:
    """Value of the primary metric, or ``None`` when it is unavailable."""
    if not metrics:
        return None
    key = primary_metric_name(problem_type)
    value = metrics.get(key)
    if value is None and problem_type == "classification":
        value = metrics.get("accuracy")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def improvement(previous: Optional[float], current: Optional[float]) -> Dict[str, Any]:
    """Absolute and relative change of the primary metric."""
    if previous is None or current is None:
        return {"absolute": None, "relative": None}
    absolute = current - previous
    relative = (absolute / abs(previous)) if abs(previous) > 1e-9 else None
    return {
        "absolute": round(absolute, 4),
        "relative": round(relative, 4) if relative is not None else None,
    }


def metric_display_rows(
    metrics: Optional[Dict[str, Any]], problem_type: str
) -> List[Dict[str, Any]]:
    """Ordered metric rows used by the UI cards and tables."""
    if not metrics:
        return []
    if problem_type == "classification":
        keys = [
            ("accuracy", "Accuracy"),
            ("precision_weighted", "Precision (weighted)"),
            ("recall_weighted", "Recall (weighted)"),
            ("f1_weighted", "F1 (weighted)"),
            ("f1_macro", "F1 (macro)"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("roc_auc", "ROC-AUC"),
            ("log_loss", "Log loss"),
        ]
    else:
        keys = [
            ("r2", "R2 score"),
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("mse", "MSE"),
            ("mape", "MAPE (%)"),
        ]
    rows: List[Dict[str, Any]] = []
    for key, label in keys:
        if metrics.get(key) is not None:
            rows.append({"key": key, "label": label, "value": metrics[key]})
    return rows


def rank_experiments(
    experiments: Sequence[Dict[str, Any]],
    problem_type: str,
    priority: str = "balanced",
) -> List[Dict[str, Any]]:
    """Rank completed experiments according to the optimization priority.

    * ``performance``: highest primary metric wins.
    * ``speed``: among experiments within a small tolerance of the best score, the
      fastest one wins (measured training time is used).
    * ``balanced``: 70% normalised performance + 30% normalised speed.
    """
    primary = primary_metric_name(problem_type)
    usable = [
        item
        for item in experiments
        if item.get("status") == "completed" and item.get("primary_metric") is not None
    ]
    if not usable:
        return []

    times = [float(item.get("training_duration_seconds") or 0.0) for item in usable]
    positive_times = [value for value in times if value > 0]
    best_time = min(positive_times) if positive_times else 1.0
    worst_time = max([*positive_times, best_time])
    best_metric = max(float(item["primary_metric"]) for item in usable)
    tolerance = max(0.01, abs(best_metric) * 0.03)

    ranked: List[Dict[str, Any]] = []
    for item in usable:
        metric = float(item["primary_metric"])
        seconds = float(item.get("training_duration_seconds") or 0.0)
        within = metric >= best_metric - tolerance
        if priority == "performance":
            score = metric
            rationale = f"performance priority: {primary} = {metric:.4f} (highest wins)"
        elif priority == "speed":
            speed_component = 1.0 - (seconds / worst_time if worst_time > 0 else 0.0)
            score = metric + (0.5 * speed_component if within else 0.0)
            rationale = (
                f"speed priority: {primary} = {metric:.4f} "
                f"({'inside' if within else 'outside'} the {tolerance:.3f} tolerance band), "
                f"training took {seconds:.2f}s"
            )
        else:
            metric_component = metric / best_metric if best_metric > 0 else metric
            speed_component = 1.0 - (seconds / worst_time if worst_time > 0 else 0.0)
            score = 0.7 * metric_component + 0.3 * speed_component
            rationale = (
                f"balanced priority: 0.7 * normalised {primary} ({metric:.4f}) + "
                f"0.3 * normalised speed (trained in {seconds:.2f}s)"
            )
        entry = dict(item)
        entry["selection_score"] = round(float(score), 6)
        entry["selection_rationale"] = rationale
        entry["is_within_speed_tolerance"] = bool(within)
        ranked.append(entry)

    ranked.sort(key=lambda item: item["selection_score"], reverse=True)
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def best_by_metric(
    experiments: Sequence[Dict[str, Any]], problem_type: str
) -> Optional[Dict[str, Any]]:
    """Experiment with the highest primary metric (ignores training time)."""
    candidates = [
        item
        for item in experiments
        if item.get("status") == "completed" and item.get("primary_metric") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["primary_metric"]))


def summarize(metrics: Optional[Dict[str, Any]], problem_type: str) -> str:
    """One-line human summary such as ``R2 0.93 / MAE 12340``."""
    if not metrics:
        return "no metrics recorded"
    if problem_type == "classification":
        return (
            f"accuracy {metrics.get('accuracy')}, F1(weighted) {metrics.get('f1_weighted')}, "
            f"balanced accuracy {metrics.get('balanced_accuracy')}"
        )
    return f"R2 {metrics.get('r2')}, MAE {metrics.get('mae')}, RMSE {metrics.get('rmse')}"


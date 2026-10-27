"""Model training, splitting and model persistence.

Everything below is real scikit-learn training:

  * the dataset is split into train / validation / test *before* preprocessing;
  * preprocessing lives inside the ``Pipeline`` so it is fitted on the training
    split only (no leakage);
  * validation data drives model selection, the test split is only touched when
    the experiment session is finalized;
  * classification uses stratified splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from errors import TrainingError
from logging_config import get_logger
from services import evaluator
from services import model_selector as selector
from services import preprocessing, resource_monitor as rm

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Prepared feature matrices and the three-way split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    numeric_features: List[str]
    categorical_features: List[str]
    dropped_features: List[Dict[str, str]] = field(default_factory=list)
    classes: Optional[List[str]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.X_train) + len(self.X_val) + len(self.X_test)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_features": len(self.numeric_features) + len(self.categorical_features),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "dropped_features": list(self.dropped_features),
            "train_rows": len(self.X_train),
            "validation_rows": len(self.X_val),
            "test_rows": len(self.X_test),
            "n_classes": len(self.classes) if self.classes else None,
            "notes": list(self.notes),
        }


def prepare_features(
    frame: pd.DataFrame, target: str, problem_type: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[Dict[str, str]], List[str]]:
    """Clean the frame, coerce datetimes and decide the feature columns."""
    notes: List[str] = []
    dropped: List[Dict[str, str]] = []

    working = frame.copy()
    if len(working) > settings.max_rows:
        working = working.sample(settings.max_rows, random_state=settings.random_state)
        notes.append(
            f"Sampled {settings.max_rows:,} rows from {len(frame):,} to respect the configured "
            "hard limit for training."
        )

    target_series = working[target]
    if evaluator.PRIMARY_METRIC.get(problem_type) == "f1_weighted":
        target_series = target_series.astype("string")
    missing_target = int(target_series.isna().sum())
    if missing_target:
        keep = target_series.notna()
        working = working.loc[keep]
        target_series = target_series.loc[keep]
        notes.append(
            f"Dropped {missing_target:,} row(s) with a missing target value before training."
        )
    if len(working) < 20:
        raise TrainingError(
            "Fewer than 20 usable rows remain after removing missing targets; cannot train a model."
        )

    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for column in working.columns:
        if column == target:
            continue
        series = working[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            days = (series - series.min()).dt.total_seconds() / 86400.0
            new_name = f"{column}_days_since_start"
            working[new_name] = days.astype("float64")
            numeric_features.append(new_name)
            notes.append(f"Datetime column '{column}' converted to numeric '{new_name}'.")
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if series.nunique(dropna=True) <= 1:
                dropped.append({"column": str(column), "reason": "constant value"})
                continue
            numeric_features.append(str(column))
            continue
        unique = int(series.nunique(dropna=True))
        if unique <= 1:
            dropped.append({"column": str(column), "reason": "constant value"})
            continue
        if unique > 2_000:
            dropped.append(
                {
                    "column": str(column),
                    "reason": f"identifier-like text column with {unique:,} distinct values",
                }
            )
            continue
        categorical_features.append(str(column))
    # Text columns are cast to object dtype with np.nan for missing values so that
    # scikit-learn's SimpleImputer (which uses ``X != X`` to detect missing values)
    # can handle them. Pandas 3.x nullable "string"/"str" dtype stores missing
    # values as ``pd.NA``, and ``pd.NA != pd.NA`` returns ``pd.NA`` (not True),
    # which breaks sklearn's missing-value detection and raises
    # "boolean value of NA is ambiguous".
    if categorical_features:
        working = working.copy()
        for column in categorical_features:
            working[column] = working[column].astype("object")
            working[column] = working[column].fillna(np.nan)
    return working, target_series, numeric_features, categorical_features, dropped, notes


def encode_target(
    target_series: pd.Series, problem_type: str
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return numeric labels for classification and the original class names."""
    if problem_type == "classification":
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        values = encoder.fit_transform(target_series.astype("string"))
        return values.astype(int), [str(label) for label in encoder.classes_]
    return pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float), None


def split_dataset(
    frame: pd.DataFrame, target: str, problem_type: str, random_state: Optional[int] = None
) -> DatasetSplit:
    """Prepare features and create the train / validation / test split."""
    from sklearn.model_selection import train_test_split

    seed = settings.random_state if random_state is None else random_state
    working, target_series, numeric_features, categorical_features, dropped, notes = (
        prepare_features(frame, target, problem_type)
    )
    y, classes = encode_target(target_series, problem_type)
    X = working.drop(columns=[target])

    # Drop rows whose target became NaN during numeric coercion.
    if problem_type == "regression":
        valid = np.isfinite(y)
        if not valid.all():
            notes.append(f"Dropped {int((~valid).sum()):,} non-finite target value(s).")
        X, y = X.loc[valid], y[valid]

    if len(X) < 30:
        raise TrainingError(
            f"Only {len(X)} usable rows remain after cleaning; at least 30 are required."
        )

    stratify = y if problem_type == "classification" else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=seed, stratify=stratify
    )
    second_stratify = y_train_val if problem_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=settings.validation_size,
        random_state=seed,
        stratify=second_stratify,
    )
    notes.append(
        f"Split {len(X):,} rows into {len(X_train):,} train / {len(X_val):,} validation / "
        f"{len(X_test):,} test"
        + (
            " using stratified sampling because the task is classification."
            if problem_type == "classification"
            else "."
        )
    )
    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_features=dropped,
        classes=classes,
        notes=notes,
    )


def build_plan_for(
    model_key: str, split: DatasetSplit, n_rows_for_sparsity: Optional[int] = None
) -> preprocessing.PreprocessingPlan:
    spec = selector.get_spec(model_key)
    rows = n_rows_for_sparsity if n_rows_for_sparsity is not None else len(split.X_train)
    return preprocessing.plan_preprocessing(
        numeric_features=split.numeric_features,
        categorical_features=split.categorical_features,
        model_needs_scaling=spec.needs_scaling,
        n_rows=rows,
    )


def make_estimator(model_key: str, params: Dict[str, Any]) -> Any:
    spec = selector.get_spec(model_key)
    clean = {key: value for key, value in (params or {}).items() if value is not None}
    try:
        return spec.builder(clean)
    except Exception as exc:
        raise TrainingError(f"Could not instantiate {spec.label}: {exc}") from exc


def positive_probabilities(pipeline: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Probability matrix when the estimator supports it, else ``None``."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        probabilities = pipeline.predict_proba(X)
        return np.asarray(probabilities, dtype=float)
    except Exception as exc:  # pragma: no cover - model specific
        logger.debug("predict_proba failed: %s", exc)
        return None



def evaluate_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    problem_type: str,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    predictions = pipeline.predict(X)
    if problem_type == "classification":
        probabilities = positive_probabilities(pipeline, X)
        return evaluator.classification_metrics(
            y, predictions, y_proba=probabilities, labels=list(range(len(classes or []))) or None
        )
    return evaluator.regression_metrics(y, predictions)


def cross_validate(
    pipeline: Any,
    split: DatasetSplit,
    problem_type: str,
    folds: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Cross-validated score on the training split (preprocessing inside folds)."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    n_splits = folds or settings.cv_folds
    y_train = split.y_train
    if problem_type == "classification":
        smallest_class = int(np.bincount(y_train).min()) if len(y_train) else 0
        n_splits = min(n_splits, max(2, smallest_class))
        if smallest_class < 2:
            return None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.random_state)
        scoring = "f1_weighted"
    else:
        n_splits = min(n_splits, max(2, len(y_train) // 20))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=settings.random_state)
        scoring = "r2"
    try:
        scores = cross_val_score(pipeline, split.X_train, y_train, cv=cv, scoring=scoring)
    except Exception as exc:
        logger.warning("Cross-validation failed: %s", exc)
        return None
    return {
        "scoring": scoring,
        "folds": int(n_splits),
        "scores": [float(round(score, 4)) for score in scores],
        "mean": float(round(float(np.mean(scores)), 4)),
        "std": float(round(float(np.std(scores)), 4)),
    }


def run_experiment(
    split: DatasetSplit,
    model_key: str,
    params: Dict[str, Any],
    problem_type: str,
    with_cv: bool = False,
    time_budget_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Train + evaluate one model configuration (the core of every experiment)."""
    spec = selector.get_spec(model_key)
    plan = build_plan_for(model_key, split)
    estimator = make_estimator(model_key, params)
    pipeline = preprocessing.build_pipeline(estimator, plan)

    budget = time_budget_seconds or settings.max_experiment_seconds
    tracker = rm.ResourceTracker(label=f"{model_key}_train")
    with tracker as measured:
        try:
            pipeline.fit(split.X_train, split.y_train)
        except MemoryError as exc:
            raise TrainingError(
                f"{spec.label} ran out of memory on {len(split.X_train):,} training rows. "
                "Reduce the dataset size or choose a lighter model."
            ) from exc
        except Exception as exc:
            raise TrainingError(f"Training {spec.label} failed: {exc}") from exc
        validation_metrics = evaluate_pipeline(
            pipeline, split.X_val, split.y_val, problem_type, split.classes
        )
        measured.sample()

    if measured.usage.wall_seconds > budget * 1.5:
        logger.warning(
            "Experiment for %s took %.1fs which exceeds the per-experiment estimate of %.1fs",
            model_key,
            measured.usage.wall_seconds,
            budget,
        )

    cv_metrics = None
    if with_cv:
        cv_metrics = cross_validate(pipeline, split, problem_type)

    primary = evaluator.primary_metric_value(validation_metrics, problem_type)
    return {
        "pipeline": pipeline,
        "model_key": model_key,
        "model_name": spec.label,
        "params": params,
        "preprocessing_plan": plan.as_dict(),
        "validation_metrics": validation_metrics,
        "cv_metrics": cv_metrics,
        "primary_metric": primary,
        "training_seconds": float(round(measured.usage.wall_seconds, 4)),
        "resource_usage": measured.usage.as_dict(),
        "feature_importance": feature_importance(pipeline, limit=15),
    }


def finalize_model(
    split: DatasetSplit,
    model_key: str,
    params: Dict[str, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """Refit the winning configuration on train+validation and score the test set."""
    import pandas as pd  # local import keeps the module import light

    plan = build_plan_for(model_key, split)
    estimator = make_estimator(model_key, params)
    pipeline = preprocessing.build_pipeline(estimator, plan)

    X_combined = pd.concat([split.X_train, split.X_val], axis=0)
    y_combined = np.concatenate([split.y_train, split.y_val])

    tracker = rm.ResourceTracker(label=f"{model_key}_final_fit")
    with tracker as measured:
        pipeline.fit(X_combined, y_combined)
        test_metrics = evaluate_pipeline(
            pipeline, split.X_test, split.y_test, problem_type, split.classes
        )
        measured.sample()

    return {
        "pipeline": pipeline,
        "test_metrics": test_metrics,
        "training_seconds": float(round(measured.usage.wall_seconds, 4)),
        "resource_usage": measured.usage.as_dict(),
        "train_rows": int(len(X_combined)),
        "test_rows": int(len(split.X_test)),
    }



def feature_importance(pipeline: Any, limit: int = 15) -> List[Dict[str, Any]]:
    """Extract the most influential transformed features (best effort)."""
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        names = list(preprocessor.get_feature_names_out())
    except Exception as exc:  # pragma: no cover - model specific
        logger.debug("Feature names unavailable: %s", exc)
        return []

    values: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)

    if values is None or values.size != len(names):
        return []
    order = np.argsort(values)[::-1][:limit]
    total = float(np.sum(np.abs(values))) or 1.0
    return [
        {
            "feature": str(names[index]).replace("numeric__", "").replace("categorical__", ""),
            "transformed_name": str(names[index]),
            "importance": float(round(values[index], 6)),
            "share_pct": float(round(values[index] / total * 100, 2)),
        }
        for index in order
    ]


def persist_model(pipeline: Any, model_key: str, experiment_id: str) -> str:
    """Serialise a fitted pipeline with joblib and return the artifact path."""
    import joblib

    directory = Path(settings.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_key}_{experiment_id}.joblib"
    joblib.dump(
        {
            "model_key": model_key,
            "experiment_id": experiment_id,
            "pipeline": pipeline,
        },
        path,
    )
    logger.info("Saved trained model to %s", path)
    return str(path)


def load_model(path: str) -> Any:
    import joblib

    return joblib.load(path)


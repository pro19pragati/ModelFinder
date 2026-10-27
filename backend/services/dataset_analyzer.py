"""Automatic dataset analysis (pandas + numpy + scikit-learn probes).

The analyzer inspects the uploaded dataset and produces:
  * structural summary (rows, columns, dtypes, memory usage)
  * missing values, duplicate rows, unique values / cardinality
  * numeric + categorical distribution summaries
  * target profile, problem-type detection and class-imbalance analysis
  * feature-target relationships and a correlation matrix
  * outlier detection (IQR per numeric column)
  * an *approximate linearity* probe for regression targets (linear vs tree CV)
  * human-readable "Agent's Dataset Insights" generated from the real numbers

Nothing in this module is hard-coded to a particular dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from errors import DatasetValidationError, ResourceLimitError
from logging_config import get_logger
from services import resource_monitor as rm

logger = get_logger(__name__)


# --------------------------------------------------------------------- helpers
def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_boolean(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)


def is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def column_kind(series: pd.Series) -> str:
    """Coarse semantic kind used by the UI and the preprocessing planner."""
    if is_datetime(series):
        return "datetime"
    if is_boolean(series):
        return "boolean"
    if is_numeric(series):
        return "numeric"
    unique = int(series.nunique(dropna=True))
    # Low-cardinality text behaves as categorical; high-cardinality text is
    # flagged separately because one-hot encoding it is expensive.
    return "categorical" if unique <= 50 else "text"


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_numeric(df[c])]


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not is_numeric(df[c]) and not is_datetime(df[c])]


def _round(value: Any, digits: int = 4) -> Optional[float]:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return round(as_float, digits)


def _safe_float(value: Any) -> Optional[float]:
    return _round(value, 6)



# ---------------------------------------------------------------- file loading
def load_dataframe(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV/XLSX file into a DataFrame with helpful error messages."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetValidationError(f"File not found: {file_path.name}", code="file_missing")
    if file_path.stat().st_size == 0:
        raise DatasetValidationError(
            "The uploaded file is empty (0 bytes).", code="empty_file"
        )

    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = _read_csv_robust(file_path, nrows)
        elif suffix in {".xlsx", ".xlsm"}:
            frame = pd.read_excel(file_path, nrows=nrows, engine="openpyxl")
        elif suffix == ".xls":
            frame = pd.read_excel(file_path, nrows=nrows)  # requires xlrd
        else:
            raise DatasetValidationError(
                f"Unsupported file format '{suffix}'. Supported formats: "
                + ", ".join(settings.supported_extensions),
                code="unsupported_format",
            )
    except DatasetValidationError:
        raise
    except ImportError as exc:  # missing optional engine (e.g. xlrd for .xls)
        raise DatasetValidationError(
            f"Reading '{suffix}' files requires an extra package: {exc}",
            code="missing_reader",
        ) from exc
    except Exception as exc:  # corrupt / undecodable file
        raise DatasetValidationError(
            f"The file could not be parsed: {exc}", code="invalid_file"
        ) from exc

    if frame is None or frame.empty or len(frame.columns) == 0:
        raise DatasetValidationError(
            "The dataset contains no rows or no columns.", code="empty_dataset"
        )
    return frame


def _read_csv_robust(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, nrows=nrows, encoding=encoding, low_memory=False)
            # A single column usually means the delimiter was guessed wrong.
            if frame.shape[1] == 1 and nrows is None:
                sample = path.read_text(encoding=encoding, errors="ignore").splitlines()[:3]
                for delimiter in (";", "\t", "|"):
                    if any(delimiter in line for line in sample):
                        return pd.read_csv(
                            path, sep=delimiter, encoding=encoding, low_memory=False
                        )
            return frame
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc
        except Exception as exc:
            last_error = exc
            break
    raise DatasetValidationError(
        f"Unable to read CSV file: {last_error}", code="invalid_file"
    )


def validate_shape(n_rows: int, n_columns: int, file_size_mb: float) -> None:
    """Raise :class:`ResourceLimitError` when hard limits are exceeded."""
    violations = rm.check_hard_limits(n_rows, n_columns, file_size_mb)
    if violations:
        raise ResourceLimitError(
            "The dataset exceeds the configured resource limits: "
            + " ".join(v["message"] for v in violations),
            violations=violations,
        )
    if n_rows < 10:
        raise DatasetValidationError(
            f"The dataset only has {n_rows} rows; at least 10 are needed to train a model.",
            code="too_few_rows",
        )



# --------------------------------------------------- target / problem detection
def target_candidates(df: pd.DataFrame, limit: int = 60) -> List[str]:
    """Rank plausible target columns (later columns usually hold the label)."""
    scored: List[Tuple[float, str]] = []
    n_rows = max(1, len(df))
    for position, column in enumerate(df.columns):
        series = df[column]
        unique = int(series.nunique(dropna=True))
        if unique <= 1:
            continue
        uniqueness = unique / n_rows
        if uniqueness > 0.98 and unique > 10:
            continue  # almost certainly an identifier
        score = 0.0
        name = str(column).lower()
        if any(
            token in name
            for token in ("target", "label", "class", "outcome", "price", "churn", "sales", "score", "y")
        ):
            score += 3.0
        if not is_numeric(series) or unique <= 20:
            score += 1.0
        score += position / max(1, len(df.columns))  # later columns rank higher
        if uniqueness < 0.02:
            score += 0.5
        scored.append((score, str(column)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


def detect_problem_type(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Decide classification vs regression from dtype + target cardinality."""
    if target not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target}' is not present in the dataset.",
            code="missing_target",
        )
    series = df[target]
    n_rows = int(len(series))
    unique = int(series.nunique(dropna=True))
    missing = int(series.isna().sum())
    numeric = is_numeric(series)
    integer_like = bool(
        numeric and np.all(np.mod(series.dropna().to_numpy(dtype=float), 1) == 0)
    )

    reasons: List[str] = []
    ambiguous = False

    if not numeric:
        problem_type = "classification"
        confidence = "high"
        reasons.append(
            f"Target '{target}' has non-numeric dtype '{series.dtype}', so the task is treated "
            "as classification."
        )
    elif unique <= 2:
        problem_type = "classification"
        confidence = "high"
        reasons.append(f"Target '{target}' takes only {unique} distinct values (binary target).")
    elif unique <= 10 and integer_like:
        problem_type = "classification"
        confidence = "medium"
        reasons.append(
            f"Target '{target}' is integer-valued with only {unique} distinct values, which "
            "behaves like discrete class labels."
        )
    elif unique <= 20 and integer_like and unique / max(1, n_rows) < 0.02:
        problem_type = "classification"
        confidence = "medium"
        ambiguous = True
        reasons.append(
            f"Target '{target}' has {unique} distinct integer values covering only "
            f"{unique / n_rows * 100:.2f}% of rows — plausible class labels, but the choice is "
            "ambiguous (a user override is allowed)."
        )
    else:
        problem_type = "regression"
        confidence = "high" if unique > 50 else "medium"
        ambiguous = unique <= 20
        reasons.append(
            f"Target '{target}' is numeric with {unique:,} distinct values, so a continuous "
            "prediction task (regression) is assumed."
        )

    if missing:
        reasons.append(
            f"{missing:,} target value(s) ({missing / n_rows * 100:.2f}% of rows) are missing and "
            "will be dropped before training."
        )
    if unique >= n_rows and n_rows > 10:
        reasons.append(
            f"Target '{target}' is unique for every row, which suggests it may be an identifier "
            "rather than a value to predict."
        )
    if problem_type == "classification" and unique > 2:
        reasons.append(f"{unique} classes detected - handled as a multiclass problem.")

    return {
        "problem_type": problem_type,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "n_unique": unique,
        "dtype": str(series.dtype),
        "integer_like": integer_like,
        "missing": missing,
        "reasons": reasons,
    }


# -------------------------------------------------------------- column profiles
def profile_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    n_rows = max(1, len(df))
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        profiles.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "kind": column_kind(series),
                "missing": missing,
                "missing_pct": round(missing / n_rows * 100, 2),
                "unique": unique,
                "cardinality_ratio": round(unique / n_rows, 4),
                "is_constant": unique <= 1,
            }
        )
    return profiles


def missing_value_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = max(1, len(df))
    per_column = [
        {
            "column": str(column),
            "missing": int(df[column].isna().sum()),
            "missing_pct": round(float(df[column].isna().sum()) / n_rows * 100, 2),
            "dtype": str(df[column].dtype),
        }
        for column in df.columns
    ]
    per_column.sort(key=lambda item: item["missing"], reverse=True)
    total_missing = int(sum(item["missing"] for item in per_column))
    return {
        "total_missing": total_missing,
        "total_cells": int(n_rows * len(df.columns)),
        "total_missing_pct": round(total_missing / max(1, n_rows * len(df.columns)) * 100, 2),
        "columns_with_missing": int(sum(1 for item in per_column if item["missing"] > 0)),
        "by_column": per_column,
    }


def duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = max(1, len(df))
    duplicates = int(df.duplicated().sum())
    return {
        "duplicate_rows": duplicates,
        "duplicate_pct": round(duplicates / n_rows * 100, 2),
        "unique_rows": int(n_rows - duplicates),
    }


def memory_report(df: pd.DataFrame) -> Dict[str, Any]:
    total_bytes = int(df.memory_usage(deep=True).sum())
    per_column = [
        {"column": str(column), "memory_kb": round(float(size) / 1024, 2)}
        for column, size in df.memory_usage(deep=True).items()
        if column != "Index"
    ]
    per_column.sort(key=lambda item: item["memory_kb"], reverse=True)
    return {
        "total_mb": round(total_bytes / (1024 ** 2), 3),
        "total_bytes": total_bytes,
        "per_column": per_column[:30],
    }


def numeric_summary(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for column in columns:
        series = df[column].dropna()
        if series.empty:
            continue
        described = series.describe()
        summaries.append(
            {
                "column": str(column),
                "count": int(described.get("count", 0)),
                "mean": _round(described.get("mean")),
                "std": _round(described.get("std")),
                "min": _round(described.get("min")),
                "q25": _round(described.get("25%")),
                "median": _round(described.get("50%")),
                "q75": _round(described.get("75%")),
                "max": _round(described.get("max")),
                "skew": _round(series.skew()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 2),
            }
        )
    return summaries


def categorical_summary(df: pd.DataFrame, columns: Sequence[str], limit: int = 12) -> List[Dict[str, Any]]:
    n_rows = max(1, len(df))
    summaries: List[Dict[str, Any]] = []
    for column in columns:
        series = df[column]
        counts = series.astype("string").value_counts(dropna=True)
        if counts.empty:
            top, top_freq = None, 0
        else:
            top = str(counts.index[0])
            top_freq = int(counts.iloc[0])
        rare = int((counts < max(2, int(0.01 * n_rows))).sum())
        summaries.append(
            {
                "column": str(column),
                "unique": int(series.nunique(dropna=True)),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "top": top,
                "top_frequency": top_freq,
                "top_pct": round(top_freq / n_rows * 100, 2),
                "rare_categories": rare,
                "top_values": [
                    {"value": str(index), "count": int(value)}
                    for index, value in counts.head(limit).items()
                ],
            }
        )
    return summaries



# ---------------------------------------------------------------- target profile
def target_profile(df: pd.DataFrame, target: str, problem_type: str) -> Dict[str, Any]:
    series = df[target]
    n_rows = max(1, len(series))
    profile: Dict[str, Any] = {
        "column": target,
        "dtype": str(series.dtype),
        "problem_type": problem_type,
        "missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "n_unique": int(series.nunique(dropna=True)),
    }

    if problem_type == "classification":
        counts = series.astype("string").value_counts(dropna=True)
        total = int(counts.sum()) or 1
        class_counts = [
            {
                "label": str(index),
                "count": int(value),
                "pct": round(int(value) / total * 100, 2),
            }
            for index, value in counts.items()
        ]
        if class_counts:
            majority = max(item["count"] for item in class_counts)
            minority = min(item["count"] for item in class_counts)
            baseline = majority / total
        else:
            majority = minority = 0
            baseline = 0.0
        profile["class_counts"] = class_counts
        profile["n_classes"] = len(class_counts)
        profile["majority_pct"] = round((majority / total) * 100, 2) if total else 0.0
        profile["class_balance"] = {
            "is_imbalanced": bool(minority and majority / max(1, minority) >= 1.5),
            "imbalance_ratio": round(majority / max(1, minority), 2),
            "majority_class": max(class_counts, key=lambda item: item["count"])["label"]
            if class_counts
            else None,
            "minority_class": min(class_counts, key=lambda item: item["count"])["label"]
            if class_counts
            else None,
            "majority_baseline_accuracy": round(baseline, 4),
            "is_binary": len(class_counts) == 2,
            "n_classes": len(class_counts),
        }
    else:
        values = pd.to_numeric(series, errors="coerce").dropna()
        profile["stats"] = {
            "mean": _round(values.mean()),
            "std": _round(values.std()),
            "min": _round(values.min()),
            "q25": _round(values.quantile(0.25)) if not values.empty else None,
            "median": _round(values.median()),
            "q75": _round(values.quantile(0.75)) if not values.empty else None,
            "max": _round(values.max()),
            "skew": _round(values.skew()),
            "iqr": _round(values.quantile(0.75) - values.quantile(0.25))
            if not values.empty
            else None,
        }
        profile["histogram"] = histogram(values, bins=24)
    return profile


def histogram(series: pd.Series, bins: int = 20) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"bins": [], "counts": [], "edges": []}
    counts, edges = np.histogram(values, bins=min(bins, max(1, values.size)))
    return {
        "bins": [round(float(edge), 4) for edge in edges[:-1]],
        "edges": [round(float(edge), 4) for edge in edges],
        "counts": [int(value) for value in counts],
    }


def feature_distributions(
    df: pd.DataFrame, columns: Sequence[str], limit: int = 12
) -> List[Dict[str, Any]]:
    distributions: List[Dict[str, Any]] = []
    for column in list(columns)[:limit]:
        series = df[column]
        if is_numeric(series):
            values = series.dropna()
            distributions.append(
                {
                    "column": str(column),
                    "kind": "numeric",
                    "histogram": histogram(values, bins=20),
                    "mean": _round(values.mean()),
                    "median": _round(values.median()),
                }
            )
        else:
            counts = series.astype("string").value_counts(dropna=True).head(10)
            distributions.append(
                {
                    "column": str(column),
                    "kind": "categorical",
                    "counts": [
                        {"label": str(index), "count": int(value)}
                        for index, value in counts.items()
                    ],
                }
            )
    return distributions



# ----------------------------------------------------- correlations & relations
def correlation_report(
    df: pd.DataFrame, target: str, numeric_cols: Sequence[str], max_columns: int = 20
) -> Dict[str, Any]:
    candidates = [c for c in numeric_cols if c != target]
    if target in df.columns and is_numeric(df[target]):
        candidates = list(candidates) + [target]
    candidates = candidates[:max_columns]
    if len(candidates) < 2:
        return {"columns": [], "matrix": [], "top_target_correlations": [], "method": "pearson"}

    subset = df[candidates].apply(pd.to_numeric, errors="coerce")
    corr = subset.corr(numeric_only=True, method="pearson")
    matrix = [
        [_round(value, 3) if _round(value, 3) is not None else 0.0 for value in row]
        for row in corr.to_numpy(dtype=float)
    ]
    result: Dict[str, Any] = {
        "columns": [str(c) for c in corr.columns],
        "matrix": matrix,
        "method": "pearson",
        "top_pairs": [],
    }
    if target in corr.columns:
        target_corr = corr[target].drop(labels=[target], errors="ignore")
        result["top_target_correlations"] = [
            {"feature": str(index), "correlation": _round(value, 3)}
            for index, value in target_corr.sort_values(
                key=lambda s: s.abs(), ascending=False
            ).items()
        ][:15]

    pairs: List[Dict[str, Any]] = []
    column_list = [c for c in corr.columns if c != target]
    for position, first in enumerate(column_list):
        for second in column_list[position + 1:]:
            value = _round(corr.loc[first, second], 3)
            if value is not None:
                pairs.append({"a": str(first), "b": str(second), "correlation": value})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    result["top_pairs"] = pairs[:10]
    return result


def _eta_squared(categories: pd.Series, values: pd.Series) -> Optional[float]:
    """Correlation ratio (sqrt of eta squared) between a categorical and a numeric."""
    try:
        frame = pd.DataFrame(
            {
                "cat": categories.astype("string"),
                "val": pd.to_numeric(values, errors="coerce"),
            }
        ).dropna()
        if frame.empty:
            return None
        grand_mean = frame["val"].mean()
        groups = frame.groupby("cat", observed=True)["val"]
        ss_between = float((groups.count() * (groups.mean() - grand_mean) ** 2).sum())
        ss_total = float(((frame["val"] - grand_mean) ** 2).sum())
        if ss_total <= 0:
            return None
        return round(math.sqrt(max(0.0, ss_between / ss_total)), 4)
    except Exception:  # pragma: no cover - defensive
        return None



def feature_target_relationships(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Quantify how strongly each feature relates to the target."""
    relationships: List[Dict[str, Any]] = []
    target_series = df[target]

    def classify_strength(score: float) -> str:
        if score >= 0.6:
            return "strong"
        return "moderate" if score >= 0.3 else "weak"

    if is_numeric(target_series):
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        for column in numeric_cols:
            if column == target:
                continue
            pair = pd.DataFrame(
                {"f": pd.to_numeric(df[column], errors="coerce"), "t": target_numeric}
            ).dropna()
            if len(pair) < 3:
                continue
            corr = float(pair["f"].corr(pair["t"]))
            if math.isnan(corr):
                continue
            relationships.append(
                {
                    "feature": str(column),
                    "feature_kind": "numeric",
                    "measure": "pearson_r",
                    "value": _round(corr, 3),
                    "strength": round(abs(corr), 3),
                    "relationship": classify_strength(abs(corr)),
                    "direction": "positive" if corr >= 0 else "negative",
                }
            )
        for column in categorical_cols:
            score = _eta_squared(df[column], target_numeric)
            if score is None:
                continue
            relationships.append(
                {
                    "feature": str(column),
                    "feature_kind": "categorical",
                    "measure": "correlation_ratio",
                    "value": score,
                    "strength": score,
                    "relationship": "strong" if score >= 0.5 else "moderate" if score >= 0.25 else "weak",
                    "direction": None,
                }
            )
    else:
        labels = target_series.astype("string")
        for column in numeric_cols:
            score = _eta_squared(labels, pd.to_numeric(df[column], errors="coerce"))
            if score is None:
                continue
            relationships.append(
                {
                    "feature": str(column),
                    "feature_kind": "numeric",
                    "measure": "correlation_ratio",
                    "value": score,
                    "strength": score,
                    "relationship": "strong" if score >= 0.5 else "moderate" if score >= 0.25 else "weak",
                    "direction": None,
                }
            )
        for column in categorical_cols:
            score = _eta_squared(labels, df[column])
            if score is None:
                continue
            relationships.append(
                {
                    "feature": str(column),
                    "feature_kind": "categorical",
                    "measure": "correlation_ratio",
                    "value": score,
                    "strength": score,
                    "relationship": "strong" if score >= 0.5 else "moderate" if score >= 0.25 else "weak",
                    "direction": None,
                }
            )

    relationships.sort(key=lambda item: item["strength"], reverse=True)
    return relationships[:limit]



# ----------------------------------------------------------------- outliers
def outlier_report(df: pd.DataFrame, columns: Sequence[str], max_columns: int = 30) -> Dict[str, Any]:
    """IQR-based outlier detection for numeric columns."""
    details: List[Dict[str, Any]] = []
    n_rows = max(1, len(df))
    total_outlier_rows = 0
    for column in list(columns)[:max_columns]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            lower, upper = q1, q3
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        if count == 0:
            continue
        total_outlier_rows += count
        details.append(
            {
                "column": str(column),
                "count": count,
                "pct": round(count / n_rows * 100, 2),
                "lower_bound": _round(lower),
                "upper_bound": _round(upper),
                "min_outlier": _round(series[mask].min()),
                "max_outlier": _round(series[mask].max()),
                "method": "IQR (1.5x)",
            }
        )
    details.sort(key=lambda item: item["count"], reverse=True)
    return {
        "total_outlier_values": total_outlier_rows,
        "columns_with_outliers": len(details),
        "by_column": details[:20],
        "method": "Tukey IQR fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR)",
    }


# ------------------------------------------------------- linearity probe
def linearity_probe(
    df: pd.DataFrame, target: str, numeric_cols: Sequence[str], random_state: int
) -> Optional[Dict[str, Any]]:
    """Cheaply compare a linear fit against a shallow tree fit on a subsample.

    If a linear model reaches (almost) the same cross-validated R² as a
    non-linear tree model, the feature/target relationships can be considered
    approximately linear — which is the signal the model selector uses to pick
    Linear / Ridge Regression as the first candidate.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    features = [c for c in numeric_cols if c != target]
    if not features or target not in df.columns:
        return None

    frame = df[features + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target])
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Drop features that are entirely empty after coercion.
    frame = frame.dropna(axis=1, how="all")
    features = [c for c in frame.columns if c != target]
    if len(frame) < 30 or not features:
        return None

    if len(frame) > settings.linearity_sample_rows:
        frame = frame.sample(settings.linearity_sample_rows, random_state=random_state)

    X = frame[features].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    y = np.nan_to_num(y)

    linear_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
         ("model", LinearRegression())]
    )
    tree_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")),
         ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state))]
    )
    folds = KFold(n_splits=min(3, max(2, len(frame) // 20)), shuffle=True, random_state=random_state)
    try:
        linear_scores = cross_val_score(linear_pipe, X, y, cv=folds, scoring="r2")
        tree_scores = cross_val_score(tree_pipe, X, y, cv=folds, scoring="r2")
    except Exception as exc:  # pragma: no cover - probe is best-effort
        logger.warning("Linearity probe failed: %s", exc)
        return None

    linear_r2 = float(np.mean(linear_scores))
    tree_r2 = float(np.mean(tree_scores))
    best = max(linear_r2, tree_r2, 1e-6)
    ratio = max(0.0, min(2.0, linear_r2 / best)) if best > 0 else 0.0
    gap = tree_r2 - linear_r2
    approximately_linear = bool(linear_r2 > 0.25 and ratio >= 0.9 and gap <= 0.08)
    return {
        "method": "3-fold cross-validated R2 on a sample (<=2000 rows)",
        "sample_rows": int(len(frame)),
        "n_features": len(features),
        "linear_cv_r2": _round(linear_r2),
        "tree_cv_r2": _round(tree_r2),
        "linear_to_tree_ratio": _round(ratio, 3),
        "nonlinearity_gap": _round(gap),
        "is_approximately_linear": approximately_linear,
        "verdict": (
            "approximately linear" if approximately_linear
            else "non-linear structure present" if tree_r2 > linear_r2 + 0.08
            else "weak linear signal"
        ),
    }



# ---------------------------------------------------------------- top-level API
def analyze_dataframe(
    df: pd.DataFrame,
    target: str,
    problem_type_override: Optional[str] = None,
    file_size_mb: float = 0.0,
) -> Dict[str, Any]:
    """Run the full analysis and return a JSON-serialisable dictionary."""
    if target not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target}' was not found in the dataset. Available columns: "
            + ", ".join(str(c) for c in df.columns[:30]),
            code="missing_target",
        )

    validate_shape(len(df), len(df.columns), file_size_mb)

    detection = detect_problem_type(df, target)
    problem_type = (problem_type_override or detection["problem_type"]).lower()
    source = "user" if problem_type_override else "agent"

    numeric_cols = [c for c in numeric_columns(df) if c != target]
    categorical_cols = [c for c in categorical_columns(df) if c != target]
    if not numeric_cols and not categorical_cols:
        raise DatasetValidationError(
            "The dataset has no usable feature columns besides the target.",
            code="no_features",
        )

    columns = profile_columns(df)
    missing = missing_value_report(df)
    duplicates = duplicate_report(df)
    memory = memory_report(df)
    target_info = target_profile(df, target, problem_type)
    correlations = correlation_report(df, target, numeric_cols)
    relationships = feature_target_relationships(
        df, target, problem_type, numeric_cols, categorical_cols
    )
    outliers = outlier_report(df, numeric_cols)
    linearity = None
    if problem_type == "regression":
        linearity = linearity_probe(df, target, numeric_cols, settings.random_state)

    onehot_width = sum(
        min(int(df[c].nunique(dropna=True)) + 1, settings.max_onehot_categories)
        for c in categorical_cols
    )
    profile = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_features": len(numeric_cols) + len(categorical_cols),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "n_classes": int(target_info.get("n_classes") or 0),
        "missing_ratio": round(
            float(missing["total_missing"]) / max(1, missing["total_cells"]), 4
        ),
        "duplicate_ratio": round(
            float(duplicates["duplicate_rows"]) / max(1, len(df)), 4
        ),
        "estimated_onehot_width": int(onehot_width + len(numeric_cols)),
        "has_high_cardinality": any(
            int(df[c].nunique(dropna=True)) > settings.max_onehot_categories
            for c in categorical_cols
        ),
        "is_imbalanced": bool(
            problem_type == "classification"
            and target_info.get("class_balance", {}).get("is_imbalanced")
        ),
        "linearity_score": (linearity or {}).get("linear_to_tree_ratio"),
    }

    analysis: Dict[str, Any] = {
        "target": target,
        "problem_type": problem_type,
        "problem_type_source": source,
        "problem_type_detection": detection,
        "problem_type_overridden": bool(problem_type_override),
        "shape": {"n_rows": int(len(df)), "n_columns": int(len(df.columns))},
        "columns": columns,
        "columns_by_kind": {
            "numeric": [str(c) for c in numeric_cols],
            "categorical": [str(c) for c in categorical_cols],
            "datetime": [str(c) for c in df.columns if is_datetime(df[c])],
        },
        "missing_values": missing,
        "duplicates": duplicates,
        "memory": memory,
        "target_profile": target_info,
        "correlations": correlations,
        "feature_target_relationships": relationships,
        "outliers": outliers,
        "linearity": linearity,
        "numeric_summary": numeric_summary(df, numeric_cols),
        "categorical_summary": categorical_summary(df, categorical_cols),
        "feature_distributions": feature_distributions(
            df, list(numeric_cols)[:8] + list(categorical_cols)[:4]
        ),
        "profile": profile,
        "resource_snapshot": rm.resource_snapshot(),
        "file_size_mb": round(file_size_mb, 3),
    }
    analysis["warnings"] = build_warnings(analysis)
    return enrich_insights(analysis)


# ------------------------------------------------------------ agent insights
def build_insights(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Explain the findings in plain language, using only the measured numbers."""
    insights: List[Dict[str, str]] = []
    shape = analysis["shape"]
    profile = analysis["profile"]
    target_info = analysis["target_profile"]
    detection = analysis["problem_type_detection"]
    problem_type = analysis["problem_type"]

    insights.append(
        {
            "title": "Dataset overview",
            "detail": (
                f"The dataset contains {shape['n_rows']:,} rows and {shape['n_columns']} columns "
                f"({profile['n_numeric_features']} numeric and {profile['n_categorical_features']} "
                f"categorical feature(s)). {analysis['memory']['total_mb']} MB of memory is used "
                "once loaded."
            ),
            "level": "info",
        }
    )

    insights.append(
        {
            "title": "Problem type detected",
            "detail": (detection["reasons"][0] if detection["reasons"] else "")
            + (
                " The detected task was overridden by the user."
                if analysis.get("problem_type_overridden")
                else f" Confidence: {detection['confidence']}."
            ),
            "level": "success",
        }
    )
    if detection.get("ambiguous"):
        insights.append(
            {
                "title": "Ambiguous target",
                "detail": (
                    "The target column could reasonably be treated either as a discrete label or "
                    "as a continuous value. The current choice is "
                    f"'{problem_type}' and can be corrected on the analysis page."
                ),
                "level": "warning",
            }
        )

    missing = analysis["missing_values"]
    if missing["total_missing"] == 0:
        insights.append(
            {
                "title": "Data completeness",
                "detail": "No missing values were found, so no imputation is required.",
                "level": "success",
            }
        )
    else:
        worst = missing["by_column"][0]
        insights.append(
            {
                "title": "Missing values",
                "detail": (
                    f"{missing['total_missing']:,} missing cells "
                    f"({missing['total_missing_pct']}% of the data) across "
                    f"{missing['columns_with_missing']} column(s). The worst column is "
                    f"'{worst['column']}' at {worst['missing_pct']}%. Numeric features are "
                    "median-imputed and categorical features mode-imputed inside the pipeline."
                ),
                "level": "warning" if missing["total_missing_pct"] > 10 else "info",
            }
        )

    duplicates = analysis["duplicates"]
    if duplicates["duplicate_rows"]:
        insights.append(
            {
                "title": "Duplicate rows",
                "detail": (
                    f"{duplicates['duplicate_rows']:,} duplicated rows "
                    f"({duplicates['duplicate_pct']}%) were detected. They are kept, but they can "
                    "inflate validation scores."
                ),
                "level": "warning",
            }
        )
    else:
        insights.append(
            {
                "title": "Duplicate rows",
                "detail": "No duplicate rows were detected.",
                "level": "success",
            }
        )
    return insights



def build_target_insights(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Target-focused insights appended after :func:`build_insights`."""
    insights: List[Dict[str, str]] = []
    target_info = analysis["target_profile"]
    problem_type = analysis["problem_type"]

    if problem_type == "classification" and "class_balance" in target_info:
        balance = target_info["class_balance"]
        if balance.get("is_imbalanced"):
            insights.append(
                {
                    "title": "Class imbalance",
                    "detail": (
                        f"The classes are imbalanced: '{balance['majority_class']}' is the majority "
                        f"class and '{balance['minority_class']}' the minority class, a ratio of "
                        f"{balance['imbalance_ratio']}:1. A model that always predicts the majority "
                        f"class would already reach "
                        f"{balance['majority_baseline_accuracy'] * 100:.1f}% accuracy, so weighted "
                        "F1 is used as the primary metric and all splits are stratified."
                    ),
                    "level": "warning",
                }
            )
        else:
            insights.append(
                {
                    "title": "Class balance",
                    "detail": (
                        f"{target_info.get('n_classes', 0)} classes are present and reasonably "
                        "balanced, so accuracy and weighted F1 will agree closely."
                    ),
                    "level": "success",
                }
            )
    elif problem_type == "regression":
        stats = target_info.get("stats", {})
        insights.append(
            {
                "title": "Target distribution",
                "detail": (
                    f"The target spans {stats.get('min')} to {stats.get('max')} with a mean of "
                    f"{stats.get('mean')} and a standard deviation of {stats.get('std')}. "
                    f"Skewness is {stats.get('skew')}."
                ),
                "level": "info",
            }
        )
        linearity = analysis.get("linearity")
        if linearity:
            insights.append(
                {
                    "title": "Linearity check",
                    "detail": (
                        f"A linear baseline scored a cross-validated R2 of "
                        f"{linearity['linear_cv_r2']} while a depth-6 decision tree reached "
                        f"{linearity['tree_cv_r2']} on the same sample, so the relationships look "
                        f"{linearity['verdict']}."
                    ),
                    "level": "success" if linearity["is_approximately_linear"] else "info",
                }
            )
    return insights



def build_relationship_insights(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Insights about feature relationships, outliers and encoding cost."""
    insights: List[Dict[str, str]] = []
    relationships = analysis.get("feature_target_relationships") or []
    if relationships:
        described = ", ".join(
            f"{item['feature']} ({item['measure']}={item['value']}, {item['relationship']})"
            for item in relationships[:3]
        )
        insights.append(
            {
                "title": "Strongest feature relationships",
                "detail": "The features most associated with the target are " + described + ".",
                "level": "info",
            }
        )
    weak = [item for item in relationships if item["strength"] < 0.15]
    if len(weak) >= 3:
        insights.append(
            {
                "title": "Low-signal features",
                "detail": (
                    f"{len(weak)} feature(s) show almost no association with the target "
                    "(strength < 0.15), for example "
                    f"{', '.join(item['feature'] for item in weak[:4])}. They are still used, but "
                    "they may add noise."
                ),
                "level": "info",
            }
        )

    outliers = analysis.get("outliers") or {}
    if outliers.get("columns_with_outliers"):
        worst = outliers["by_column"][0]
        insights.append(
            {
                "title": "Outliers",
                "detail": (
                    f"{outliers['total_outlier_values']:,} extreme values were flagged across "
                    f"{outliers['columns_with_outliers']} numeric column(s) using the IQR rule. The "
                    f"most affected column is '{worst['column']}' ({worst['count']:,} values, "
                    f"{worst['pct']}% of rows). Tree-based models tolerate these; linear models are "
                    "more sensitive."
                ),
                "level": "info",
            }
        )

    correlations = analysis.get("correlations") or {}
    pairs = correlations.get("top_pairs") or []
    if pairs and abs(pairs[0]["correlation"]) >= 0.85:
        insights.append(
            {
                "title": "Highly correlated features",
                "detail": (
                    f"'{pairs[0]['a']}' and '{pairs[0]['b']}' are correlated at "
                    f"{pairs[0]['correlation']}. Multicollinearity like this can destabilise plain "
                    "linear models, which is why Ridge Regression is considered as an alternative."
                ),
                "level": "warning",
            }
        )

    categorical = (analysis.get("columns_by_kind") or {}).get("categorical") or []
    if categorical:
        cardinalities = sorted(
            (
                (column["name"], column["unique"])
                for column in analysis["columns"]
                if column["name"] in categorical
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_name, top_unique = cardinalities[0]
        insights.append(
            {
                "title": "Categorical encoding",
                "detail": (
                    f"{len(categorical)} categorical column(s) will be one-hot encoded (capped at "
                    f"{settings.max_onehot_categories} categories per column). The widest column is "
                    f"'{top_name}' with {top_unique} distinct values."
                ),
                "level": "info" if top_unique <= settings.max_onehot_categories else "warning",
            }
        )
    return insights


def build_warnings(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    profile = analysis["profile"]
    if profile["n_rows"] < 200:
        warnings.append(
            {
                "code": "small_dataset",
                "message": (
                    f"Only {profile['n_rows']} rows are available; validation scores will be noisy "
                    "and cross-validation folds are small."
                ),
            }
        )
    if profile["has_high_cardinality"]:
        warnings.append(
            {
                "code": "high_cardinality",
                "message": (
                    "At least one categorical feature has many distinct values; one-hot encoding is "
                    f"capped at {settings.max_onehot_categories} categories per column."
                ),
            }
        )
    if profile["is_imbalanced"]:
        warnings.append(
            {
                "code": "class_imbalance",
                "message": (
                    "Class imbalance detected: weighted F1 is used as the primary metric instead of "
                    "accuracy."
                ),
            }
        )
    if analysis["duplicates"]["duplicate_pct"] > 5:
        warnings.append(
            {
                "code": "duplicates",
                "message": (
                    f"{analysis['duplicates']['duplicate_pct']}% of rows are duplicates, which can "
                    "make the evaluation optimistic."
                ),
            }
        )
    if analysis["missing_values"]["total_missing_pct"] > 40:
        warnings.append(
            {
                "code": "many_missing",
                "message": (
                    f"{analysis['missing_values']['total_missing_pct']}% of all cells are missing; "
                    "imputation may dominate the signal."
                ),
            }
        )
    return warnings


def enrich_insights(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the target- and relationship-level insights to an analysis dict."""
    analysis["insights"] = (
        build_insights(analysis) + build_target_insights(analysis) + build_relationship_insights(analysis)
    )
    return analysis


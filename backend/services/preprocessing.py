"""Leak-safe preprocessing pipelines built with ``ColumnTransformer``.

Rules:
  * the train / validation / test split happens *before* any transformation;
  * imputers and encoders are fitted on the training split only (they live inside
    the scikit-learn ``Pipeline``, so every ``fit`` uses training data only);
  * scaling is applied only when the selected model needs it;
  * categorical features are one-hot encoded with a category cap so that
    high-cardinality columns cannot explode the feature space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingPlan:
    """Describes how the feature matrix will be built."""

    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    scale_numeric: bool = False
    sparse_output: bool = False
    max_categories: int = settings.max_onehot_categories
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "scale_numeric": self.scale_numeric,
            "sparse_output": self.sparse_output,
            "max_categories": self.max_categories,
            "impute_numeric": self.impute_numeric,
            "impute_categorical": self.impute_categorical,
            "notes": list(self.notes),
        }


def plan_preprocessing(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_needs_scaling: bool,
    n_rows: int,
    missing_numeric: bool = False,
    missing_categorical: bool = False,
) -> PreprocessingPlan:
    sparse = n_rows > settings.dense_row_threshold
    plan = PreprocessingPlan(
        numeric_features=[str(c) for c in numeric_features],
        categorical_features=[str(c) for c in categorical_features],
        scale_numeric=bool(model_needs_scaling),
        sparse_output=bool(sparse and categorical_features),
        max_categories=settings.max_onehot_categories,
    )
    if plan.scale_numeric:
        plan.notes.append(
            "Numeric features are standardised because the selected model is sensitive to "
            "feature scale (distance- or margin-based)."
        )
    else:
        plan.notes.append(
            "Numeric features are left unscaled because the selected model is scale-invariant."
        )
    if plan.categorical_features:
        plan.notes.append(
            f"{len(plan.categorical_features)} categorical feature(s) are one-hot encoded with a "
            f"cap of {plan.max_categories} categories per column; unseen categories are ignored."
        )
    if missing_numeric or missing_categorical:
        plan.notes.append(
            "Missing values are imputed inside the pipeline with statistics learned from the "
            "training split only (no leakage)."
        )
    if plan.sparse_output:
        plan.notes.append(
            "One-hot output is kept sparse because the dataset is large "
            f"(> {settings.dense_row_threshold:,} rows)."
        )
    return plan


def build_preprocessor(plan: PreprocessingPlan) -> Optional[ColumnTransformer]:
    """Build the ``ColumnTransformer`` described by ``plan`` (``None`` if no features)."""
    transformers: List[tuple] = []

    if plan.numeric_features:
        numeric_steps: List[tuple] = [
            ("impute", SimpleImputer(strategy=plan.impute_numeric))
        ]
        if plan.scale_numeric:
            # with_mean is only valid for dense matrices.
            numeric_steps.append(("scale", StandardScaler(with_mean=not plan.sparse_output)))
        transformers.append(("numeric", Pipeline(numeric_steps), plan.numeric_features))

    if plan.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy=plan.impute_categorical, fill_value="missing"
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=plan.max_categories,
                                sparse_output=plan.sparse_output,
                                dtype="float64",
                            ),
                        ),
                    ]
                ),
                plan.categorical_features,
            )
        )

    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def build_pipeline(model: Any, plan: PreprocessingPlan) -> Pipeline:
    """Full pipeline: preprocessing -> estimator (fitted on training data only)."""
    preprocessor = build_preprocessor(plan)
    if preprocessor is None:
        raise ValueError("Cannot build a pipeline without any feature columns")
    return Pipeline([("preprocess", preprocessor), ("model", model)])

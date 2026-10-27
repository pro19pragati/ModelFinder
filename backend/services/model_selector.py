"""Transparent, rule-based model selection.

The agent never trains every model blindly. Instead it reads the dataset
analysis (size, linearity probe, class balance, cardinality, sparsity) and the
available resources, then applies explicit rules to pick *one* candidate model
plus an ordered list of alternatives to try later.

The rule set is intentionally explainable: every decision carries the numbers
that produced it. New strategies can be added by extending :data:`MODEL_REGISTRY`
or adding rules at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

REGRESSION = "regression"
CLASSIFICATION = "classification"


# --------------------------------------------------------------- builders
def _linear_regression(params: Dict[str, Any]):
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**_filter_params(LinearRegression, params))


def _ridge(params: Dict[str, Any]):
    from sklearn.linear_model import Ridge

    return Ridge(**_filter_params(Ridge, params))


def _decision_tree(task: str):
    def builder(params: Dict[str, Any]):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cls = DecisionTreeRegressor if task == REGRESSION else DecisionTreeClassifier
        return cls(random_state=settings.random_state, **_filter_params(cls, params))

    return builder


def _random_forest(task: str):
    def builder(params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestRegressor if task == REGRESSION else RandomForestClassifier
        return cls(random_state=settings.random_state, n_jobs=-1, **_filter_params(cls, params))

    return builder


def _gradient_boosting(task: str):
    def builder(params: Dict[str, Any]):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        hist = bool(params.pop("hist", False))
        if hist:
            cls = HistGradientBoostingRegressor if task == REGRESSION else HistGradientBoostingClassifier
            params.setdefault("max_iter", 200)
        else:
            cls = GradientBoostingRegressor if task == REGRESSION else GradientBoostingClassifier
            params.setdefault("n_estimators", 200)
        return cls(random_state=settings.random_state, **_filter_params(cls, params))

    return builder


def _support_vector(task: str):
    def builder(params: Dict[str, Any]):
        from sklearn.svm import SVC, SVR

        cls = SVR if task == REGRESSION else SVC
        return cls(**_filter_params(cls, params))

    return builder


def _neighbours(params: Dict[str, Any]):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(**_filter_params(KNeighborsClassifier, params))


def _logistic(params: Dict[str, Any]):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**_filter_params(LogisticRegression, params))


def _filter_params(cls: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keyword arguments the estimator actually accepts."""
    try:
        import inspect

        signature = inspect.signature(cls.__init__)
        allowed = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
        }
        accepted = {key: value for key, value in params.items() if key in allowed}
        dropped = set(params) - set(accepted)
        if dropped:
            logger.debug("Dropped unsupported params for %s: %s", cls.__name__, sorted(dropped))
        return accepted
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return dict(params)


@dataclass
class ModelSpec:
    """Everything the agent needs to know about one candidate algorithm."""

    key: str
    label: str
    task: str
    builder: Callable[[Dict[str, Any]], Any]
    default_params: Dict[str, Any]
    param_space: Optional[Callable[[Any], Dict[str, Any]]] = None
    complexity: float = 1.0          # relative training cost, 1.0 == plain linear model
    needs_scaling: bool = False
    supports_sparse: bool = True
    max_rows: Optional[int] = None   # practical ceiling for this algorithm
    description: str = ""
    strengths: str = ""
    weaknesses: str = ""
    family: str = "linear"
    tags: List[str] = field(default_factory=list)


# --------------------------------------------------------------- param spaces
def _space_linear(trial) -> Dict[str, Any]:
    return {"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}


def _space_ridge(trial) -> Dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _space_logistic(trial) -> Dict[str, Any]:
    penalty = trial.suggest_categorical("penalty", ["l2"])
    params: Dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
        "penalty": penalty,
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }
    if params["solver"] == "liblinear" and penalty == "l2":
        params["solver"] = "liblinear"
    return params


def _space_tree(task: str) -> Callable[[Any], Dict[str, Any]]:
    def space(trial) -> Dict[str, Any]:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        }
        if task == CLASSIFICATION:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        else:
            params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"])
        return params

    return space


def _space_forest(task: str) -> Callable[[Any], Dict[str, Any]]:
    def space(trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 320, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }
        return params

    return space


def _space_gradient(task: str) -> Callable[[Any], Dict[str, Any]]:
    def space(trial) -> Dict[str, Any]:
        hist = bool(trial.study.user_attrs.get("hist", False))
        if hist:
            return {
                "hist": True,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "max_iter": trial.suggest_int("max_iter", 80, 320, step=20),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            }
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "n_estimators": trial.suggest_int("n_estimators", 60, 320, step=20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        }
        return params

    return space


def _space_svr(trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
    }


def _space_svc(trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
        "probability": True,
    }


def _space_knn(trial) -> Dict[str, Any]:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2]),
    }



# ------------------------------------------------------------------- registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.key] = spec


# ------------------------------------------------------------- regression zoo
_register(
    ModelSpec(
        key="linear_regression",
        label="Linear Regression",
        task=REGRESSION,
        builder=_linear_regression,
        default_params={},
        param_space=_space_linear,
        complexity=1.0,
        needs_scaling=False,
        family="linear",
        description="Ordinary least squares fit of a straight-line relationship.",
        strengths="Fast, transparent coefficients, ideal when relationships are approximately linear.",
        weaknesses="Cannot capture interactions or non-linear curvature; sensitive to outliers.",
        tags=["fast", "interpretable", "baseline"],
    )
)
_register(
    ModelSpec(
        key="ridge_regression",
        label="Ridge Regression",
        task=REGRESSION,
        builder=_ridge,
        default_params={"alpha": 1.0},
        param_space=_space_ridge,
        complexity=1.0,
        needs_scaling=True,
        family="linear",
        description="L2-regularised linear regression that shrinks unstable coefficients.",
        strengths="Stable when features are correlated or the matrix is ill-conditioned.",
        weaknesses="Still limited to linear relationships.",
        tags=["fast", "interpretable", "regularised"],
    )
)
_register(
    ModelSpec(
        key="decision_tree_regressor",
        label="Decision Tree Regressor",
        task=REGRESSION,
        builder=_decision_tree(REGRESSION),
        default_params={"max_depth": 10, "min_samples_leaf": 8},
        param_space=_space_tree(REGRESSION),
        complexity=2.5,
        needs_scaling=False,
        family="tree",
        description="Recursive binary splits that capture non-linear patterns and interactions.",
        strengths="Handles non-linearity, robust to outliers, easy to explain.",
        weaknesses="High variance; a single tree overfits without depth limits.",
        tags=["nonlinear", "interpretable"],
    )
)
_register(
    ModelSpec(
        key="random_forest_regressor",
        label="Random Forest Regressor",
        task=REGRESSION,
        builder=_random_forest(REGRESSION),
        default_params={"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 2},
        param_space=_space_forest(REGRESSION),
        complexity=12.0,
        needs_scaling=False,
        family="ensemble",
        description="Bagged ensemble of decorrelated trees.",
        strengths="Strong default for non-linear tabular data, robust to outliers and noise.",
        weaknesses="Larger model, slower to train, less transparent than a single tree.",
        tags=["nonlinear", "robust"],
    )
)
_register(
    ModelSpec(
        key="gradient_boosting_regressor",
        label="Gradient Boosting Regressor",
        task=REGRESSION,
        builder=_gradient_boosting(REGRESSION),
        default_params={"n_estimators": 200, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.9},
        param_space=_space_gradient(REGRESSION),
        complexity=14.0,
        needs_scaling=False,
        family="ensemble",
        description="Sequentially boosted trees (histogram-based for large datasets).",
        strengths="Usually the strongest accuracy on structured/tabular regression.",
        weaknesses="More hyperparameters, slower training, can overfit small datasets.",
        tags=["nonlinear", "boosted"],
    )
)
_register(
    ModelSpec(
        key="svr",
        label="Support Vector Regression",
        task=REGRESSION,
        builder=_support_vector(REGRESSION),
        default_params={"C": 10.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"},
        param_space=_space_svr,
        complexity=20.0,
        needs_scaling=True,
        supports_sparse=False,
        max_rows=20_000,
        family="margin",
        description="Kernel machine that fits an epsilon-insensitive tube around the target.",
        strengths="Effective in moderate dimensions with smooth non-linear responses.",
        weaknesses="Scales poorly with dataset size; needs feature scaling.",
        tags=["nonlinear", "kernel"],
    )
)



# ----------------------------------------------------------- classification zoo
_register(
    ModelSpec(
        key="logistic_regression",
        label="Logistic Regression",
        task=CLASSIFICATION,
        builder=_logistic,
        default_params={"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
        param_space=_space_logistic,
        complexity=1.5,
        needs_scaling=True,
        family="linear",
        description="Linear decision boundary with a logistic link and L2 regularisation.",
        strengths="Fast, calibrated probabilities, reliable baseline for linearly separable data.",
        weaknesses="Cannot model complex interactions or non-linear boundaries.",
        tags=["fast", "interpretable", "baseline", "probabilistic"],
    )
)
_register(
    ModelSpec(
        key="decision_tree_classifier",
        label="Decision Tree Classifier",
        task=CLASSIFICATION,
        builder=_decision_tree(CLASSIFICATION),
        default_params={"max_depth": 10, "min_samples_leaf": 5},
        param_space=_space_tree(CLASSIFICATION),
        complexity=2.5,
        needs_scaling=False,
        family="tree",
        description="Axis-aligned splits producing an interpretable rule set.",
        strengths="Simple to explain, handles mixed feature types.",
        weaknesses="Unstable and prone to overfitting on small samples.",
        tags=["nonlinear", "interpretable"],
    )
)
_register(
    ModelSpec(
        key="random_forest_classifier",
        label="Random Forest Classifier",
        task=CLASSIFICATION,
        builder=_random_forest(CLASSIFICATION),
        default_params={"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 2},
        param_space=_space_forest(CLASSIFICATION),
        complexity=12.0,
        needs_scaling=False,
        family="ensemble",
        description="Bagged decision trees with class-probability averaging.",
        strengths="Robust, handles imbalance with class weights, gives feature importance.",
        weaknesses="Slower than a linear model and less interpretable.",
        tags=["nonlinear", "robust", "probabilistic"],
    )
)
_register(
    ModelSpec(
        key="gradient_boosting_classifier",
        label="Gradient Boosting Classifier",
        task=CLASSIFICATION,
        builder=_gradient_boosting(CLASSIFICATION),
        default_params={"n_estimators": 200, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.9},
        param_space=_space_gradient(CLASSIFICATION),
        complexity=14.0,
        needs_scaling=False,
        family="ensemble",
        description="Boosted trees (histogram-based for large datasets).",
        strengths="Typically the best accuracy on tabular classification.",
        weaknesses="Slower training and more hyperparameters to tune.",
        tags=["nonlinear", "boosted", "probabilistic"],
    )
)
_register(
    ModelSpec(
        key="svc",
        label="Support Vector Classifier",
        task=CLASSIFICATION,
        builder=_support_vector(CLASSIFICATION),
        default_params={"C": 10.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        param_space=_space_svc,
        complexity=20.0,
        needs_scaling=True,
        max_rows=20_000,
        family="margin",
        description="Maximum-margin classifier with an RBF kernel and probability estimates.",
        strengths="Strong on small/medium datasets with smooth class boundaries.",
        weaknesses="Does not scale to large datasets; sensitive to scaling and C.",
        tags=["nonlinear", "kernel", "probabilistic"],
    )
)
_register(
    ModelSpec(
        key="knn_classifier",
        label="K-Nearest Neighbors",
        task=CLASSIFICATION,
        builder=_neighbours,
        default_params={"n_neighbors": 9, "weights": "distance"},
        param_space=_space_knn,
        complexity=1.5,
        needs_scaling=True,
        max_rows=50_000,
        family="instance",
        description="Majority vote among the closest training samples.",
        strengths="No training phase, naturally non-linear, easy to reason about.",
        weaknesses="Sensitive to scaling, noisy features and curse of dimensionality.",
        tags=["nonlinear", "lazy"],
    )
)



# ------------------------------------------------------------- selection rules
@dataclass
class SelectionResult:
    model_key: str
    model_name: str
    rule: str
    reason: str
    evidence: Dict[str, Any]
    alternates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_pool: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "rule": self.rule,
            "reason": self.reason,
            "evidence": self.evidence,
            "alternates": self.alternates,
            "candidate_pool": self.candidate_pool,
        }


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'")
    return MODEL_REGISTRY[model_key]


def models_for_task(task: str) -> List[ModelSpec]:
    return [spec for spec in MODEL_REGISTRY.values() if spec.task == task]


def practical_models(task: str, n_rows: int, effective_width: int) -> List[ModelSpec]:
    """Filter the zoo down to models that are practical for this dataset."""
    usable = []
    for spec in models_for_task(task):
        if spec.max_rows is not None and n_rows > spec.max_rows:
            continue
        if not spec.supports_sparse and effective_width > 20_000:
            continue
        usable.append(spec)
    # Cheapest first so the initial pick stays resource-friendly.
    usable.sort(key=lambda spec: spec.complexity)
    return usable


def _max_feature_correlation(analysis: Dict[str, Any]) -> float:
    pairs = (analysis.get("correlations") or {}).get("top_pairs") or []
    return max((abs(float(pair["correlation"])) for pair in pairs), default=0.0)


def _alternate_reason(spec: ModelSpec, analysis: Dict[str, Any], chosen: str) -> str:
    linearity = analysis.get("linearity") or {}
    if spec.family == "linear":
        return "Cheap linear alternative that is useful as a reference baseline."
    if spec.family == "tree":
        return "Single interpretable tree: good for capturing abrupt thresholds in the data."
    if spec.family == "ensemble":
        return (
            "Ensemble of trees; expected to improve on the current best model if the remaining "
            f"error is non-linear (linear probe R2 {linearity.get('linear_cv_r2')})."
        )
    if spec.family == "margin":
        return "Kernel model; effective when the response surface is smooth and the data is small."
    if spec.family == "instance":
        return "Instance-based baseline that makes no assumption about the functional form."
    return f"Alternative candidate to the current model '{chosen}'."


def select_initial_model(
    analysis: Dict[str, Any], resource_snapshot: Optional[Dict[str, Any]] = None
) -> SelectionResult:
    """Pick the first candidate model using explicit, documented rules."""
    profile = analysis.get("profile") or {}
    problem_type = analysis.get("problem_type", REGRESSION)
    n_rows = int(profile.get("n_rows") or (analysis.get("shape") or {}).get("n_rows", 0))
    n_numeric = int(profile.get("n_numeric_features", 0))
    n_categorical = int(profile.get("n_categorical_features", 0))
    effective_width = int(profile.get("estimated_onehot_width") or (n_numeric + n_categorical))
    memory_available = (resource_snapshot or {}).get("memory_available_mb")
    linearity = analysis.get("linearity") or {}
    balance = (analysis.get("target_profile") or {}).get("class_balance") or {}
    max_corr = _max_feature_correlation(analysis)

    pool = practical_models(problem_type, n_rows, effective_width)
    pool_keys = [spec.key for spec in pool]

    evidence: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_numeric_features": n_numeric,
        "n_categorical_features": n_categorical,
        "effective_feature_width": effective_width,
        "max_feature_correlation": round(max_corr, 3),
        "linearity": linearity,
        "class_balance": balance,
        "memory_available_mb": memory_available,
    }

    if problem_type == REGRESSION:
        linear_r2 = linearity.get("linear_cv_r2")
        tree_r2 = linearity.get("tree_cv_r2")
        is_linear = bool(linearity.get("is_approximately_linear"))
        known = linear_r2 is not None
        evidence["linearity_known"] = known
        chosen, rule, reason = _select_regression(
            n_rows, n_numeric, effective_width, max_corr, linear_r2, tree_r2, is_linear, known
        )
    else:
        chosen, rule, reason = _select_classification(
            n_rows, n_numeric, n_categorical, effective_width, balance, profile
        )
        evidence["n_classes"] = int(balance.get("n_classes") or profile.get("n_classes") or 0)
        evidence["is_imbalanced"] = bool(balance.get("is_imbalanced"))


    # Guard rail: under memory pressure prefer memory-efficient boosted models.
    if (
        memory_available is not None
        and memory_available < 1_500
        and n_rows > 20_000
        and chosen in {"random_forest_regressor", "random_forest_classifier"}
    ):
        chosen = (
            "gradient_boosting_regressor"
            if problem_type == REGRESSION
            else "gradient_boosting_classifier"
        )
        rule = "memory_constrained_boosted"
        reason = (
            f"Only {memory_available:.0f} MB of RAM is available, so a memory-efficient "
            "histogram-based Gradient Boosting model is preferred over a Random Forest for this "
            f"{n_rows:,}-row dataset."
        )

    if chosen not in MODEL_REGISTRY:
        chosen = pool_keys[0] if pool_keys else chosen
    spec = MODEL_REGISTRY[chosen]

    alternates = [
        {
            "model_key": candidate.key,
            "model_name": candidate.label,
            "family": candidate.family,
            "complexity": candidate.complexity,
            "reason": _alternate_reason(candidate, analysis, chosen),
        }
        for candidate in pool
        if candidate.key != chosen
    ]

    logger.info(
        "Selected initial model %s via rule '%s' (rows=%s, width=%s)",
        spec.key,
        rule,
        n_rows,
        effective_width,
    )
    return SelectionResult(
        model_key=spec.key,
        model_name=spec.label,
        rule=rule,
        reason=reason,
        evidence=evidence,
        alternates=alternates,
        candidate_pool=pool_keys,
    )


def _select_regression(
    n_rows: int,
    n_numeric: int,
    effective_width: int,
    max_corr: float,
    linear_r2: Optional[float],
    tree_r2: Optional[float],
    is_linear: bool,
    known: bool,
) -> tuple:
    """Rule table for regression datasets. Returns (model_key, rule, reason)."""
    if is_linear and max_corr < 0.9 and effective_width <= 500:
        return (
            "linear_regression",
            "linear_relationships",
            (
                f"The linearity probe reached a cross-validated R2 of {linear_r2} versus {tree_r2} "
                "for a depth-6 decision tree on a 2,000-row sample, and the largest "
                f"feature-to-feature correlation is {max_corr:.2f}. A plain Linear Regression is "
                "expected to capture the signal without overfitting."
            ),
        )
    if is_linear and (max_corr >= 0.9 or effective_width > 500):
        return (
            "ridge_regression",
            "linear_with_multicollinearity",
            (
                "The relationships look approximately linear, but the design matrix is poorly "
                f"conditioned (largest feature-to-feature correlation {max_corr:.2f}, effective "
                f"width {effective_width} columns), so Ridge Regression is used to keep the "
                "coefficients stable through L2 regularisation."
            ),
        )
    if known and tree_r2 is not None and linear_r2 is not None and tree_r2 - linear_r2 > 0.08:
        if n_rows <= 2_000:
            return (
                "decision_tree_regressor",
                "nonlinear_small_dataset",
                (
                    f"The tree probe scored R2 {tree_r2} against {linear_r2} for a linear model, so "
                    "clear non-linear structure is present. With "
                    f"{n_rows:,} rows a single regularised Decision Tree is the cheapest way to "
                    "capture it before trying an ensemble."
                ),
            )
        if n_rows <= 30_000:
            return (
                "random_forest_regressor",
                "nonlinear_tabular",
                (
                    f"A linear model scored R2 {linear_r2} while the tree probe reached {tree_r2}, "
                    f"so the relationships are non-linear. For {n_rows:,} rows a Random Forest is "
                    "the best trade-off: low variance, robust to outliers and no scaling required."
                ),
            )
        return (
            "gradient_boosting_regressor",
            "nonlinear_large_dataset",
            (
                f"Non-linear structure is present (linear R2 {linear_r2} vs tree R2 {tree_r2}) and "
                f"the dataset is large ({n_rows:,} rows), so histogram-based Gradient Boosting "
                "gives the most accuracy per unit of training time."
            ),
        )
    if known and max(linear_r2 or 0.0, tree_r2 or 0.0) < 0.2:
        return (
            "random_forest_regressor",
            "weak_signal",
            (
                f"Neither a linear model (R2 {linear_r2}) nor a depth-6 tree (R2 {tree_r2}) found "
                "much signal, so a Random Forest is used to search for non-linear structure before "
                "the agent concludes that the features are uninformative."
            ),
        )
    if n_rows <= 2_000 and n_numeric <= 12:
        return (
            "svr",
            "small_dataset_smooth_function",
            (
                f"The linearity probe is inconclusive and the dataset is small ({n_rows:,} rows, "
                f"{n_numeric} numeric features), where kernel Support Vector Regression is both "
                "fast and effective."
            ),
        )
    return (
        "random_forest_regressor",
        "default_nonlinear",
        (
            "The linearity probe was inconclusive, so the agent starts from the robust Random "
            "Forest default for tabular regression."
        ),
    )


def _select_classification(
    n_rows: int,
    n_numeric: int,
    n_categorical: int,
    effective_width: int,
    balance: Dict[str, Any],
    profile: Dict[str, Any],
) -> tuple:
    """Rule table for classification datasets. Returns (model_key, rule, reason)."""
    n_classes = int(balance.get("n_classes") or profile.get("n_classes") or 0)
    imbalanced = bool(balance.get("is_imbalanced"))

    if n_rows <= 1_000 and (n_numeric + n_categorical) <= 12:
        return (
            "knn_classifier",
            "small_dataset_local_structure",
            (
                f"With only {n_rows:,} rows and {n_numeric + n_categorical} features, "
                "K-Nearest Neighbors provides a strong assumption-free baseline that trains almost "
                "instantly."
            ),
        )
    if effective_width <= 300 and n_classes <= 5:
        suffix = (
            f"; class weights are balanced because the classes are imbalanced "
            f"({balance.get('imbalance_ratio')}:1)."
            if imbalanced
            else "."
        )
        return (
            "logistic_regression",
            "linear_decision_boundary",
            (
                f"The target has {n_classes} class(es) and the encoded feature width is "
                f"{effective_width} columns, so a regularised Logistic Regression gives a fast, "
                "calibrated baseline with interpretable coefficients" + suffix
            ),
        )
    if n_rows <= 40_000:
        suffix = (
            f" with balanced class weights for the {balance.get('imbalance_ratio')}:1 imbalance."
            if imbalanced
            else "."
        )
        return (
            "random_forest_classifier",
            "mixed_nonlinear_features",
            (
                f"The dataset mixes {n_numeric} numeric and {n_categorical} categorical features "
                f"({effective_width} encoded columns) and is unlikely to be linearly separable, so "
                "a Random Forest Classifier is chosen" + suffix
            ),
        )
    return (
        "gradient_boosting_classifier",
        "large_dataset_boosted",
        (
            f"With {n_rows:,} rows, histogram-based Gradient Boosting delivers the best accuracy "
            "per unit of training time for tabular classification."
        ),
    )


# --------------------------------------------------------------- public helpers
def default_params_for(
    model_key: str, problem_type: str, profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Reasonable starting hyperparameters, adjusted for dataset size/imbalance."""
    spec = get_spec(model_key)
    params = dict(spec.default_params)
    profile = profile or {}
    n_rows = int(profile.get("n_rows") or 0)
    imbalanced = bool(profile.get("is_imbalanced"))
    effective_width = int(profile.get("estimated_onehot_width") or 0)

    if "n_estimators" in params and n_rows > 40_000:
        params["n_estimators"] = min(params["n_estimators"], 120)
    if model_key in {"gradient_boosting_regressor", "gradient_boosting_classifier"}:
        if n_rows > 30_000 or effective_width > 300:
            params["hist"] = True
            params.pop("subsample", None)
            params.pop("min_samples_leaf", None)
            params.setdefault("max_iter", 200)
            params.pop("n_estimators", None)
    if problem_type == CLASSIFICATION and imbalanced:
        if model_key in {
            "logistic_regression",
            "random_forest_classifier",
            "decision_tree_classifier",
            "svc",
        }:
            params["class_weight"] = "balanced"
    if model_key == "logistic_regression" and n_rows > 50_000:
        params["max_iter"] = 1500
    return params


def param_search_space(model_key: str) -> Optional[Callable[[Any], Dict[str, Any]]]:
    return get_spec(model_key).param_space


def describe_model(model_key: str) -> Dict[str, Any]:
    spec = get_spec(model_key)
    return {
        "model_key": spec.key,
        "model_name": spec.label,
        "task": spec.task,
        "family": spec.family,
        "complexity": spec.complexity,
        "needs_scaling": spec.needs_scaling,
        "supports_sparse": spec.supports_sparse,
        "max_rows": spec.max_rows,
        "description": spec.description,
        "strengths": spec.strengths,
        "weaknesses": spec.weaknesses,
        "default_params": dict(spec.default_params),
        "searchable": spec.param_space is not None,
        "tags": list(spec.tags),
    }


def all_models(task: Optional[str] = None) -> List[Dict[str, Any]]:
    keys = [spec.key for spec in models_for_task(task)] if task else list(MODEL_REGISTRY)
    return [describe_model(key) for key in keys]


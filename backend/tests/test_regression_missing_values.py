"""Focused regression test for missing-value handling during regression training.

Root cause being guarded against
--------------------------------
When pandas 3.x reads a CSV, string columns arrive as the nullable ``str``
dtype and missing cells are stored as ``pd.NA`` (not ``np.nan``).  The old
``prepare_features`` code in ``trainer.py`` cast categorical columns with
``astype("string")``, keeping ``pd.NA``.  scikit-learn's ``SimpleImputer``
detects missing values via ``X != X``; for ``pd.NA`` that expression returns
``pd.NA`` rather than ``True``, which raises::

    TypeError: boolean value of NA is ambiguous

The fix converts categorical columns to ``object`` dtype and replaces
``pd.NA`` with ``np.nan`` before the data enters the sklearn pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import trainer
from services import model_selector as ms

# --------------------------------------------------------------------- data factory
def _house_price_frame(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic house-price dataset with **both** numeric and categorical gaps."""
    rng = np.random.default_rng(seed)
    area = rng.normal(1800, 520, n_rows).clip(420, 5200).round(1)
    bedrooms = rng.integers(1, 6, n_rows)
    age = rng.integers(0, 60, n_rows).astype(float)
    dist = np.abs(rng.normal(8, 4.5, n_rows))
    location = rng.choice(["Urban", "Suburban", "Rural"], n_rows)
    property_type = rng.choice(["Villa", "Apartment"], n_rows)
    parking = rng.choice(["Yes", "No"], n_rows)

    price = (
        90_000
        + 178.5 * area
        + 9500 * bedrooms
        - 1450 * age
        - 6100 * dist
        + rng.normal(0, 15000, n_rows)
    )

    df = pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "property_age_years": age,
            "location": location,
            "property_type": property_type,
            "parking": parking,
            "distance_to_city_km": dist,
            "house_price_inr": price.round(0),
        }
    )

    # Inject realistic missing values: numeric NaNs + categorical pd.NA.
    mask_num = rng.choice(n_rows, size=25, replace=False)
    df.loc[mask_num, "property_age_years"] = np.nan

    mask_cat = rng.choice(n_rows, size=30, replace=False)
    df.loc[mask_cat, "parking"] = pd.NA

    return df


def _regression_model_keys() -> list[str]:
    """All regression model keys that the selector registers."""
    return [spec.key for spec in ms.models_for_task(ms.REGRESSION)]


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def house_price_df() -> pd.DataFrame:
    return _house_price_frame()


@pytest.fixture
def regression_split(house_price_df):
    """Prepare the dataset split exactly as the training pipeline does."""
    return trainer.split_dataset(house_price_df, "house_price_inr", "regression")


class TestRegressionMissingValues:
    """Guard against the ``boolean value of NA is ambiguous`` regression."""

    def test_numeric_columns_keep_np_nan(self, regression_split):
        """Numeric features with missing values use ``np.nan`` (float dtype)."""
        for col in regression_split.numeric_features:
            dtype = str(regression_split.X_train[col].dtype)
            assert dtype in ("float64", "int64"), (
                f"Numeric feature '{col}' has unexpected dtype {dtype}"
            )

    def test_categorical_columns_are_object_not_nullable_string(self, regression_split):
        """Categorical features must be ``object`` dtype (not ``str``/``string``)
        so that ``pd.NA`` is replaced by ``np.nan`` before sklearn sees them."""
        for col in regression_split.categorical_features:
            dtype = str(regression_split.X_train[col].dtype)
            assert dtype == "object", (
                f"Categorical feature '{col}' is {dtype!r}, expected 'object'. "
                "Nullable 'string'/'str' dtype stores missing values as pd.NA "
                "which crashes scikit-learn SimpleImputer."
            )

    def test_missing_categorical_values_present(self, regression_split):
        """Sanity check: the parking column actually has missing values."""
        assert "parking" in regression_split.categorical_features
        assert int(regression_split.X_train["parking"].isna().sum()) > 0

    @pytest.mark.parametrize("model_key", _regression_model_keys())
    def test_all_regression_models_train_with_missing_values(
        self, regression_split, model_key
    ):
        """Every regression model key must fit without 'boolean value of NA'."""
        try:
            result = trainer.run_experiment(
                regression_split, model_key, {}, "regression", with_cv=False
            )
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"Training {model_key} failed with missing values: {exc}")

        metrics = result["validation_metrics"]
        # Real metric values (not None) must be produced.
        assert metrics.get("r2") is not None, f"{model_key} produced no R2 metric"
        assert metrics.get("mae") is not None, f"{model_key} produced no MAE metric"
        assert metrics.get("rmse") is not None, f"{model_key} produced no RMSE metric"
        assert metrics["n_samples"] > 0

    def test_finalize_model_works_with_missing_values(self, regression_split):
        """finalize_model (refit on train+val, score on test) must not crash."""
        plan = trainer.build_plan_for("linear_regression", regression_split)
        estimator = trainer.make_estimator("linear_regression", {})
        pipeline = trainer.preprocessing.build_pipeline(estimator, plan)

        X_combined = pd.concat(
            [regression_split.X_train, regression_split.X_val], axis=0
        )
        y_combined = np.concatenate(
            [regression_split.y_train, regression_split.y_val]
        )
        pipeline.fit(X_combined, y_combined)

        metrics = trainer.evaluate_pipeline(
            pipeline,
            regression_split.X_test,
            regression_split.y_test,
            "regression",
            regression_split.classes,
        )
        assert metrics.get("r2") is not None
        assert metrics.get("mae") is not None
        assert metrics.get("rmse") is not None




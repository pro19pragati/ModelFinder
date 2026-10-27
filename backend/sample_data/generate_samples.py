"""Generate the bundled sample datasets used for demos and tests.

Run from the ``backend`` directory::

    python sample_data/generate_samples.py

Creates three CSV files:
  * ``sample_linear_regression.csv``    - house prices, approximately linear
  * ``sample_nonlinear_regression.csv`` - energy demand, clearly non-linear
  * ``sample_classification.csv``       - customer churn, imbalanced + categoricals
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42


def linear_regression_dataset(n_rows: int = 6000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size_sqft = rng.normal(1800, 520, n_rows).clip(420, 5200)
    bedrooms = rng.integers(1, 6, n_rows)
    bathrooms = rng.integers(1, 4, n_rows)
    age_years = rng.integers(0, 60, n_rows)
    distance_km = np.abs(rng.normal(8, 4.5, n_rows))
    school_score = rng.normal(7.2, 1.4, n_rows).clip(1, 10).round(1)
    garage = rng.integers(0, 3, n_rows)
    neighbourhood = rng.choice(
        ["Downtown", "Suburb", "Riverside", "Countryside"], n_rows, p=[0.3, 0.4, 0.18, 0.12]
    )

    # Deliberately linear: price is an almost exact linear combination + noise.
    premium = {"Downtown": 42000, "Suburb": 12000, "Riverside": 26000, "Countryside": -15000}
    price = (
        90000
        + 178.5 * size_sqft
        + 9500 * bedrooms
        + 14500 * bathrooms
        - 1450 * age_years
        - 6100 * distance_km
        + 8200 * school_score
        + 7200 * garage
        + np.array([premium[value] for value in neighbourhood])
        + rng.normal(0, 15000, n_rows)
    )
    return pd.DataFrame(
        {
            "size_sqft": size_sqft.round(1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "distance_to_center_km": distance_km.round(2),
            "school_score": school_score,
            "garage_spaces": garage,
            "neighbourhood": neighbourhood,
            "price": price.clip(45000, None).round(0),
        }
    )


def nonlinear_regression_dataset(n_rows: int = 6000, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temperature = rng.normal(18, 9, n_rows).clip(-8, 42)
    humidity = rng.normal(58, 16, n_rows).clip(10, 100)
    hour = rng.integers(0, 24, n_rows)
    day_of_week = rng.integers(0, 7, n_rows)
    industrial_load = np.abs(rng.normal(55, 18, n_rows))
    solar_output = rng.normal(30, 18, n_rows).clip(0, 90)

    # Strong non-linear structure: quadratic temperature response + interaction.
    energy = (
        120
        + 2.6 * (temperature - 19) ** 2
        + 0.55 * humidity
        + 11.0 * np.sin(hour / 24 * 2 * np.pi)
        + 0.42 * industrial_load
        - 0.85 * solar_output
        + 0.00045 * industrial_load * humidity
        + 6.5 * (day_of_week >= 5)
        + rng.normal(0, 9, n_rows)
    )
    return pd.DataFrame(
        {
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "industrial_load_mw": industrial_load.round(2),
            "solar_output_mw": solar_output.round(2),
            "energy_demand_mw": energy.clip(5, None).round(2),
        }
    )


def classification_dataset(n_rows: int = 6000, seed: int = SEED + 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 72, n_rows)
    monthly_charges = rng.normal(72, 26, n_rows).clip(18, 160).round(2)
    total_charges = (tenure_months * monthly_charges * rng.normal(1.0, 0.05, n_rows)).round(2)
    support_tickets = rng.poisson(1.1, n_rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        n_rows,
        p=[0.35, 0.28, 0.22, 0.15],
    )
    has_tech_support = rng.choice([0, 1, 2], n_rows, p=[0.45, 0.30, 0.25])
    is_senior = rng.choice([0, 1], n_rows, p=[0.82, 0.18])
    usage_gb = np.abs(rng.normal(340, 180, n_rows)).round(1)

    contract_weight = {"Month-to-month": 1.35, "One year": -0.35, "Two year": -1.15}
    payment_weight = {
        "Electronic check": 0.75,
        "Credit card": -0.25,
        "Bank transfer": -0.35,
        "Mailed check": 0.05,
    }
    logits = (
        -1.35
        + 0.165 * (monthly_charges - 70) / 10
        - 0.045 * tenure_months
        + 0.42 * support_tickets
        + np.array([contract_weight[value] for value in contract])
        + np.array([payment_weight[value] for value in payment_method])
        - 0.55 * has_tech_support
        + 0.45 * is_senior
        - 0.0009 * usage_gb
    )
    probability = 1 / (1 + np.exp(-logits))
    churn = (rng.random(n_rows) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "usage_gb": usage_gb,
            "contract_type": contract,
            "payment_method": payment_method,
            "tech_support_level": has_tech_support,
            "is_senior": is_senior,
            "churn": churn,
        }
    )
    # Introduce realistic missing data plus a handful of duplicate rows.
    missing_idx = rng.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    frame.loc[missing_idx, "total_charges"] = np.nan
    return pd.concat([frame, frame.iloc[:25]], ignore_index=True)


def write_all() -> None:
    outputs = {
        "sample_linear_regression.csv": linear_regression_dataset(),
        "sample_nonlinear_regression.csv": nonlinear_regression_dataset(),
        "sample_classification.csv": classification_dataset(),
    }
    summary = {}
    for filename, frame in outputs.items():
        path = HERE / filename
        frame.to_csv(path, index=False)
        summary[filename] = {
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "target": str(frame.columns[-1]),
        }
        print(f"wrote {path.name}: {frame.shape[0]} rows x {frame.shape[1]} cols")
    (HERE / "sample_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_all()

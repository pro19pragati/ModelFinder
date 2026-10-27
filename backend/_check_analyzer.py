"""Ad-hoc validation of the dataset analyzer against the three sample datasets."""

import json
from pathlib import Path

from services import dataset_analyzer as da

SAMPLES = Path(__file__).parent / "sample_data"

CASES = [
    ("sample_linear_regression.csv", "price"),
    ("sample_nonlinear_regression.csv", "energy_demand_mw"),
    ("sample_classification.csv", "churn"),
]

for filename, target in CASES:
    path = SAMPLES / filename
    frame = da.load_dataframe(path)
    analysis = da.analyze_dataframe(
        frame,
        target,
        file_size_mb=path.stat().st_size / (1024 * 1024),
    )
    print("=" * 78)
    print(f"{filename} -> target='{target}'")
    print("  shape:", analysis["shape"])
    print("  problem_type:", analysis["problem_type"], "| source:", analysis["problem_type_source"])
    print("  detection:", analysis["problem_type_detection"]["confidence"],
          "| ambiguous:", analysis["problem_type_detection"]["ambiguous"])
    print("  profile:", json.dumps(analysis["profile"]))
    print("  missing:", analysis["missing_values"]["total_missing"],
          "| duplicates:", analysis["duplicates"]["duplicate_rows"])
    print("  top target corr:", analysis["correlations"].get("top_target_correlations", [])[:3])
    print("  relationships:", analysis["feature_target_relationships"][:3])
    print("  outliers:", analysis["outliers"]["total_outlier_values"])
    print("  linearity:", analysis["linearity"] if analysis["linearity"] else "n/a")
    print("  warnings:", [w["code"] for w in analysis["warnings"]])
    print("  INSIGHTS:")
    for item in analysis["insights"]:
        print(f"    - [{item['level']}] {item['title']}: {item['detail']}")
    print("  target candidates:", da.target_candidates(frame)[:5])
    # Ensure JSON serialisable (FastAPI will do this)
    json.dumps(analysis)
print("=" * 78)
print("analyzer OK")

"""Ad-hoc validation of the model selector against the three sample datasets."""

from pathlib import Path

from services import dataset_analyzer as da
from services import model_selector as ms
from services import resource_monitor as rm

SAMPLES = Path(__file__).parent / "sample_data"
CASES = [
    ("sample_linear_regression.csv", "price"),
    ("sample_nonlinear_regression.csv", "energy_demand_mw"),
    ("sample_classification.csv", "churn"),
]

snapshot = rm.resource_snapshot()
for filename, target in CASES:
    path = SAMPLES / filename
    frame = da.load_dataframe(path)
    analysis = da.analyze_dataframe(frame, target)
    result = ms.select_initial_model(analysis, snapshot)
    print("=" * 78)
    print(f"{filename}: selected {result.model_name} (rule={result.rule})")
    print("  reason:", result.reason)
    print("  evidence:", result.evidence)
    print("  pool:", result.candidate_pool)
    print("  alternates:", [item["model_name"] for item in result.alternates])
    print("  default params:",
          ms.default_params_for(result.model_key, analysis["problem_type"], analysis["profile"]))
    print("  all models:", [m["model_name"] for m in ms.all_models(analysis["problem_type"])])
print("=" * 78)
print("selector OK")

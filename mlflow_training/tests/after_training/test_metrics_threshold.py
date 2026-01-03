# Checks that the exported metrics are numeric (finite) and that RMSE stays within an expected bound.
import json
import math
from pathlib import Path

def test_metrics_are_finite():
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["mae"])

def test_rmse_reasonable():
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    assert metrics["rmse"] < 2.0

# This test checks for the existence of specific artifacts after model training.

from pathlib import Path

def test_pipeline_artifact_exists():
    assert Path("artifacts/pipeline.pkl").exists()

def test_metrics_artifact_exists():
    assert Path("artifacts/metrics.json").exists()

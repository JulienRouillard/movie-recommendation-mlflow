# Ensures the training job produced the expected pipeline and metrics artifacts on disk.
from pathlib import Path

def test_pipeline_artifact_exists():
    assert Path("artifacts/pipeline.pkl").exists()

def test_metrics_artifact_exists():
    assert Path("artifacts/metrics.json").exists()

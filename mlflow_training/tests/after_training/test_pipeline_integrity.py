# Confirms the serialized pipeline contains all required keys for downstream prediction logic.
import pickle
from pathlib import Path

def test_pipeline_keys_exist():
    pipeline = pickle.loads(Path("artifacts/pipeline.pkl").read_bytes())

    required_keys = [
        "user_to_idx", "movie_to_idx", "user_factors", "movie_factors",
        "user_means", "movie_means", "global_mean_rating",
        "movie_embeddings_dict", "hybrid_model", "hybrid_scaler"
    ]

    for k in required_keys:
        assert k in pipeline

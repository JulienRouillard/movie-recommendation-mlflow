# Validates that the serialized pipeline can run inference and returns a rating within the allowed scale.
import pickle
from pathlib import Path
from train_mlflow import predict_final

def test_trained_pipeline_can_predict():
    pipeline = pickle.loads(Path("artifacts/pipeline.pkl").read_bytes())

    user_id = next(iter(pipeline["user_to_idx"].keys()))
    movie_id = next(iter(pipeline["movie_to_idx"].keys()))

    pred, strategy = predict_final(
        user_id, movie_id,
        pipeline["user_to_idx"],
        pipeline["movie_to_idx"],
        pipeline["user_factors"],
        pipeline["movie_factors"],
        pipeline["user_means"],
        pipeline["movie_embeddings_dict"],
        pipeline["movie_means"],
        pipeline["global_mean_rating"],
        pipeline["hybrid_model"],
        pipeline["hybrid_scaler"],
    )

    assert 0.5 <= float(pred) <= 5.0
    assert isinstance(strategy, str)

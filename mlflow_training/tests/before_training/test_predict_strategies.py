"""Module tests for prediction strategy selection used in the training pipeline.

This test module verifies the behavior of the `predict_final` function, which
combines collaborative and content/hybrid components to produce a final
prediction and a string describing which prediction strategy was used.

The tests use pytest fixtures (defined in the surrounding test package) such
as `tiny_factors`, `fake_embeddings`, and `tiny_hybrid_model` to provide small
deterministic inputs. The assertions check that:
- the returned `strategy` string matches the expected strategy for a given
    (user_id, movie_id) scenario (e.g. known user & known movie -> hybrid_full),
- the numeric prediction is in a valid rating range (0.5 to 5.0).

This header is intentionally high-level to help future readers quickly
understand the intent of these unit tests.
"""

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from train_mlflow import predict_final

@pytest.fixture
def tiny_hybrid_model():
    scaler = StandardScaler()
    X = np.array([[0.1, 0.0, 0.2, 0.1], [0.2, 0.2, 0.1, 0.0]])
    scaler.fit(X)
    model = SGDRegressor(random_state=42)
    model.partial_fit(scaler.transform(X), np.array([4.0, 3.0]))
    return model, scaler

@pytest.mark.fast
@pytest.mark.parametrize(
    "user_id, movie_id, expected_strategy",
    [
        (1, 10, "hybrid_full"),
        (1, 999, "hybrid_new_movie"),
        (999, 10, "hybrid_new_user"),
        (999, 777, "fallback_global_mean"),
    ]
)
def test_predict_final_strategies(user_id, movie_id, expected_strategy, tiny_factors, fake_embeddings, tiny_hybrid_model):
    user_to_idx, movie_to_idx, user_factors, movie_factors, user_means, movie_means, global_mean = tiny_factors
    hybrid_model, hybrid_scaler = tiny_hybrid_model

    pred, strategy = predict_final(
        user_id, movie_id,
        user_to_idx, movie_to_idx,
        user_factors, movie_factors, user_means,
        fake_embeddings, movie_means, global_mean,
        hybrid_model, hybrid_scaler
    )

    assert strategy == expected_strategy
    assert 0.5 <= float(pred) <= 5.0

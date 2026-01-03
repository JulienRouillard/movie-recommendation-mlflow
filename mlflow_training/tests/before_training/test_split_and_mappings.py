import pytest
import pandas as pd
from train_mlflow import temporal_split, create_mappings

@pytest.mark.fast
def test_temporal_split_order():
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3],
        "movieId": [10, 20, 10, 30, 20],
        "rating": [4.0, 5.0, 3.0, 4.5, 2.5],
        "timestamp": [100, 200, 150, 250, 300],
    })

    train, test = temporal_split(ratings_df, test_ratio=0.4)
    assert len(train) + len(test) == len(ratings_df)
    if len(train) > 0 and len(test) > 0:
        assert train["timestamp"].max() <= test["timestamp"].min()

@pytest.mark.fast
def test_create_mappings_bijection():
    train_df = pd.DataFrame({
        "userId": [1, 2, 2],
        "movieId": [10, 10, 20],
        "rating": [4.0, 3.0, 5.0],
        "timestamp": [100, 150, 200],
    })

    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = create_mappings(train_df)

    for u, idx in user_to_idx.items():
        assert idx_to_user[idx] == u
    for m, idx in movie_to_idx.items():
        assert idx_to_movie[idx] == m

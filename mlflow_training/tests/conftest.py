import pytest
import numpy as np

@pytest.fixture
def tiny_factors():
    user_to_idx = {1: 0, 2: 1}
    movie_to_idx = {10: 0, 20: 1}
    user_factors = np.array([[0.2, 0.1], [0.0, 0.3]])
    movie_factors = np.array([[0.1, -0.1], [0.2, 0.0]])
    user_means = np.array([3.5, 4.0])
    movie_means = {10: 3.8, 20: 4.2}
    global_mean = 3.9
    return user_to_idx, movie_to_idx, user_factors, movie_factors, user_means, movie_means, global_mean

@pytest.fixture
def fake_embeddings():
    return {
        10: np.array([0.1, 0.0, 0.2]),
        20: np.array([0.0, 0.2, 0.1]),
        999: np.array([0.2, 0.1, 0.0]),
    }

import numpy as np

from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS, compute_history_features, numeric_feature_vector
from rps13.game.env import RPS13Env


def test_features_empty_history():
    features = compute_history_features([])
    vector = numeric_feature_vector([])
    assert len(vector) == len(NUMERIC_FEATURE_KEYS)
    assert np.isfinite(vector).all()
    assert features["last_human_move"] == 3


def test_features_populated_history():
    env = RPS13Env()
    env.step(1, 0)
    env.step(2, 1)
    vector = numeric_feature_vector(env.history)
    assert len(vector) == len(NUMERIC_FEATURE_KEYS)
    assert np.isfinite(vector).all()

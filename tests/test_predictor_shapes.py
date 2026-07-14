import numpy as np
import torch

from rps13.agents.hybrid_agent import HybridAdaptiveAgent
from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS
from rps13.game.env import RPS13Env
from rps13.models.opponent_predictor import OpponentPredictorGRU


def test_predictor_forward_shape():
    batch = 4
    seq = 6
    model = OpponentPredictorGRU()
    logits, hidden = model(
        torch.full((batch, seq), 3, dtype=torch.long),
        torch.full((batch, seq), 3, dtype=torch.long),
        torch.full((batch, seq), 3, dtype=torch.long),
        torch.zeros((batch, seq, len(NUMERIC_FEATURE_KEYS)), dtype=torch.float32),
        torch.ones(batch, dtype=torch.long),
    )
    assert logits.shape == (batch, 3)
    assert hidden.shape[-1] == model.hidden_dim


def test_predictor_predict_proba_valid():
    model = OpponentPredictorGRU()
    probs = model.predict_proba([], sequence_length=5)
    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)


def test_hybrid_policy_valid():
    env = RPS13Env()
    agent = HybridAdaptiveAgent(seed=1)
    policy = agent.policy(env.get_observation())
    assert policy.shape == (3,)
    assert np.isclose(policy.sum(), 1.0)
    assert (policy >= 0).all()

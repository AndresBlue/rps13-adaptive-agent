"""Neural model definitions."""

from rps13.models.actor_critic import ActorCriticGRU
from rps13.models.opponent_predictor import OpponentPredictorGRU

__all__ = ["ActorCriticGRU", "OpponentPredictorGRU"]

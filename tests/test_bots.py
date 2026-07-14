from rps13.bots.bot_population import BOT_REGISTRY, build_bot
from rps13.constants import VALID_ACTIONS
from rps13.game.env import RPS13Env


def test_random_bot_actions_valid():
    bot = build_bot("random", seed=1)
    env = RPS13Env()
    for _ in range(50):
        assert bot.choose_action(env.get_observation()) in [int(action) for action in VALID_ACTIONS]


def test_all_bots_actions_valid_empty_and_populated_history():
    valid = [int(action) for action in VALID_ACTIONS]
    for name in BOT_REGISTRY:
        bot = build_bot(name, seed=2)
        env = RPS13Env()
        assert bot.choose_action(env.get_observation()) in valid
        env.step(0, 1)
        assert bot.choose_action(env.get_observation()) in valid

"""Bot registry and config-driven construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rps13.agents.base import BaseOpponent
from rps13.bots.adaptive_bots import (
    AdaptiveCounterBot,
    MixedBiasBot,
    NoisyHumanLikeBot,
    PressureBot,
    ReverseThenCopyBot,
    SwitchingStrategyBot,
)
from rps13.bots.extended_bots import (
    BrockbankOppCopyBot,
    BrockbankOppPlusBot,
    BrockbankSelfMinusBot,
    BrockbankSelfPlusBot,
    HumanReplayBot,
    Period2ABBot,
    Period3ABCBot,
    Period4Bot,
    PseudoRandomBiasedBot,
    StickyPaperBot,
    StickyRockBot,
    StickyScissorsBot,
    WinChangeLoseStayBot,
    WSLCClassicBot,
)
from rps13.bots.simple_bots import (
    AntiLastMoveBot,
    AvoidRepeatBot,
    BiasedPaperBot,
    BiasedRockBot,
    BiasedScissorsBot,
    CopyOpponentBot,
    CycleBot,
    LoseStayWinShiftBot,
    RandomBot,
    ReverseCycleBot,
    WinStayLoseShiftBot,
)

BOT_REGISTRY: dict[str, type[BaseOpponent]] = {
    "random": RandomBot,
    "biased_rock": BiasedRockBot,
    "biased_paper": BiasedPaperBot,
    "biased_scissors": BiasedScissorsBot,
    "win_stay_lose_shift": WinStayLoseShiftBot,
    "lose_stay_win_shift": LoseStayWinShiftBot,
    "cycle": CycleBot,
    "reverse_cycle": ReverseCycleBot,
    "copy_opponent": CopyOpponentBot,
    "anti_last_move": AntiLastMoveBot,
    "avoid_repeat": AvoidRepeatBot,
    "pressure": PressureBot,
    "noisy_human_like": NoisyHumanLikeBot,
    "switching_strategy": SwitchingStrategyBot,
    "adaptive_counter": AdaptiveCounterBot,
    "mixed_bias": MixedBiasBot,
    "reverse_then_copy": ReverseThenCopyBot,
    "sticky_rock": StickyRockBot,
    "sticky_paper": StickyPaperBot,
    "sticky_scissors": StickyScissorsBot,
    "period2_ab": Period2ABBot,
    "period3_abc": Period3ABCBot,
    "period4": Period4Bot,
    "brockbank_self_plus": BrockbankSelfPlusBot,
    "brockbank_self_minus": BrockbankSelfMinusBot,
    "brockbank_opp_plus": BrockbankOppPlusBot,
    "brockbank_opp_copy": BrockbankOppCopyBot,
    "wslc_classic": WSLCClassicBot,
    "win_change_lose_stay": WinChangeLoseStayBot,
    "pseudo_random_biased": PseudoRandomBiasedBot,
    "human_replay": HumanReplayBot,
}

DEFAULT_BOT_NAMES = list(BOT_REGISTRY.keys())


def build_bot(name: str, seed: int | None = None, **kwargs: Any) -> BaseOpponent:
    """Instantiate a registered bot."""

    if name not in BOT_REGISTRY:
        available = ", ".join(sorted(BOT_REGISTRY))
        raise KeyError(f"Unknown bot {name!r}. Available: {available}")
    return BOT_REGISTRY[name](seed=seed, **kwargs)


def build_bot_population(
    config: Iterable[str | dict[str, Any]] | None = None,
    seed: int | None = None,
) -> list[BaseOpponent]:
    """Build a list of bots from names or config dictionaries."""

    items = list(config) if config is not None else DEFAULT_BOT_NAMES
    bots: list[BaseOpponent] = []
    for idx, item in enumerate(items):
        bot_seed = None if seed is None else seed + idx
        if isinstance(item, str):
            bots.append(build_bot(item, seed=bot_seed))
        else:
            params = dict(item)
            name = params.pop("name")
            params.setdefault("seed", bot_seed)
            bots.append(build_bot(name, **params))
    return bots

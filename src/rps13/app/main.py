"""FastAPI app for playing RPS-13 against the adaptive agent."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rps13.agents.base import BaseAgent
from rps13.agents.factory import AGENT_VERSION, build_agent
from rps13.agents.session_memory import SessionPlayerMemory
from rps13.constants import ACTION_LABELS_ES, ACTION_NAMES, Action
from rps13.game.env import RPS13Env
from rps13.utils.io import append_jsonl, load_yaml


class PlayRoundRequest(BaseModel):
    human_move: int = Field(ge=0, le=2)


@dataclass
class GameSession:
    session_id: str
    match_id: str
    env: RPS13Env
    agent: BaseAgent
    log_path: Path
    player_memory: SessionPlayerMemory = field(default_factory=SessionPlayerMemory)
    user_agent: str | None = None
    match_started_at: str = ""
    match_end_logged: bool = False
    last_seen: float = field(default_factory=time.time)


STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"
SESSION_COOKIE = "rps13_sid"
SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_SESSIONS = 200


def _load_app_config() -> dict[str, Any]:
    config_path = Path(os.environ.get("RPS13_CONFIG", str(CONFIG_PATH)))
    if config_path.exists():
        return load_yaml(config_path)
    return {}


def _root_prefix() -> str:
    raw = os.environ.get("RPS13_ROOT_PATH", CONFIG.get("root_path", "")).strip()
    if not raw or raw == "/":
        return ""
    return "/" + raw.strip("/")


CONFIG = _load_app_config()
ROOT_PREFIX = _root_prefix()
SESSIONS: dict[str, GameSession] = {}
LOGGER = logging.getLogger(__name__)


def _agent_version() -> str:
    return str(CONFIG.get("agent_version", AGENT_VERSION))


def _route(path: str) -> str:
    if path == "/":
        return f"{ROOT_PREFIX}/" if ROOT_PREFIX else "/"
    return f"{ROOT_PREFIX}{path}"


def _cookie_path() -> str:
    return f"{ROOT_PREFIX}/" if ROOT_PREFIX else "/"


def _rooted(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _count_human_logs() -> int:
    logs_dir = _rooted(CONFIG.get("human_logs_dir", "data/human_logs"))
    if not logs_dir.exists():
        return 0
    return len(list(logs_dir.glob("*.jsonl")))


def _purge_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, session in SESSIONS.items() if now - session.last_seen > SESSION_TTL_SECONDS]
    for sid in stale:
        SESSIONS.pop(sid, None)
    if len(SESSIONS) <= MAX_SESSIONS:
        return
    ordered = sorted(SESSIONS.items(), key=lambda item: item[1].last_seen)
    for sid, _ in ordered[: max(0, len(SESSIONS) - MAX_SESSIONS)]:
        SESSIONS.pop(sid, None)


def _archive_match_to_memory(session: GameSession) -> None:
    """Persist finished (or interrupted) match patterns for the next round."""

    history = list(session.env.history)
    if not history:
        return
    expert_scores: dict[str, float] | None = None
    expert_hits: dict[str, list[int]] | None = None
    if hasattr(session.agent, "export_session_learning"):
        expert_scores, expert_hits = session.agent.export_session_learning()
    session.player_memory.archive_match(history, expert_scores=expert_scores, expert_hits=expert_hits)


def _seed_agent_from_memory(agent: BaseAgent, memory: SessionPlayerMemory) -> None:
    if memory.matches_played <= 0:
        return
    if hasattr(agent, "seed_from_session"):
        agent.seed_from_session(memory)


def _build_match(session_id: str, user_agent: str | None, memory: SessionPlayerMemory | None = None) -> GameSession:
    target_score = int(CONFIG.get("target_score", 13))
    logs_dir = _rooted(CONFIG.get("human_logs_dir", "data/human_logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    match_id = f"human_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"
    agent = build_agent(CONFIG, project_root=PROJECT_ROOT)
    player_memory = memory or SessionPlayerMemory()
    _seed_agent_from_memory(agent, player_memory)
    return GameSession(
        session_id=session_id,
        match_id=match_id,
        env=RPS13Env(target_score=target_score),
        agent=agent,
        log_path=logs_dir / f"{match_id}.jsonl",
        player_memory=player_memory,
        user_agent=user_agent,
        match_started_at=datetime.now(timezone.utc).isoformat(),
    )


def create_session(user_agent: str | None = None) -> GameSession:
    """Create a fresh human-vs-agent session."""

    _purge_stale_sessions()
    session_id = str(uuid.uuid4())
    return _build_match(session_id, user_agent)


def start_new_match(session: GameSession) -> GameSession:
    """Start another match in the same browser session, learning from prior play."""

    _archive_match_to_memory(session)
    memory = session.player_memory
    user_agent = session.user_agent
    return _build_match(session.session_id, user_agent, memory=memory)


def _append_log(session: GameSession, row: dict[str, Any]) -> str | None:
    try:
        append_jsonl(session.log_path, row)
        return None
    except OSError as exc:
        LOGGER.warning("Could not write human log to %s: %s", session.log_path, exc)
        return str(exc)


def _maybe_log_match_end(session: GameSession) -> str | None:
    if session.match_end_logged or not session.env.is_done():
        return None
    winner = session.env.winner()
    row = {
        "event": "match_end",
        "session_id": session.session_id,
        "match_id": session.match_id,
        "agent_version": _agent_version(),
        "agent_type": str(CONFIG.get("agent_type", "hybrid")),
        "winner": winner,
        "status": "ganaste" if winner == "human" else "perdiste" if winner == "ai" else "empate",
        "human_score": session.env.human_score,
        "ai_score": session.env.ai_score,
        "rounds_played": len(session.env.history),
        "match_started_at": session.match_started_at,
        "match_ended_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": session.user_agent,
        "session_matches_played": session.player_memory.matches_played,
        "session_prior_rounds": len(session.player_memory.prior_rounds),
    }
    session.player_memory.record_set_win(winner)
    row["session_sets"] = session.player_memory.sets_summary()
    session.match_end_logged = True
    return _append_log(session, row)


def _set_session_cookie(response: Response, session_id: str) -> None:
    secure_env = os.environ.get("RPS13_COOKIE_SECURE", "").strip().lower()
    if secure_env in {"1", "true", "yes"}:
        secure = True
    elif secure_env in {"0", "false", "no"}:
        secure = False
    else:
        secure = bool(ROOT_PREFIX)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=secure,
        path=_cookie_path(),
        max_age=SESSION_TTL_SECONDS,
    )


def get_session(session_id: str | None, *, create: bool = False, user_agent: str | None = None) -> GameSession:
    _purge_stale_sessions()
    if session_id and session_id in SESSIONS:
        session = SESSIONS[session_id]
        session.last_seen = time.time()
        if user_agent and not session.user_agent:
            session.user_agent = user_agent
        return session
    if not create:
        raise HTTPException(status_code=404, detail="Sesión no encontrada. Inicia una partida nueva.")
    session = create_session(user_agent=user_agent)
    SESSIONS[session.session_id] = session
    return session


def serialize_state(session: GameSession) -> dict[str, Any]:
    obs = session.env.get_observation()
    winner = session.env.winner()
    state = obs.to_dict()
    state.update(
        {
            "session_id": session.session_id,
            "match_id": session.match_id,
            "winner": winner,
            "status": "ganaste" if winner == "human" else "perdiste" if winner == "ai" else "en_curso",
            "actions": {int(action): ACTION_LABELS_ES[action] for action in Action},
            "root_path": ROOT_PREFIX or "/",
            "agent_version": _agent_version(),
            "session_matches_played": session.player_memory.matches_played,
            "session_prior_rounds": len(session.player_memory.prior_rounds),
            "session_sets": session.player_memory.sets_summary(),
        }
    )
    return state


def create_app() -> FastAPI:
    """Build the FastAPI app."""

    app = FastAPI(title="ACG — RPS-13 Adaptive Agent")
    app.mount(_route("/static"), StaticFiles(directory=STATIC_DIR), name="static")

    if ROOT_PREFIX:

        @app.get(ROOT_PREFIX)
        def index_no_slash() -> RedirectResponse:
            return RedirectResponse(url=f"{ROOT_PREFIX}/", status_code=307)

    @app.get(_route("/"))
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get(_route("/api/new_game"))
    def new_game(
        response: Response,
        user_agent: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session = get_session(None, create=True, user_agent=user_agent)
        _set_session_cookie(response, session.session_id)
        return serialize_state(session)

    @app.post(_route("/api/reset"))
    def reset(
        response: Response,
        user_agent: str | None = Header(default=None),
        rps13_sid: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        if rps13_sid and rps13_sid in SESSIONS:
            session = start_new_match(SESSIONS[rps13_sid])
        else:
            session = create_session(user_agent=user_agent)
        SESSIONS[session.session_id] = session
        _set_session_cookie(response, session.session_id)
        state = serialize_state(session)
        state["session_learning"] = session.player_memory.summary()
        return state

    @app.get(_route("/api/game_state"))
    def game_state(
        rps13_sid: str | None = Cookie(default=None),
        user_agent: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session = get_session(rps13_sid, create=True, user_agent=user_agent)
        return serialize_state(session)

    @app.post(_route("/api/play_round"))
    def play_round(
        payload: PlayRoundRequest,
        response: Response,
        user_agent: str | None = Header(default=None),
        rps13_sid: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        session = get_session(rps13_sid, create=True, user_agent=user_agent)
        _set_session_cookie(response, session.session_id)
        if session.env.is_done():
            raise HTTPException(status_code=400, detail="Game is already finished. Reset to play again.")
        session.user_agent = session.user_agent or user_agent
        obs = session.env.get_observation()
        decision = session.agent.select_action(obs)
        next_obs, _reward, _done, info = session.env.step(decision.action, payload.human_move)
        if hasattr(session.agent, "observe_round"):
            session.agent.observe_round(payload.human_move)
        debug = decision.debug or {}
        log_row = {
            "event": "round",
            "session_id": session.session_id,
            "match_id": session.match_id,
            "agent_version": _agent_version(),
            "agent_type": str(CONFIG.get("agent_type", "hybrid")),
            "round": info["round"],
            "human_move": info["human_move"],
            "ai_move": info["ai_move"],
            "human_move_name": ACTION_NAMES[Action(info["human_move"])],
            "ai_move_name": ACTION_NAMES[Action(info["ai_move"])],
            "result_for_ai": info["result_for_ai"],
            "human_score": info["human_score"],
            "ai_score": info["ai_score"],
            "agent_policy": decision.policy,
            "agent_prediction": debug.get("predicted_probs"),
            "agent_confidence": debug.get("confidence"),
            "expert_chosen": debug.get("expert_chosen"),
            "pattern_flags": debug.get("pattern_flags"),
            "pattern_stability": debug.get("pattern_stability"),
            "session_matches_played": session.player_memory.matches_played,
            "session_prior_rounds": len(session.player_memory.prior_rounds),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_agent": session.user_agent,
        }
        log_error = _append_log(session, log_row)
        match_end_error = _maybe_log_match_end(session)
        state = serialize_state(session)
        state["last_round"] = log_row
        state["log_error"] = log_error or match_end_error
        state["history"] = [record.to_dict() for record in next_obs.history]
        return state

    @app.get(_route("/api/health"))
    def health() -> dict[str, Any]:
        checkpoint = _rooted(CONFIG.get("agent_checkpoint_path", "models/opponent_predictor.pt"))
        return {
            "ok": True,
            "root_path": ROOT_PREFIX or "/",
            "active_sessions": len(SESSIONS),
            "checkpoint_exists": checkpoint.exists(),
            "agent_version": _agent_version(),
            "agent_type": str(CONFIG.get("agent_type", "hybrid")),
            "human_log_files": _count_human_logs(),
        }

    return app


app = create_app()

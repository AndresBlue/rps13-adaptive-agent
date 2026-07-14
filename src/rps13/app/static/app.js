const labels = ["Piedra", "Papel", "Tijera"];
const glyphs = ["✊", "✋", "✌️"];
const resultLabels = {
  "1": "Gana la IA",
  "0": "Empate",
  "-1": "Ganas tu",
};
const TARGET_SCORE = 13;
const KEY_MOVES = {
  "1": 0,
  "2": 1,
  "3": 2,
  r: 0,
  p: 1,
  t: 2,
};

const state = {
  done: false,
  selectedMove: null,
  animating: false,
  current: null,
};

function apiUrl(path) {
  const rel = String(path).replace(/^\//, "");
  return new URL(rel, document.baseURI).href;
}

async function api(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function moveGlyph(move) {
  return move === undefined || move === null ? "—" : glyphs[move];
}

function resultClass(resultForAi) {
  const key = String(resultForAi);
  if (key === "-1") return "row-win";
  if (key === "1") return "row-loss";
  return "row-draw";
}

function resultBadgeClass(resultForAi) {
  const key = String(resultForAi);
  if (key === "-1") return "badge-win";
  if (key === "1") return "badge-loss";
  return "badge-draw";
}

function renderSessionSets(sets) {
  const summary = sets || { human: 0, ai: 0, total: 0, streak: 0, streak_holder: null };
  document.getElementById("stat-human-sets").textContent = summary.human ?? 0;
  document.getElementById("stat-ai-sets").textContent = summary.ai ?? 0;
  document.getElementById("stat-total-sets").textContent = summary.total ?? 0;

  const streakEl = document.getElementById("stat-set-streak");
  if (!summary.total) {
    streakEl.textContent = "—";
    streakEl.className = "";
    return;
  }
  if (summary.streak_holder === "human") {
    streakEl.textContent = `${summary.streak}W`;
    streakEl.className = "streak-win";
  } else if (summary.streak_holder === "ai") {
    streakEl.textContent = `${summary.streak}L`;
    streakEl.className = "streak-loss";
  } else {
    streakEl.textContent = "—";
    streakEl.className = "";
  }
}

function setStatus(data) {
  const status = document.getElementById("status");
  status.classList.remove("status-live", "status-win", "status-loss");
  if (data.status === "ganaste") {
    status.textContent = "Ganaste";
    status.classList.add("status-win");
  } else if (data.status === "perdiste") {
    status.textContent = "Perdiste";
    status.classList.add("status-loss");
  } else {
    status.textContent = "En curso";
    status.classList.add("status-live");
  }
}

function setRaceBars(data, animate = false) {
  const humanPct = Math.min(100, (data.human_score / TARGET_SCORE) * 100);
  const aiPct = Math.min(100, (data.ai_score / TARGET_SCORE) * 100);
  const humanBar = document.getElementById("human-bar");
  const aiBar = document.getElementById("ai-bar");
  humanBar.style.width = `${humanPct}%`;
  aiBar.style.width = `${aiPct}%`;
  document.getElementById("human-bar-label").textContent = `${data.human_score}/${TARGET_SCORE}`;
  document.getElementById("ai-bar-label").textContent = `${data.ai_score}/${TARGET_SCORE}`;
  if (animate) {
    [humanBar, aiBar].forEach((bar) => {
      bar.classList.remove("bar-fill");
      window.requestAnimationFrame(() => bar.classList.add("bar-fill"));
      setTimeout(() => bar.classList.remove("bar-fill"), 500);
    });
  }
}

function setScore(data, animate = false) {
  const humanScore = document.getElementById("human-score");
  const aiScore = document.getElementById("ai-score");
  const roundNumber = document.getElementById("round-number");
  humanScore.textContent = data.human_score;
  aiScore.textContent = data.ai_score;
  roundNumber.textContent = data.done ? data.round_number - 1 : data.round_number;
  setRaceBars(data, animate);
  if (animate) {
    [humanScore, aiScore, roundNumber].forEach((node) => {
      node.classList.remove("bump");
      window.requestAnimationFrame(() => node.classList.add("bump"));
      setTimeout(() => node.classList.remove("bump"), 420);
    });
  }
}

function setReveal(human = "—", ai = "—", result = "—", stage = "", round = null) {
  document.getElementById("human-choice").textContent = human;
  document.getElementById("ai-choice").textContent = ai;
  document.getElementById("round-result").textContent = result;

  const showHumanGlyph = stage === "human" || stage === "ai" || stage === "result";
  const showAiGlyph = stage === "ai" || stage === "result";
  document.getElementById("human-glyph").textContent =
    showHumanGlyph && round ? moveGlyph(round.human_move) : "—";
  document.getElementById("ai-glyph").textContent =
    showAiGlyph && round ? moveGlyph(round.ai_move) : "—";

  const humanCard = document.getElementById("human-card");
  const aiCard = document.getElementById("ai-card");
  const resultCard = document.getElementById("result-card");
  [humanCard, aiCard, resultCard].forEach((card) => {
    card.classList.remove("active", "win", "loss", "draw");
  });

  if (stage === "human") humanCard.classList.add("active");
  if (stage === "ai") {
    humanCard.classList.add("active");
    aiCard.classList.add("active");
  }
  if (stage === "result") {
    humanCard.classList.add("active");
    aiCard.classList.add("active");
    resultCard.classList.add("active");
    if (round) {
      const key = String(round.result_for_ai);
      if (key === "-1") {
        humanCard.classList.add("win");
        resultCard.classList.add("win");
      } else if (key === "1") {
        aiCard.classList.add("loss");
        resultCard.classList.add("loss");
      } else {
        resultCard.classList.add("draw");
      }
    }
  }
}

function renderHistory(history, options = {}) {
  const tbody = document.getElementById("history");
  tbody.innerHTML = "";
  const rows = [...(history || [])].reverse();
  document.getElementById("history-count").textContent =
    rows.length === 1 ? "1 ronda" : `${rows.length} rondas`;

  rows.forEach((round, index) => {
    const tr = document.createElement("tr");
    tr.className = resultClass(round.result_for_ai);
    if (options.animateLatest && index === 0) tr.classList.add("row-enter");
    tr.innerHTML = `
      <td>${round.round}</td>
      <td><span class="move-cell">${moveGlyph(round.human_move)} ${labels[round.human_move]}</span></td>
      <td><span class="move-cell">${moveGlyph(round.ai_move)} ${labels[round.ai_move]}</span></td>
      <td><span class="result-badge ${resultBadgeClass(round.result_for_ai)}">${resultLabels[round.result_for_ai]}</span></td>
      <td>${round.human_score} — ${round.ai_score}</td>
    `;
    tbody.appendChild(tr);
  });
}

function scrollHistoryIntoView() {
  const wrap = document.querySelector(".history-scroll");
  if (wrap) wrap.scrollTop = 0;
}

function updateControls() {
  const disabled = state.done || state.animating;
  document.querySelectorAll(".move").forEach((button) => {
    const move = Number(button.dataset.move);
    button.disabled = disabled;
    button.classList.toggle("selected", state.selectedMove === move);
  });
  const confirm = document.getElementById("confirm");
  confirm.disabled = disabled || state.selectedMove === null;
  const selection = document.getElementById("selection");
  if (disabled) {
    selection.textContent = state.animating ? "Resolviendo ronda..." : "Partida finalizada";
  } else if (state.selectedMove === null) {
    selection.textContent = "Clic en una jugada para jugar · 1/2/3 o R/P/T";
  } else {
    selection.textContent = `Seleccionaste ${labels[state.selectedMove]} · Enter para confirmar`;
  }
}

function setResultLine(text, pulse = false) {
  const line = document.getElementById("result-line");
  line.textContent = text;
  line.classList.toggle("pulse", pulse);
  if (pulse) setTimeout(() => line.classList.remove("pulse"), 600);
}

function showOverlay(data) {
  const overlay = document.getElementById("match-overlay");
  const card = document.getElementById("overlay-card");
  const won = data.status === "ganaste";
  card.classList.remove("overlay-win", "overlay-loss");
  card.classList.add(won ? "overlay-win" : "overlay-loss");
  document.getElementById("overlay-title").textContent = won ? "Victoria" : "Derrota";
  document.getElementById("overlay-score").textContent = `${data.human_score} — ${data.ai_score}`;
  document.getElementById("overlay-detail").textContent = won
    ? "Llegaste a 13 antes que la IA."
    : "La IA llego a 13 primero.";
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  document.getElementById("match-overlay").classList.add("hidden");
}

function render(data, options = {}) {
  state.current = data;
  setScore(data, Boolean(options.animateScore));
  setStatus(data);
  state.done = data.done;

  if (data.last_round) {
    const human = labels[data.last_round.human_move];
    const ai = labels[data.last_round.ai_move];
    setResultLine(
      `${resultLabels[data.last_round.result_for_ai]}: jugaste ${human}, la IA jugo ${ai}.`,
      Boolean(options.pulseResult)
    );
  } else if (data.done) {
    setResultLine(
      data.status === "ganaste" ? "Partida finalizada: ganaste." : "Partida finalizada: gano la IA."
    );
  } else {
    setResultLine("Elige una jugada para jugar.");
  }

  renderHistory(data.history, { animateLatest: Boolean(options.animateHistory) });
  renderSessionSets(data.session_sets);
  if (!options.keepReveal) setReveal();
  updateControls();

  if (options.showOverlay && data.done) showOverlay(data);
  else if (!data.done) hideOverlay();
}

async function newGame() {
  const data = await api("/api/new_game");
  state.selectedMove = null;
  state.animating = false;
  hideOverlay();
  render(data);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function playRound(humanMove) {
  if (state.animating || state.done) return;
  const previous = state.current;
  state.animating = true;
  state.selectedMove = humanMove;
  updateControls();
  try {
    const data = await api("/api/play_round", {
      method: "POST",
      body: JSON.stringify({ human_move: humanMove }),
    });
    state.current = data;
    await animateRound(previous, data);
    state.selectedMove = null;
    state.animating = false;
    render(data, { keepReveal: true, animateHistory: true, pulseResult: true, showOverlay: true });
    scrollHistoryIntoView();
  } catch (error) {
    state.animating = false;
    setResultLine(error.message);
    updateControls();
  }
}

async function animateRound(previous, data) {
  const round = data.last_round;
  const previousScore = previous || data;
  setScore(previousScore);
  renderHistory(previous ? previous.history : []);
  setReveal();
  setResultLine("Confirmando jugada...");
  await wait(260);
  setReveal(labels[round.human_move], "—", "—", "human", round);
  setResultLine("Tu jugada esta bloqueada.");
  await wait(700);
  setReveal(labels[round.human_move], labels[round.ai_move], "—", "ai", round);
  setResultLine("La IA revela su jugada.");
  await wait(800);
  setReveal(
    labels[round.human_move],
    labels[round.ai_move],
    resultLabels[round.result_for_ai],
    "result",
    round
  );
  setResultLine(resultLabels[round.result_for_ai], true);
  await wait(760);
  setScore(data, true);
  renderHistory(data.history, { animateLatest: true });
  await wait(220);
}

function selectMove(move) {
  if (state.done || state.animating) return;
  state.selectedMove = move;
  updateControls();
}

document.querySelectorAll(".move").forEach((button) => {
  button.addEventListener("click", () => {
    const move = Number(button.dataset.move);
    playRound(move);
  });
});

document.getElementById("confirm").addEventListener("click", () => {
  if (state.selectedMove === null || state.animating || state.done) return;
  playRound(state.selectedMove);
});

document.getElementById("reset").addEventListener("click", async () => {
  const data = await api("/api/reset", { method: "POST" });
  state.selectedMove = null;
  state.animating = false;
  hideOverlay();
  render(data);
});

document.getElementById("overlay-revancha").addEventListener("click", async () => {
  const data = await api("/api/reset", { method: "POST" });
  state.selectedMove = null;
  state.animating = false;
  hideOverlay();
  render(data);
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  const key = event.key.toLowerCase();

  if (key === "escape") {
    if (state.animating || state.done) return;
    state.selectedMove = null;
    updateControls();
    return;
  }

  if (key === "enter" && state.selectedMove !== null && !state.animating && !state.done) {
    event.preventDefault();
    playRound(state.selectedMove);
    return;
  }

  if (KEY_MOVES[key] !== undefined && !state.animating && !state.done) {
    event.preventDefault();
    if (event.shiftKey) {
      selectMove(KEY_MOVES[key]);
    } else {
      playRound(KEY_MOVES[key]);
    }
  }
});

newGame().catch((error) => {
  setResultLine(error.message);
});

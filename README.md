# RPS-13 Adaptive Agent

RPS-13 Adaptive Agent is a local Python project for training, evaluating and
playing against an adaptive rock-paper-scissors AI. Matches are 1v1 and end
when either player reaches 13 points.

Rock-paper-scissors has a trivial Nash equilibrium: play each action with
probability 1/3. That is unbeatable against a perfectly random opponent, but it
does not exploit human patterns. This project focuses on the practical problem:
detect biased, cyclic, reactive or pressure-dependent human behavior, exploit it
softly, and fall back near uniform play when there is no reliable signal.

## Architecture

- `rps13.game`: deterministic rules, `RPS13Env`, and optional `RPS13PlusEnv`.
- `rps13.bots`: scripted opponent population for training and evaluation.
- `rps13.data`: synthetic generation, external CSV normalization and features.
- `rps13.models`: `OpponentPredictorGRU` and `ActorCriticGRU`.
- `rps13.training`: supervised predictor training and a modular RL baseline.
- `rps13.agents`: mixture (production), hybrid (v1), neural and random agents.
- `rps13.experts`: heuristic + neural predictors for the mixture ensemble.
- `rps13.evaluation`: bot evaluation metrics and plots.
- `rps13.app`: FastAPI backend plus HTML/CSS/JS frontend.

### How inference works (MixtureAdaptiveAgent v2)

Production uses a **mixture of predictors** (HMOP): several experts guess
`P(human next move)`, one hypothesis is selected online, then the AI mixes a
soft best-response with the Nash uniform policy.

1. **Observe** — merge match history with optional session memory
   (`effective_history`).
2. **Predict** — 11 experts each output `P̂ ∈ R³` (frequencies, cycles, PPM,
   Markov, Brockbank transitions, and a GRU predictor when a checkpoint exists).
3. **Meta + select** — each expert is expanded with Iocaine-style metas
   (`P.0`, `P.1`, `P'.0`). Virtual scores (decayed hit/miss) pick the best
   hypothesis; sticky patterns raise exploit strength.
4. **Policy** — compute action EVs from `P̂`, take
   `soft_best = softmax(EV / T)`, then
   `π = (1−α)·U + α·soft_best` with `α = α_cap · confidence`.
5. **Act** — sample `a_AI ~ π` (floor probabilities so the policy never
   collapses to pure deterministic play).
6. **Adapt** — after the human move is revealed, update expert scores
   (`observe_round`) for the next decision.

![Pipeline de inferencia (D2, 2:1)](docs/inference_pipeline.svg)

Fuente del diagrama: [`docs/inference_pipeline.d2`](docs/inference_pipeline.d2). Regenerar SVG (lienzo 2:1):

```bash
python scripts/render_inference_d2.py
```

Diagramas PNG/PDF detallados (matplotlib):

```bash
python scripts/plot_inference_summary.py
python scripts/plot_inference_architecture.py
```


## Installation

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For editable imports during development:

```bash
pip install -e .
```

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --matches 1000
```

Default output:

```text
data/synthetic/synthetic_matches.csv
```

The CSV contains one row per round, including match id, bot type, human move,
AI move, result, scores and terminal flag.

## Train the Opponent Predictor

```bash
python scripts/train_predictor.py --config configs/train_predictor.yaml
```

Outputs:

- `models/opponent_predictor.pt`
- `reports/metrics/predictor_metrics.json`
- `reports/figures/predictor_loss.png`
- `reports/figures/predictor_accuracy.png`

Metrics include validation accuracy, per-class accuracy, a random baseline and a
majority-class baseline.

## Train the RL Baseline

```bash
python scripts/train_rl.py --config configs/train_rl.yaml
```

This is a simple recurrent actor-critic baseline with entropy regularization,
round reward and terminal match bonus. It is intentionally modular so PPO
clipping, GAE and batched recurrent rollouts can be added later.

Output:

```text
models/actor_critic.pt
reports/metrics/rl_metrics.json
```

## Evaluate

```bash
python scripts/evaluate.py --agent mixture --matches 500
```

Outputs:

- `reports/metrics/evaluation.csv` (or `evaluation_v2.csv` for the gate run)
- `reports/figures/eval_win_rate_by_bot.png`
- `reports/figures/eval_score_diff_by_bot.png`
- `reports/figures/eval_predictor_accuracy_by_bot.png`

Key metrics:

- `win_rate`: match win rate against each bot.
- `round_win_rate`: per-round win rate.
- `draw_rate`: per-round draw rate.
- `average_score_diff`: final AI score minus human score.
- `uniform_deviation`: how far the policy moves from `[1/3, 1/3, 1/3]`.
- `exploitability_proxy_vs_random`: uniform deviation specifically against
  `RandomBot`; lower is better here.
- `prediction_accuracy`: argmax accuracy of predicted human move when available.

### Evaluation results (mixture v2)

Gate run from `reports/metrics/evaluation_v2.csv`: **2000 matches per bot**,
first-to-13, agent `mixture`. Win rate is the fraction of matches won by the AI.

| Bot | Win rate | Avg score diff | Notes |
|-----|----------|----------------|-------|
| `random` | 0.488 | −0.09 | Near 50%: cannot beat pure noise without being exploitable |
| `pseudo_random_biased` | 0.511 | +0.11 | Weak bias; stays close to Nash |
| `noisy_human_like` | 0.535 | +0.34 | Mild human-like noise |
| `avoid_repeat` | 0.675 | +1.92 | Partial pattern, partial exploit |
| `human_replay` | 0.780 | +3.72 | Replay of logged human sequences |
| `biased_rock` | 0.928 | +6.35 | Static action bias |
| `biased_paper` | 0.952 | +6.81 | Static action bias |
| `biased_scissors` | 0.952 | +6.85 | Static action bias |
| `sticky_rock` | 0.996 | +9.55 | Strong sticky habit |
| `sticky_scissors` | 0.999 | +9.74 | Strong sticky habit |
| `sticky_paper` | 0.999 | +9.76 | Strong sticky habit |
| `period4` | 0.999 | +8.98 | Periodic pattern |
| `period2_ab` | 1.000 | +10.40 | Period-2 |
| `wslc_classic` | 1.000 | +9.05 | Win-stay / lose-change |
| `win_change_lose_stay` | 0.998 | +8.54 | Outcome-reactive |
| `cycle` | 1.000 | +10.45 | Fixed cycle |
| `reverse_cycle` | 1.000 | +9.28 | Reverse cycle |
| `period3_abc` | 1.000 | +9.94 | Period-3 |
| `copy_opponent` | 1.000 | +9.10 | Copies last AI move |
| `brockbank_self_plus` | 1.000 | +10.39 | Self transition `+` |
| `brockbank_self_minus` | 1.000 | +9.89 | Self transition `−` |
| `brockbank_opp_plus` | 1.000 | +9.15 | Opp transition `+` |
| `brockbank_opp_copy` | 1.000 | +9.13 | Opp transition copy |

**Reading the table:** patterned and sticky bots are almost always beaten (win
rate ≈ 1.0) because experts lock onto the regularity and soft-BR exploits it.
Against `random` / near-random bots the agent stays near 50%, which is the
intended Nash fallback: strong exploitation only when there is a reliable
signal.

## Run the Local Web App

On Windows, double-click or run:

```bat
start_local.bat
```

That creates `.venv` if needed, installs dependencies, opens the browser and starts the app.

Or manually:

```bash
python scripts/run_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Default config is `configs/app.yaml` (`127.0.0.1:8000`, empty `root_path`).
If `models/opponent_predictor.pt` is missing, the agent falls back to heuristic mode.

Sessions are cookie-scoped so multiple browsers can play at once.

The app lets you play a complete first-to-13 match against the mixture agent (v2 HMOP).
Human rounds are logged automatically to:

```text
data/human_logs/
```

That directory is gitignored. Logs include session id, match id, round, moves, score,
agent policy, prediction, confidence, timestamp and optional user agent.

For a reverse-proxy deploy behind a path prefix, copy
`configs/app.web.yaml.example` to the local-only `configs/app.web.yaml` and point
your proxy at that process. See the local `DEPLOY.md` (gitignored) if present.

## Add Real Datasets

Put external CSV files in:

```text
data/raw/
```

Use `normalize_rps_dataset(input_path, output_path, mapping_config)` from
`rps13.data.dataset_loader` to map arbitrary column names to the internal schema.
No dataset is downloaded automatically, so local datasets such as Brockbank RPS
or hm_rps_public can be added manually.

Minimal mapping example:

```python
from rps13.data.dataset_loader import normalize_rps_dataset

normalize_rps_dataset(
    "data/raw/my_rps.csv",
    "data/processed/my_rps_normalized.csv",
    {
        "match_id": "game_id",
        "round": "turn",
        "human_move": "player_move",
        "ai_move": "opponent_move",
    },
)
```

## Tests

```bash
pytest
```

The test suite covers rules, environment reset/termination, bot action validity,
feature engineering, predictor tensor shapes and hybrid policy validity.

## Limitations

- There is no unbeatable agent against perfect random play.
- The intended behavior is near-uniform play against random opponents and
  careful exploitation against biased or patterned opponents.
- Real performance must be validated with human matches, not only scripted bots.
- The RL module is a functional baseline, not a full recurrent PPO
  implementation yet.

## Expected Minimal Workflow

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_synthetic_data.py --matches 1000
python scripts/train_predictor.py --config configs/train_predictor.yaml
python scripts/evaluate.py --agent mixture --matches 500
python scripts/run_app.py
```

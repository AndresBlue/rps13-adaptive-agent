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
- `rps13.agents`: random, neural and hybrid adaptive agents.
- `rps13.evaluation`: bot evaluation metrics and plots.
- `rps13.app`: FastAPI backend plus HTML/CSS/JS frontend.

The main production path is the hybrid agent:

1. Estimate `P(human next move)` using a trained GRU predictor when available.
2. Use heuristic feature estimates when no checkpoint exists.
3. Compute expected values for ROCK/PAPER/SCISSORS.
4. Mix a soft best response with the uniform Nash policy.
5. Keep a minimum action probability so the agent does not collapse into a
   deterministic strategy.

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
python scripts/evaluate.py --agent hybrid --matches 500
```

Outputs:

- `reports/metrics/evaluation.csv`
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
python scripts/evaluate.py --agent hybrid --matches 500
python scripts/run_app.py
```

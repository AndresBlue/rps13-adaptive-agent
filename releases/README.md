# Releases ACG

Versiones congeladas del agente desplegado en producción.

| Versión | Carpeta | Agente | Checkpoint | URL |
|---------|---------|--------|------------|-----|
| **1.0.0** | `cachipunaa-v1.0.0/` | HybridAdaptiveAgent | `opponent_predictor.pt` (synthetic GRU) | [acg](https://poblete.servehttp.com/acg/) |
| **2.0.0** | `cachipunaa-v2.0.0/` | MixtureAdaptiveAgent (HMOP) | ensemble + `opponent_predictor.pt` | [acg](https://poblete.servehttp.com/acg/) |

> Las carpetas `cachipunaa-v*` son artefactos históricos; el producto se llama **ACG**.

## Rollback v2 → v1

```bat
copy releases\cachipunaa-v1.0.0\opponent_predictor.pt models\
copy releases\cachipunaa-v1.0.0\app.web.yaml configs\
```

Edita `configs\app.web.yaml`: `agent_type: hybrid`, `agent_version: "1.0.0"`, `root_path: /acg`.

## Empaquetar nueva release

```bat
python scripts\package_release.py --version 2.0.0
```

Genera `releases/acg-v2.0.0/` con checkpoint, config, métricas y MODEL_CARD.

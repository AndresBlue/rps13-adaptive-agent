# ACG v2.0.0 — Model Card (Mixture HMOP)

**Codename:** mixture-hmop-v2  
**Agente:** `MixtureAdaptiveAgent`  
**URL:** https://poblete.servehttp.com/acg/

## Cambios vs v1

- Ensemble de 11 experts (ciclo, PPM, Markov, Brockbank, GRU, hedge aleatorio).
- Scoring virtual con decay + meta-niveles P.0 / P.1 / P'.0.
- **Sticky boost:** sobreexplotación cuando ciclo/PPM/frecuencia detectan patrón fijo.
- Logs humanos con `match_end`, `agent_version`, `expert_chosen`, `pattern_flags`.
- 29 bots de evaluación (17 originales + 12 nuevos).

## Métricas gate (evaluation_v2.csv, 2000 matches/bot)

Ver `metrics/evaluation_v2.csv` en este paquete.

Targets:
- cycle ≥ 85%
- random 45–55%
- sticky_* ≥ 90%
- biased_* sin regresión

## Rollback a v1

```bat
copy releases\cachipunaa-v1.0.0\opponent_predictor.pt models\
copy releases\cachipunaa-v1.0.0\app.web.yaml configs\
```

En `app.web.yaml` v1: `agent_type: hybrid`, `agent_version: "1.0.0"`.

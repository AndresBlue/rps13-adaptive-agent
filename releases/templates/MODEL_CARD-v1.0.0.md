# ACG v1.0.0 — Model Card

**Codename:** hybrid-synthetic-v1  
**Desplegado en:** https://poblete.servehttp.com/acg/  
**Fecha de empaquetado:** julio 2026

## Resumen ejecutivo

Esta es la **primera versión congelada** del agente adaptativo de piedra/papel/tijera (primero a 13). Combina:

1. **Predictor GRU** (`OpponentPredictorGRU`) que estima P(jugada humana siguiente).
2. **Política híbrida** que mezcla best-response suavizado con el equilibrio Nash uniforme (1/3, 1/3, 1/3).
3. **Fallback heurístico** si no hay checkpoint (frecuencias, ventanas recientes, patrones cíclicos).

El objetivo de diseño no es “ganar siempre”, sino **explotar patrones detectables** sin volverse determinista ni predecible contra un oponente aleatorio.

---

## Arquitectura en producción

| Componente | Valor |
|------------|-------|
| Agente | `HybridAdaptiveAgent` |
| Checkpoint | `models/opponent_predictor.pt` (1.6 MB) |
| Predictor | GRU, hidden=96, layers=1, seq_len=20 |
| Entrenamiento | Datos sintéticos (1 300 train / 376 val) |
| Política | temperature=0.6, max_alpha=0.85, min_action_prob=0.03 |
| Partida | 1v1, primero a 13 puntos |

### Flujo de decisión

```
Historial → GRU → P(ROCK/PAPER/SCISSORS humano)
         → EV por acción → softmax(temperature)
         → α = max_alpha × confianza
         → π = (1-α)·uniform + α·soft_best
         → clip min_action_prob → muestreo estocástico
```

---

## Métricas de entrenamiento del predictor

Entrenado con `configs/train_predictor.yaml` sobre `data/synthetic/synthetic_matches.csv`.

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Val accuracy** | **46.3%** | +13 pp sobre azar (33.3%) |
| Baseline random | 33.3% | Techo teórico contra oponente perfectamente aleatorio |
| Baseline majority | 31.4% | Clase más frecuente (ROCK) |
| Val loss (final) | 1.029 | Estable tras 8 épocas |
| Acc ROCK | 46.6% | |
| Acc PAPER | 40.3% | Peor clase (más difícil de predecir) |
| Acc SCISSORS | 51.1% | Mejor clase |

**Curva de validación:** accuracy sube de 34.0% (época 1) a **47.1%** (época 7), ligera caída en época 8 → checkpoint final en 46.3%.

### Predictor robusto (no desplegado)

Existe un segundo checkpoint (`opponent_predictor_robust.pt`) entrenado con ~1.6M ejemplos reales/sintéticos (`rps_train_full.csv`, GRU 192×2 capas, seq=40):

| Métrica | Robust | Synthetic (prod) |
|---------|--------|------------------|
| Val accuracy | 43.2% | **46.3%** |
| Train examples | 1 647 378 | 1 300 |
| Mejor val | 43.3% | 47.1% |

**Decisión v1:** se despliega el predictor **sintético** porque en evaluación contra bots el híbrido con ese checkpoint mostró mejor equilibrio exploitabilidad/rendimiento en la batería inicial. El robust queda reservado para v2.

### RL baseline (referencia, no producción)

Actor-critic recurrente (`actor_critic.pt`, 10 000 episodios):

| Métrica | Valor |
|---------|-------|
| Win rate global | 65.4% |
| Last-100 win rate | 85.0% |
| Uniform deviation vs random | **~0.96** (muy alto → casi determinista) |

El actor-critic **explota fuerte** sesgos (99% vs biased_*) pero **colapsa** contra `reverse_cycle` (0.2% win) y no es seguro para humanos. No se usa en la app web.

---

## Evaluación vs estrategias (2 000 partidas/bot)

Fuente: `metrics/evaluation_robust.csv` — agente **hybrid** con `opponent_predictor.pt`.

### Tabla por bot

| Bot | Win rate | Score diff | Pred. acc | Uniform dev | Perfil |
|-----|----------|------------|-----------|-------------|--------|
| **random** | 49.3% | -0.05 | 33.2% | 0.16 | ✅ Casi Nash |
| biased_rock | 85.9% | +4.05 | 60.9% | 0.25 | Explota sesgo |
| biased_paper | 91.4% | +4.75 | 58.0% | 0.28 | Explota sesgo |
| biased_scissors | 91.8% | +4.86 | 58.2% | 0.28 | Explota sesgo |
| win_stay_lose_shift | 92.4% | +4.56 | 50.8% | 0.23 | Patrón reactivo |
| lose_stay_win_shift | 93.3% | +4.39 | 51.7% | 0.21 | Patrón reactivo |
| **cycle** | **32.2%** | **-2.47** | 43.1% | 0.34 | ❌ Punto débil |
| reverse_cycle | 88.4% | +4.04 | 41.4% | 0.46 | Ciclo invertido |
| copy_opponent | 97.4% | +5.50 | 52.8% | 0.24 | Copia IA |
| anti_last_move | 95.6% | +5.09 | 51.4% | 0.23 | Anti-última |
| avoid_repeat | 61.5% | +1.08 | 38.1% | 0.15 | Difícil |
| pressure | 85.6% | +3.25 | 52.4% | 0.25 | Presión score |
| noisy_human_like | 56.6% | +0.64 | 36.1% | 0.18 | Humano ruidoso |
| switching_strategy | 89.9% | +4.11 | 60.0% | 0.26 | Cambia estrategia |
| adaptive_counter | 88.7% | +3.87 | 50.6% | 0.33 | Contra-adaptativo |
| mixed_bias | 89.6% | +4.58 | 59.3% | 0.27 | Sesgo mixto |
| reverse_then_copy | 95.8% | +5.05 | 59.2% | 0.25 | Híbrido |

### Agregados

| Grupo | Win rate media | Comentario |
|-------|----------------|------------|
| Sesgados (rock/paper/scissors) | **89.7%** | Fortaleza principal |
| Reactivos (WSLS, LSWS, copy, anti) | **94.5%** | Muy fuerte |
| Cíclicos | cycle 32% / reverse 88% | Asimetría: pierde vs ciclo puro |
| “Humanos” (noisy, avoid, random) | **55.8%** | Comportamiento objetivo real |

### Lectura estratégica

**Fortalezas v1**
- Contra oponentes con **sesgo estable** o **patrones reactivos** claros, gana ~85–97% de partidas.
- Contra **random**, win rate ~49% y desviación uniforme moderada (0.16): no overfita a explotar cuando no hay señal.
- Predicción del movimiento humano supera 50% en bots sesgados y reactivos.

**Debilidades v1**
- **`cycle` (R→P→S fijo):** 32% win rate. El predictor confunde el ciclo determinista; la política no lo contrarresta bien.
- **`avoid_repeat` / `noisy_human_like`:** ~56–62% — oponentes más impredecibles resisten la explotación.
- Accuracy de predicción global vs random ≈ 33% (esperado: no hay patrón).

**Comparación hybrid vs actor_critic (misma batería, 1000 matches en evaluation_actor_critic.csv)**

| Bot | Hybrid win | Actor-critic win |
|-----|------------|------------------|
| random | 49.3% | 49.7% |
| cycle | 32.2% | 23.6% |
| reverse_cycle | 88.4% | **0.2%** |
| copy_opponent | 97.4% | 42.7% |

El híbrido es **más robusto** ante estrategias mixtas; el RL es un cañón de trucos contra sesgos pero frágil.

---

## Limitaciones conocidas

1. No existe agente invencible contra juego aleatorio perfecto (teoría de juegos).
2. Métricas validadas principalmente contra **bots script**, no población humana real.
3. Logs humanos en `data/human_logs/` permitirán recalibrar en v2.
4. Sin figuras PNG en `reports/figures/` en este snapshot (métricas CSV/JSON sí incluidas).

---

## Roadmap v2 (sugerido)

- [ ] Evaluar `opponent_predictor_robust.pt` o ensemble synthetic+robust
- [ ] Fine-tuning con logs humanos reales
- [ ] Detección explícita de ciclo / anti-ciclo en capa heurística
- [ ] PPO recurrente completo (GAE, clipping) como alternativa al actor-critic actual
- [ ] A/B en producción con flag de versión en API

---

## Restauración del paquete

```bat
copy releases\cachipunaa-v1.0.0\opponent_predictor.pt models\
copy releases\cachipunaa-v1.0.0\app.web.yaml configs\
```

Verificar checksum en `MANIFEST.json` antes de desplegar.

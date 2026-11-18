# Geometry Dash RL: entorno Gymnasium + oráculo + agentes PPO/DQN (nivel fijo)

## Estructura
- `gd_env/physics.py`: dinámica determinista pura (cubo, gravedad, salto, colisiones). La usan el entorno **y** el oráculo.
- `gd_env/levels.py`: nivel fijo `level_1()` (ASCII exportable con `to_ascii`), `parse_level`, `build_level`.
- `gd_env/env.py`: `GeometryDashEnv` (Gymnasium), registrado como `GeometryDash-v0`.
- `gd_env/oracle.py`: planificador exacto (programación dinámica por capas de tick) + `OraclePolicy`.
- `gd_env/render.py`: frames, GIF del episodio y mapa del recorrido (`record_episode`).
- `agents/__init__.py`: expone la API de agentes (`make_train_env`, `build_model`, `evaluate_policy_phases`, `summarize`, `EVAL_PHASES`, `TRAIN_PHASE_RANGE`).
- `agents/common.py`: `make_train_env` (con/sin fase aleatoria vía el wrapper `RandomPhase`), `build_model` (PPO/DQN con hiperparámetros iniciales), `load_model`.
- `agents/evaluate.py`: `evaluate_policy_phases` (un episodio determinista por fase), `save_rows` (CSV) y `summarize` (tabla markdown entre seeds).
- `scripts/run_oracle.py`, `scripts/record.py`, `tests/test_env.py`.
- `scripts/train.py`: entrena PPO/DQN con varias seeds, evalúa y graba GIFs. `scripts/evaluate.py`: re-evalúa modelos guardados. `scripts/record.py --model ... --algo PPO`: GIF de un modelo suelto.

## Uso
```python
from gd_env import GeometryDashEnv, OraclePolicy
env = GeometryDashEnv(obs_mode="grid")      # "grid" | "features" | "dict"
obs, info = env.reset()
pol = OraclePolicy(env)
action, _ = pol.predict(obs)                # acción experta para el estado actual
```
`python scripts/run_oracle.py --ascii` resuelve el nivel y lo reproduce. `pytest tests` corre las pruebas.

## Ver cómo el agente completa el nivel
```python
from gd_env import GeometryDashEnv, OraclePolicy, record_episode
env = GeometryDashEnv(obs_mode="features")
record_episode(env, OraclePolicy(env), "videos/oraculo.gif", "videos/oraculo_recorrido.png")
# con un agente entrenado de SB3 (cualquier objeto con .predict):
# record_episode(env, PPO.load("modelos/ppo"), "videos/ppo.gif", "videos/ppo_recorrido.png")
```
- El GIF sigue al cubo con cámara, muestra la estela, los saltos (triángulos verdes), el progreso y el motivo de muerte.
- `*_recorrido.png` muestra el nivel entero con el camino recorrido; si el agente muere, queda una cruz roja donde falló. Sirve para el informe y para comparar agentes.
- En Colab: `from IPython.display import Image; Image("videos/ppo.gif")`.
- También funciona `env.render()` con `render_mode="rgb_array"` (frame suelto) o `"ansi"` (texto).

## Agentes: entrenamiento, evaluación y GIFs
Instalación: `pip install -r requirements.txt` (instala Stable-Baselines3 y PyTorch).

```bash
# PPO y DQN, 3 seeds cada uno (observación `features` + MlpPolicy, recompensa por defecto)
python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000
# prueba rápida (~30 s) para chequear que todo corre
python scripts/train.py --algos ppo dqn --seeds 0 --timesteps 12000 --run-name humo
# entrenar con fase de arranque aleatoria en vez de nivel fijo
python scripts/train.py --algos ppo dqn --train-phase random --run-name features_random
# re-evaluar modelos guardados (más fases, regrabar GIFs)
python scripts/evaluate.py --run runs/features_fixed --n-phases 20 --gif
# re-evaluar un modelo suelto
python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo
```
Cada agente deja en `runs/<run-name>/<algo>/seed<N>/`: `model.zip`, `monitor/` (curva de aprendizaje por entorno),
`eval_fases.csv`, y `episodio_x0_<x0>.gif` + `*_recorrido.png` (fase por defecto x0=2.0 y, si falla alguna, la fase donde más progreso perdió).
`runs/<run-name>/resumen.md` trae la tabla entre seeds y `config.json` los parámetros de la corrida.

- **Evaluación**: 10 fases de arranque `x0 ∈ [2.0, 2.95]` (todas con solución comprobada por el oráculo), una política determinista por fase.
  Se reportan éxito, progreso, retorno, saltos usados vs. los del oráculo, y dónde murió.
- **Entrenamiento `--train-phase`**: `fixed` entrena siempre con x0=2.0 (mide cuánto generaliza a otras fases); `random` sortea x0 en cada episodio.
- **Hiperparámetros** (`agents/common.py`): PPO `n_steps=256` × 8 entornos, red [64, 64], `ent_coef=0.01`; DQN `lr=1e-3`, buffer 100k, ε 1→0.05 en el 30% del entrenamiento. Son un punto de partida, no están tuneados.
- Sólo `obs_mode="features"` con `MlpPolicy` por ahora; `grid`/`dict` necesitan el extractor CNN propio (ver nota para SB3 más abajo).
- La recompensa se cambia pasando `reward_fn` en `make_train_env(..., reward_fn=...)` (`agents/common.py`) y en `evaluate_policy_phases(..., reward_fn=...)`.

### Uso de los agentes desde Python (por ejemplo, en Colab)
```python
from agents import make_train_env, build_model, evaluate_policy_phases, summarize
from agents.common import load_model

env = make_train_env("features", n_envs=8, seed=0, random_phase=False)   # random_phase=True sortea x0 en cada episodio
model = build_model("ppo", env, seed=0)                                  # "ppo" | "dqn"
model.learn(total_timesteps=300_000)
model.save("modelos/ppo")

rows = evaluate_policy_phases(model, "features")                         # 10 fases x0 en [2.0, 2.95]
print(summarize({"PPO": [rows]}))                                        # una lista de filas por seed

model = load_model("ppo", "modelos/ppo.zip")                             # recargar un modelo guardado
```
Cada fila de `evaluate_policy_phases` tiene: `x0, success, progress, return, ticks, status, n_jumps, oracle_jumps, death_x`.

## Decisiones de diseño
- **Un tick = un paso**; velocidad horizontal constante (0.2 tiles/tick), así el nivel dura 520 pasos.
- **Acción**: 0 = nada, 1 = saltar (sólo surte efecto apoyado; equivale a mantener apretado).
- **Muerte**: choque de costado con un bloque, pincho (hitbox 0.3×0.3), o caer a un pozo.
- **Observación `grid`**: (3, 7, 16) = bloques / pinchos / jugador, ventana ego-céntrica. No incluye la velocidad vertical; por eso existe `dict` (grilla + vector).
- **Recompensa por defecto**: progreso (suma 1 en el nivel) + 1 al ganar − 1 al morir. Reemplazable con `reward_fn(prev, new, status, level)` para la curva de timing del profesor.
- `reset(options={"x0": ...})` cambia la fase de arranque: sirve para evaluar generalización sobre variaciones del nivel fijo. Probado con 20 fases: todas tienen solución.

## Oráculo
Como x depende sólo del tick, el espacio de estados es un DAG por capas; se recorre con DP guardando los saltos mínimos. Devuelve el plan con **menos saltos** (13 en el nivel 1, jugable en 520 ticks, retorno 2.0). `plan.jump_x` da dónde saltar (útil para el timing reward), y `Oracle.is_winnable(state)` indica si un estado todavía es salvable.

## Dificultad medida (ticks de despegue válidos por salto)
Entre 3 y 10 según el obstáculo; los dobles pinchos (x≈23 y x≈65) son los más estrechos (3 ticks). Una política aleatoria no gana nunca (tests).

## Nota para SB3
`CnnPolicy` por defecto (NatureCNN) exige imágenes de al menos 36×36; la grilla (7×16) es más chica. Opciones: extractor CNN propio con `policy_kwargs`, o `obs_mode="features"` con `MlpPolicy` como baseline.

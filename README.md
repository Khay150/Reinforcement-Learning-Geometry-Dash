# Geometry Dash RL: entorno Gymnasium + oráculo + agentes PPO/DQN (nivel fijo)

## Estructura
- `gd_env/physics.py`: dinámica determinista pura (cubo, gravedad, salto, colisiones). La usan el entorno **y** el oráculo.
- `gd_env/levels.py`: nivel fijo `level_1()` (ASCII exportable con `to_ascii`), `parse_level`, `build_level`.
- `gd_env/env.py`: `GeometryDashEnv` (Gymnasium), registrado como `GeometryDash-v0`.
- `gd_env/oracle.py`: planificador exacto (programación dinámica por capas de tick) + `OraclePolicy`.
- `gd_env/render.py`: frames, GIF del episodio y mapa del recorrido (`record_episode`).
- `agents/__init__.py`: expone la API de agentes (`make_train_env`, `build_model`, `evaluate_policy_phases`, `summarize`, `EVAL_PHASES`, `TRAIN_PHASE_RANGE`).
- `agents/common.py`: `make_train_env` (con/sin fase aleatoria vía el wrapper `RandomPhase`), `build_model` (PPO/DQN con hiperparámetros iniciales), `load_model`, `load_for_training` (seguir entrenando un modelo).
- `agents/callbacks.py`: callbacks de SB3: `PerformanceJumpPenaltyRamp` y `JumpPenaltySchedule` (rampas de la penalización por salto) y `BestModelCallback` (evalúa periódicamente y guarda el mejor modelo).
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
- Penalización por salto, guardado del mejor modelo y continuar un entrenamiento: ver las secciones de abajo.
- La recompensa se cambia pasando `reward_fn` en `make_train_env(..., reward_fn=...)` (`agents/common.py`) y en `evaluate_policy_phases(..., reward_fn=...)`.

### Penalización por salto (opcional)
Sin penalización el agente aprende a ganar saltando casi sin parar (~27 saltos contra 13 del oráculo): saltar de más
no le cuesta nada. `--jump-penalty c` resta `c` en cada salto **efectivo** (con el cubo apoyado; apretar en el aire
no cuesta). Como se maximiza el retorno acumulado, el agente sigue pagando los saltos necesarios (no saltar un pincho
cuesta `−death_penalty` y el resto del nivel) y aprende a ahorrarse los que sobran.

Hay tres modos, elegidos con `--ramp` (por defecto `none`):

| Modo | Comando | Qué hace |
|---|---|---|
| Fijo | `--jump-penalty 0.05` | `c = 0.05` durante todo el entrenamiento |
| Rampa por desempeño | `--jump-penalty 0.1 --ramp performance` | `c` arranca en 0 y sube de a `--ramp-step` (0.01) cada vez que el agente gana ≥ `--ramp-up` (80%) de los últimos `--ramp-window` (20) episodios de entrenamiento; baja si gana ≤ `--ramp-down` (40%). Nunca pasa de `--jump-penalty` |
| Rampa por tiempo | `--jump-penalty 0.1 --ramp time --ramp-time 0.3 0.6` | `c = 0` hasta el 30% del entrenamiento, sube lineal y llega a 0.1 en el 60% |

```bash
# fijo
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.05
# rampa por desempeño (recomendada si c es grande): primero aprende a ganar, después a ahorrar saltos
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.1 --ramp performance
# rampa por desempeño más exigente: sube sólo si gana el 90% de los últimos 40 episodios
python scripts/train.py --algos ppo --jump-penalty 0.1 --ramp performance --ramp-up 0.9 --ramp-window 40
# con otra penalización por morir (por defecto, la de GeometryDashEnv)
python scripts/train.py --algos ppo --jump-penalty 0.05 --death-penalty 1
```

Qué valor usar (PPO, 300k pasos, muerte = −1, 3 seeds; ver `runs/`):
- `c = 0.02`: casi no cambia nada (~25 saltos).
- `c = 0.05` fijo: gana el 93% de las fases con ~19 saltos. Buen punto de partida.
- `c = 0.1` fijo: **no gana nunca**: aprende a no saltar antes de aprender a cruzar el primer pozo. Para valores así
  conviene la rampa por desempeño.
- La rampa **por tiempo** (0.3–0.6) no superó a `c = 0.05` fijo: en 300k pasos PPO recién gana a los 150–250k, así
  que la penalización empieza a subir antes de que sepa ganar. Por eso existe la rampa por desempeño, que espera.

Las rampas guardan la evolución de `c` en `jump_penalty.csv` (paso, `c`, tasa de éxito que motivó el cambio).
Los éxitos de la rampa por desempeño son de episodios de **entrenamiento** (acciones muestreadas, con exploración), que
salen más bajos que en la evaluación determinista: por eso el umbral por defecto es 80% y no 100%.

### Qué se guarda y cuál es el "mejor modelo"
```
runs/<run-name>/
├── config.json              parámetros de la corrida (se escribe al empezar)
├── resumen.md               tablas entre seeds: modelo final y mejor modelo
└── <algo>/seed<N>/
    ├── model.zip            modelo al terminar el entrenamiento
    ├── best_model.zip       mejor modelo de las evaluaciones periódicas
    ├── best_model.json      en qué paso se obtuvo y con qué métricas
    ├── eval_history.csv     todas las evaluaciones periódicas (curva de aprendizaje determinista)
    ├── eval_fases.csv       evaluación del modelo final   ·  eval_fases_best.csv  del mejor modelo
    ├── checkpoints/         model_<pasos>_steps.zip cada --checkpoint-freq pasos (50k)
    ├── monitor/             un CSV por entorno con cada episodio de entrenamiento
    ├── jump_penalty.csv     evolución de c (sólo con --ramp)
    └── episodio_*.gif, *_recorrido.png  (final)  ·  mejor_episodio_*.gif, mejor_*_recorrido.png  (mejor)
```
- **Mejor modelo**: cada `--eval-freq` pasos (25k; `0` lo desactiva) y al terminar se evalúa el modelo de forma
  determinista en las 10 fases. Gana el de **más fases ganadas**; a igualdad, más progreso; a igualdad, **menos saltos**.
  Hace falta porque PPO es muy inestable acá: entre el checkpoint de 300.000 pasos y el modelo final (301.056, una sola
  actualización) un agente pasó de ganar 0/10 fases a 8/10 con 13 saltos. Reportar sólo el último modelo es una lotería.
- **Nunca se pisa una corrida**: si `runs/<run-name>/` ya existe, la nueva se guarda en `runs/<run-name>_<fecha-hora>/`.
- Re-evaluar los mejores modelos de una corrida: `python scripts/evaluate.py --run runs/<run-name> --best [--gif]`
  (escribe `eval_fases_best.csv` y `mejor_*.gif`, sin pisar los del modelo final).

### Continuar entrenando un modelo ya entrenado
`--init-model` carga un `.zip` (un `model.zip`, un `best_model.zip` o un checkpoint) y sigue entrenándolo en vez de
empezar de cero:
```bash
# 200k pasos MÁS sobre el mejor modelo de una corrida anterior
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip
# típico: primero aprender a ganar sin penalización y después afinar con penalización por salto
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip --jump-penalty 0.05
```
- `--algos` tiene que tener **un solo** algoritmo, el mismo con el que se entrenó el modelo (`ppo` o `dqn`).
- `--timesteps` son pasos **adicionales**. Los checkpoints y `eval_history.csv` cuentan desde 0 en la corrida nueva.
- Se conservan los pesos **y los hiperparámetros** del modelo cargado (learning rate, red, etc.), no los de `build_model`.
- La recompensa (`--jump-penalty`, `--ramp`, `--death-penalty`) y `--train-phase` son los de la corrida nueva:
  se puede cambiar la recompensa respecto de la original, que es la idea del ejemplo de arriba.
- Con varias `--seeds`, cada seed parte del mismo modelo y sigue con otra secuencia aleatoria.
- **DQN**: el replay buffer no se guarda en el `.zip`, así que arranca vacío (vuelven a pasar los primeros 5000 pasos
  sin entrenar para llenarlo). La exploración ε arranca en su valor final (0.05) en vez de 1.0, para no tirar lo aprendido
  jugando al azar.
- La corrida nueva se guarda en `runs/<run-name>_continuado/` (o con `--run-name`), y `config.json` registra `init_model`.
- PPO puede continuar con otra cantidad de entornos (`--n-envs`) que la original.

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

# penalización por salto con rampa por desempeño + guardado del mejor modelo
from agents.callbacks import BestModelCallback, PerformanceJumpPenaltyRamp
env = make_train_env("features", n_envs=8, seed=0, random_phase=False, jump_penalty=0.0, death_penalty=1)
model = build_model("ppo", env, seed=0)
model.learn(300_000, callback=[PerformanceJumpPenaltyRamp(c_max=0.1, log_path="modelos/jump_penalty.csv"),
                               BestModelCallback(eval_freq=25_000, save_dir="modelos", jump_penalty=0.1)])

# seguir entrenando un modelo guardado
from agents.common import load_for_training
model = load_for_training("ppo", "modelos/best_model.zip", env, seed=0)
model.learn(200_000)
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

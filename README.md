# Geometry Dash RL: entorno Gymnasium + oráculo + agentes PPO/A2C/DQN/QR-DQN (nivel fijo)

## Estructura
- `gd_env/physics.py`: dinámica determinista pura (cubo, gravedad, salto, colisiones). La usan el entorno **y** el oráculo.
- `gd_env/levels.py`: nivel fijo `level_1()` (ASCII exportable con `to_ascii`), `parse_level`, `build_level`.
- `gd_env/env.py`: `GeometryDashEnv` (Gymnasium), registrado como `GeometryDash-v0`.
- `gd_env/oracle.py`: planificador exacto (programación dinámica por capas de tick) + `OraclePolicy`.
- `gd_env/render.py`: frames, GIF del episodio y mapa del recorrido (`record_episode`).
- `agents/__init__.py`: expone la API de agentes (`make_train_env`, `build_model`, `evaluate_policy_phases`, `summarize`, `EVAL_PHASES`, `TRAIN_PHASE_RANGE`).
- `agents/common.py`: `make_train_env` (con/sin fase aleatoria vía el wrapper `RandomPhase`), `build_model` (PPO/A2C/DQN/QR-DQN con hiperparámetros iniciales), `load_model` y `load_for_training` (seguir entrenando un modelo).
- `agents/callbacks.py`: callbacks de SB3: `PerformanceJumpPenaltyRamp` (rampa de la penalización por salto) y `BestModelCallback` (evalúa periódicamente y guarda el mejor modelo).
- `agents/evaluate.py`: `evaluate_policy_phases` (un episodio determinista por fase), `metrics`/`describe`, `save_rows`/`append_row` (CSV) y `summarize` (tabla markdown entre seeds).
- `agents/runs.py`: `evaluate_and_record` (evalúa un modelo y deja su CSV y GIFs), compartido por `train.py` y `evaluate.py`.
- `scripts/run_oracle.py`, `scripts/record.py`, `tests/test_env.py`.
- `scripts/train.py`: entrena PPO/A2C/DQN/QR-DQN con varias seeds, evalúa y graba GIFs. `scripts/evaluate.py`: re-evalúa modelos guardados. `scripts/record.py --model ... --algo PPO`: GIF de un modelo suelto.

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
# E1: comparación de algoritmos (PPO, A2C, DQN, QR-DQN), 3 seeds cada uno
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos
# prueba rápida (~30 s) para chequear que todo corre
python scripts/train.py --algos ppo dqn --seeds 0 --timesteps 12000 --run-name humo
# penalización por salto con rampa (ver abajo) y fine-tuning de un modelo ya entrenado
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.1 --ramp
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --init-model runs/features_fixed/ppo/seed0/best_model.zip --jump-penalty 0.05
# entrenar con fase de arranque aleatoria en vez de nivel fijo
python scripts/train.py --algos ppo dqn --train-phase random --run-name features_random
# re-evaluar modelos guardados (más fases, regrabar GIFs)
python scripts/evaluate.py --run runs/features_fixed --n-phases 20 --gif
# re-evaluar un modelo suelto
python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo
```
Cada agente deja en `runs/<run-name>/<algo>/seed<N>/` el modelo final, el mejor modelo, checkpoints, evaluaciones y GIFs
(detalle en "Qué se guarda" más abajo). `runs/<run-name>/resumen.md` trae la tabla entre seeds y `config.json` los parámetros de la corrida.

- **Evaluación**: 10 fases de arranque `x0 ∈ [2.0, 2.95]` (todas con solución comprobada por el oráculo), una política determinista por fase.
  Se reportan éxito, progreso, retorno, saltos usados vs. los del oráculo, y dónde murió.
- **Entrenamiento `--train-phase`**: `fixed` entrena siempre con x0=2.0 (mide cuánto generaliza a otras fases); `random` sortea x0 en cada episodio.
- **Algoritmos** (`--algos`): `ppo`, `a2c`, `dqn`, `qrdqn`. PPO/A2C (on-policy) usan `--n-envs` entornos en paralelo; DQN/QR-DQN (off-policy, replay buffer) usan 1. QR-DQN viene de `sb3-contrib` (ya está en `requirements.txt`).
- **Hiperparámetros** (`agents/common.py`): PPO `n_steps=256` × 8 entornos, red [64, 64], `ent_coef=0.01`; A2C `lr=7e-4`, `n_steps=16` × 8 entornos, `ent_coef=0.01`; DQN `lr=1e-3`, buffer 100k, ε 1→0.05 en el 30% del entrenamiento; QR-DQN igual que DQN con 50 cuantiles. Son un punto de partida, no están tuneados.
- Sólo `obs_mode="features"` con `MlpPolicy` por ahora; `grid`/`dict` necesitan el extractor CNN propio (ver nota para SB3 más abajo).
- Penalización por salto, qué se guarda y continuar un entrenamiento: ver las secciones de abajo.
- La recompensa se cambia pasando `reward_fn` en `make_train_env(..., reward_fn=...)` (`agents/common.py`) y en `evaluate_policy_phases(..., reward_fn=...)`.

### Penalización por salto (opcional)
Sin penalización el agente aprende a ganar saltando casi sin parar (~27 saltos contra 13 del oráculo): saltar de más
no le cuesta nada. `--jump-penalty c` resta `c` en cada salto **efectivo** (con el cubo apoyado; apretar en el aire no cuesta).
`--death-penalty` cambia lo que cuesta morir (por defecto 10, ver `GeometryDashEnv`): tiene que ser alto frente a `c` para que
el agente prefiera pagar un salto necesario antes que morir.

| Modo | Comando | Qué hace |
|---|---|---|
| Fijo | `--jump-penalty 0.05` | `c = 0.05` durante todo el entrenamiento (el run queda con sufijo `_jp0.05`) |
| Rampa por desempeño | `--jump-penalty 0.1 --ramp` | `c` arranca en 0 y sube de a `--ramp-step` (0.01) cada vez que el agente gana ≥ `--ramp-up` (80%) de los últimos `--ramp-window` (20) episodios de entrenamiento; baja si gana ≤ `--ramp-down` (40%). Nunca pasa de `--jump-penalty`. El run queda con sufijo `_ramp` |

La rampa espera a que el agente sepa ganar antes de apretar: con `c` alto y fijo (0.1, medido con muerte = −1) el agente aprende a
no saltar antes de aprender a cruzar el primer pozo y no gana nunca. Con `c = 0.05` fijo, en cambio, ganó ~93% de las fases con ~19 saltos.
La evolución de `c` queda en `jump_penalty.csv` (paso, `c`, tasa de éxito que motivó el cambio). Los éxitos que mira la rampa son de
episodios de **entrenamiento** (acciones muestreadas, con exploración), más bajos que en la evaluación determinista: por eso el umbral
por defecto es 80% y no 100%.

### Qué se guarda
```
runs/<run-name>/
├── config.json              parámetros de la corrida (se escribe al empezar)
├── resumen.md               tablas entre seeds: modelo final y mejor modelo
└── <algo>/seed<N>/
    ├── model.zip            modelo al terminar el entrenamiento
    ├── best_model.zip       mejor modelo de las evaluaciones periódicas  ·  best_model.json: en qué paso y con qué métricas
    ├── checkpoints/         model_<pasos>_steps.zip cada --checkpoint-freq pasos (25k)
    ├── eval_history.csv     una fila por evaluación periódica (mismos pasos que los checkpoints); is_best=1 marca cuándo mejoró el récord
    ├── eval_fases.csv       evaluación del modelo final  ·  eval_fases_best.csv  del mejor modelo
    ├── monitor/             un CSV por entorno con cada episodio de entrenamiento
    ├── jump_penalty.csv     evolución de c (sólo con --ramp)
    └── episodio_*.gif, *_recorrido.png  (final)  ·  mejor_episodio_*.gif, mejor_*_recorrido.png  (mejor)
```
- **Mejor modelo**: cada `--eval-freq` pasos (25k; `0` lo desactiva) y al terminar se evalúa el modelo de forma determinista en las
  10 fases. Gana el de **más fases ganadas**; a igualdad, más progreso; a igualdad, **menos saltos**. Hace falta porque PPO es muy
  inestable acá: reportar sólo el último modelo es una lotería.
- **Checkpoints**: como `--checkpoint-freq` y `--eval-freq` coinciden, cada fila de `eval_history.csv` tiene su `checkpoints/model_<paso>_steps.zip`
  (el mejor es el de `best_model.json["timesteps"]`, que además se copia como `best_model.zip`). Así se conservan todos los modelos y queda claro cuál fue el mejor.
- **Nunca se pisa una corrida**: si `runs/<run-name>/` ya existe, la nueva se guarda en `runs/<run-name>_<fecha-hora>/`.
- Re-evaluar los mejores modelos: `python scripts/evaluate.py --run runs/<run-name> --best [--gif]`.

### Continuar entrenando un modelo ya entrenado (fine-tuning)
`--init-model` carga un `.zip` (`model.zip`, `best_model.zip` o un checkpoint) y sigue entrenándolo en vez de empezar de cero:
```bash
# 200k pasos MÁS sobre el mejor modelo de una corrida anterior, ahora con penalización por salto
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip --jump-penalty 0.05
```
- `--algos` tiene que tener **un solo** algoritmo, el mismo con el que se entrenó el modelo.
- `--timesteps` son pasos **adicionales**; checkpoints y `eval_history.csv` cuentan desde 0 en la corrida nueva.
- Se conservan los pesos **y los hiperparámetros** del modelo cargado, no los de `build_model`. La recompensa (`--jump-penalty`, `--ramp`,
  `--death-penalty`) y `--train-phase` son los de la corrida nueva: se puede cambiar la recompensa respecto de la original.
- **DQN / QR-DQN**: el replay buffer no se guarda en el `.zip`, así que arranca vacío (vuelven a pasar los primeros 5000 pasos sin entrenar).
  La exploración ε arranca en su valor final (0.05) en vez de 1.0, para no tirar lo aprendido jugando al azar.
- Con `--ramp`, `c` vuelve a arrancar en 0 (la rampa es parte de la corrida nueva).
- La corrida nueva se guarda en `runs/<run-name>_continuado/` (o con `--run-name`) y `config.json` registra `init_model`.
### Uso de los agentes desde Python (por ejemplo, en Colab)
```python
from agents import make_train_env, build_model, evaluate_policy_phases, summarize
from agents.common import load_model

env = make_train_env("features", n_envs=8, seed=0, random_phase=False)   # random_phase=True sortea x0 en cada episodio
model = build_model("ppo", env, seed=0)                                  # "ppo" | "a2c" | "dqn" | "qrdqn"
model.learn(total_timesteps=300_000)
model.save("modelos/ppo")

rows = evaluate_policy_phases(model, "features")                         # 10 fases x0 en [2.0, 2.95]
print(summarize({"PPO": [rows]}))                                        # una lista de filas por seed

model = load_model("ppo", "modelos/ppo.zip")                             # recargar un modelo guardado

# penalización por salto con rampa por desempeño + guardado del mejor modelo
from agents.callbacks import BestModelCallback, PerformanceJumpPenaltyRamp
env = make_train_env("features", n_envs=8, seed=0, random_phase=False, jump_penalty=0.1)
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
- **Recompensa por defecto**: progreso (suma 1 en el nivel) + 1 al ganar − 10 al morir (`death_penalty`). Reemplazable con `reward_fn(prev, new, status, level)` para la curva de timing del profesor.
- `reset(options={"x0": ...})` cambia la fase de arranque: sirve para evaluar generalización sobre variaciones del nivel fijo. Probado con 20 fases: todas tienen solución.

## Oráculo
Como x depende sólo del tick, el espacio de estados es un DAG por capas; se recorre con DP guardando los saltos mínimos. Devuelve el plan con **menos saltos** (13 en el nivel 1, jugable en 520 ticks, retorno 2.0). `plan.jump_x` da dónde saltar (útil para el timing reward), y `Oracle.is_winnable(state)` indica si un estado todavía es salvable.

## Dificultad medida (ticks de despegue válidos por salto)
Entre 3 y 10 según el obstáculo; los dobles pinchos (x≈23 y x≈65) son los más estrechos (3 ticks). Una política aleatoria no gana nunca (tests).

## Nota para SB3
`CnnPolicy` por defecto (NatureCNN) exige imágenes de al menos 36×36; la grilla (7×16) es más chica. Opciones: extractor CNN propio con `policy_kwargs`, o `obs_mode="features"` con `MlpPolicy` como baseline.

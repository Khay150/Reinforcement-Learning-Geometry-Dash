# Geometry Dash RL: entorno Gymnasium + oráculo + agentes PPO/A2C/DQN/QR-DQN (uno o varios niveles fijos)

## Estructura
- `gd_env/physics.py`: dinámica determinista pura (cubo, gravedad, salto, colisiones). La usan el entorno **y** el oráculo.
- `gd_env/levels.py`: niveles (`level_1()` y los demás de "Niveles registrados", exportables a ASCII con `to_ascii`; en el ASCII `#` es bloque, `^` pincho, `v` pincho colgante), `parse_level`, `build_level`, el registro `LEVELS` (nombre → nivel) y `resolve_levels` (traduce `--levels` en niveles de entrenamiento con peso y de validación).
- `gd_env/env.py`: `GeometryDashEnv` (Gymnasium), registrado como `GeometryDash-v0`.
- `gd_env/rewards.py`: la recompensa: `Transition` (qué pasó en un tick), `default_reward`, los **términos** que se prenden y apagan con `--rewards` (`REWARD_TERMS`, `reward_terms`, `salto_preciso_term`, `validate_terms`) y la carga de recompensas propias (`load_reward_fn`). Ver "Recompensa y `train()`".
- `gd_env/oracle.py`: planificador exacto (programación dinámica por capas de tick) + `OraclePolicy`.
- `gd_env/render.py`: frames, GIF del episodio y mapa del recorrido (`record_episode`).
- `gd_env/difficulty.py`: dificultad de un nivel medida con la física, sin entrenar (`level_difficulty`): precisión de cada salto, trampas, saltos encadenados y probabilidad de ganar al azar. Ver "Dificultad de un nivel".
- `agents/__init__.py`: expone la API de agentes (`make_train_env`, `build_model`, `evaluate_policy_phases`, `summarize`, `EVAL_PHASES`, `TRAIN_PHASE_RANGE`).
- `agents/common.py`: `make_train_env` (con/sin fase aleatoria vía el wrapper `RandomPhase`), `build_model` (PPO/A2C/DQN/QR-DQN con hiperparámetros iniciales; elige MLP o CNN según `obs_mode`), `policy_for` / `policy_kwargs_for`, `load_model` y `load_for_training` (seguir entrenando un modelo).
- `agents/cnn.py`: extractores CNN propios para SB3: `GridCNN` (observación `grid`) y `GridVecCNN` (observación `dict`: CNN para la grilla + capa lineal para el vector). Ver "Observaciones y CNN".
- `agents/callbacks.py`: callbacks de SB3: `PerformanceJumpPenaltyRamp` (rampa de la penalización por salto), `LevelProgression` (introduce los niveles por etapas y reparte los episodios según el desempeño; ver "Progresión de niveles") y `BestModelCallback` (evalúa periódicamente y guarda el mejor modelo).
- `agents/evaluate.py`: `evaluate_policy_phases` (un episodio determinista por fase), `metrics`/`describe`, `save_rows`/`append_row` (CSV) y `summarize` (tabla markdown entre seeds).
- `ejemplos/`: recompensas propias de ejemplo (`reward_ejemplo.py`, `reward_santi.py`). `niveles/`: conjuntos de niveles con pesos y validación para `--levels` (`ejemplo.json`, `santi.json`, `nacho.json`, y `nacho_progresion.json` con etapas).
- `agents/runs.py`: `evaluate_and_record` (evalúa un modelo y deja su CSV y GIFs), compartido por `train.py` y `evaluate.py`.
- `scripts/run_oracle.py`, `scripts/record.py`, `scripts/difficulty.py` (tabla de dificultad de los niveles), `tests/test_env.py` (entorno, oráculo, términos de recompensa, callbacks), `tests/test_cnn.py` (extractores CNN con los 4 algoritmos), `tests/test_difficulty.py` y `tests/test_spike_down.py` (pincho colgante `v`).
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
# lo mismo con CNN: observación `grid` (imagen 3x7x16) o `dict` (grilla + vector); el run queda como grid_fixed / dict_fixed
python scripts/train.py --algos ppo dqn --obs grid --seeds 0 1 2 --timesteps 300000
python scripts/train.py --algos ppo dqn --obs dict --seeds 0 1 2 --timesteps 300000
# E1: comparación de algoritmos (PPO, A2C, DQN, QR-DQN), 3 seeds cada uno
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos
# prueba rápida (~30 s) para chequear que todo corre
python scripts/train.py --algos ppo dqn --seeds 0 --timesteps 12000 --run-name humo
# penalización por salto con rampa (ver abajo) y fine-tuning de un modelo ya entrenado
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.1 --ramp
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --init-model runs/features_fixed/ppo/seed0/best_model.zip --jump-penalty 0.05
# entrenar con fase de arranque aleatoria en vez de nivel fijo
python scripts/train.py --algos ppo dqn --train-phase random --run-name features_random
# varios niveles (con pesos) y niveles de validación held out, definidos en un .json (ver "Varios niveles")
python scripts/train.py --algos ppo --levels niveles/ejemplo.json
# los 5 niveles de Santi con su recompensa (términos a elección, ver "Recompensa y `train()`")
python scripts/train.py --algos ppo --levels niveles/santi.json --rewards progress win death salto_preciso
# niveles introducidos por etapas según el desempeño (ver "Progresión de niveles")
python scripts/train.py --algos ppo --obs grid --levels niveles/nacho_progresion.json
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
- **Observación** (`--obs`): `features` (vector + `MlpPolicy`, el baseline), `grid` (imagen 3×7×16 + CNN propia) o `dict` (grilla + vector). Detalle de la red en "Observaciones y CNN".
- Penalización por salto, qué se guarda y continuar un entrenamiento: ver las secciones de abajo.
- La recompensa se configura por parámetros o con una función propia: ver "Recompensa y `train()`" justo abajo.
- **Niveles** (`--levels`): por defecto sólo `level_1`. Con varios, en cada episodio se sortea uno según su peso; ver "Varios niveles".

### Recompensa y `train()`
Todo lo de la recompensa se pasa como parámetro, por línea de comandos o desde Python con `agents.train`:

| Parámetro (Python) | Flag | Por defecto | Qué hace |
|---|---|---|---|
| `win_bonus` | `--win-bonus` | 1 | premio al llegar a la meta |
| `death_penalty` | `--death-penalty` | 10 | castigo al morir |
| `jump_penalty` | `--jump-penalty` | 0 | castigo por cada salto efectivo (con `ramp`, el máximo) |
| `progress_scale` | `--progress-scale` | 1 | lo que vale cruzar todo el nivel |
| `survive_bonus` | `--survive-bonus` | 0 | bonus por supervivencia |
| `survive_interval` | `--survive-interval` | 1 | el bonus se da cada N ticks vivo (1 = en todos los ticks) |
| `survive_rate` | `--survive-rate` | 0 | alternativa a `survive_bonus`: bonus por tick = `survive_rate × jump_penalty` (necesita `jump_penalty > 0`) |
| `rewards` | `--rewards TERMINO ...` | `progress win death survive jump_penalty` | qué términos de la recompensa quedan prendidos (ver abajo) |
| `reward_fn` | `--reward-fn archivo.py:funcion` | — | reemplaza la recompensa de por defecto (ver abajo) |

```python
from agents import train, summarize
out = train("ppo", seeds=[0, 1], timesteps=200_000, win_bonus=2.0, death_penalty=5.0, jump_penalty=0.05)
print(summarize(out["final"]))          # out["root"] = carpeta de la corrida
```
```bash
python scripts/train.py --algos ppo --win-bonus 2 --death-penalty 5 --jump-penalty 0.05
```
`train` acepta los mismos argumentos que `scripts/train.py` (`obs`, `train_phase`, `n_envs`, `ramp`, `init_model`, ...) y guarda
todo en `config.json`, de modo que `scripts/evaluate.py` re-evalúa con la misma recompensa.

**Términos que se prenden y apagan (`--rewards`).** La recompensa está hecha de piezas y en cada entrenamiento se elige cuáles
valen. Sin `--rewards` queda la de siempre (todo menos `salto_preciso`), así las corridas viejas se reproducen igual.

| Término | Qué hace | Parámetro |
|---|---|---|
| `progress` | avanzar: cruzar todo el nivel vale `progress_scale` | `--progress-scale` |
| `win` | premio al llegar a la meta | `--win-bonus` |
| `death` | castigo al morir | `--death-penalty` |
| `survive` | bonus por supervivencia | `--survive-bonus`, `--survive-interval`, `--survive-rate` |
| `jump_penalty` | castigo fijo por cada salto efectivo (con `--ramp`, sube según el desempeño) | `--jump-penalty` |
| `salto_preciso` | (Santi) saltar sobre pinchos o pozos sale gratis y suma `+0.03` por cada peligro desde 2 (`^^`, pozo ancho); saltar sin ningún obstáculo en los 3.2 bloques del salto resta `0.06` | — |

```bash
# sólo progreso, meta y muerte + la recompensa de Santi, en los 5 niveles de Santi
python scripts/train.py --algos ppo --levels niveles/santi.json --rewards progress win death salto_preciso
# la de siempre, pero sin castigo por morir
python scripts/train.py --algos ppo --rewards progress win survive jump_penalty --jump-penalty 0.05
```
```python
train("ppo", levels="niveles/santi.json", rewards=["progress", "win", "death", "salto_preciso"])
```
Reglas (se chequean antes de entrenar, con un error claro):
- `jump_penalty` y `salto_preciso` son **excluyentes**: los dos fijan cuánto cuesta saltar. Tampoco se acepta `salto_preciso` con `--jump-penalty` o `--ramp`.
- Un parámetro de un término apagado (p. ej. `--death-penalty 5` sin `death` en `--rewards`) es un error, no se ignora en silencio.
- `--rewards` no se combina con `--reward-fn` (esa ya reemplaza toda la recompensa).
- Los términos quedan en `config.json` (así `evaluate.py` re-evalúa igual) y en el nombre de la carpeta (`..._rw-progress+win+death+salto_preciso`).
- `ejemplos/reward_santi.py:salto_preciso` sigue existiendo como `--reward-fn` suelto (la versión original, con `win_bonus=2`).

**Recompensa propia.** Se escribe una función que recibe un `Transition` (qué pasó en el tick) y devuelve un float; se puede
partir de `default_reward` y modificarla. Cualquier archivo `.py` sirve (ejemplo en `ejemplos/reward_ejemplo.py`):
```python
from gd_env import Transition, default_reward

def mi_recompensa(tr: Transition) -> float:
    r = default_reward(tr, win_bonus=2.0, death_penalty=5.0)    # opcional: partir de la de por defecto
    return r - 0.05 if tr.jumped else r
```
```bash
python scripts/train.py --algos ppo --reward-fn mi_archivo.py:mi_recompensa
```
```python
train("ppo", reward_fn=mi_recompensa)             # o directo: GeometryDashEnv(reward_fn=mi_recompensa)
```
`tr` tiene `prev`/`new` (estados), `status`, `dead`, `won`, `jumped`, `dx`, `d_progress`, `progress` y `level` (detalle en
`gd_env/rewards.py`). Si hay `reward_fn`, `win_bonus`/`death_penalty`/`progress_scale`/`survive_bonus`/`survive_rate` no se usan (los
reemplaza la función); `jump_penalty` se sigue sumando. La firma vieja `reward_fn(prev, new, status, level)` también funciona.
En config.json la función queda como `modulo:nombre`: con funciones definidas en un notebook o lambdas, `evaluate.py` no puede
recargarla y re-evalúa con la recompensa por defecto (sólo cambia la columna `return`).


### Penalización por salto (opcional)
Sin penalización el agente aprende a ganar saltando casi sin parar (~27 saltos contra 13 del oráculo): saltar de más
no le cuesta nada. `--jump-penalty c` resta `c` en cada salto **efectivo** (con el cubo apoyado; apretar en el aire no cuesta).
`--death-penalty` cambia lo que cuesta morir (por defecto 10, ver `GeometryDashEnv`): tiene que ser alto frente a `c` para que
el agente prefiera pagar un salto necesario antes que morir.
`jump_penalty` es uno de los términos de `--rewards` y es **excluyente** con `salto_preciso` (que ya decide cuánto cuesta saltar según el obstáculo):
con `salto_preciso` no se pueden usar `--jump-penalty` ni `--ramp`.

| Modo | Comando | Qué hace |
|---|---|---|
| Fijo | `--jump-penalty 0.05` | `c = 0.05` durante todo el entrenamiento (el run queda con sufijo `_jp0.05`) |
| Rampa por desempeño | `--jump-penalty 0.1 --ramp` | `c` arranca en 0 y sube de a `--ramp-step` (0.01) cada vez que el agente gana ≥ `--ramp-up` (80%) de los últimos `--ramp-window` (20) episodios de entrenamiento; baja si gana ≤ `--ramp-down` (40%). Nunca pasa de `--jump-penalty`. El run queda con sufijo `_ramp` |

La rampa espera a que el agente sepa ganar antes de apretar: con `c` alto y fijo (0.1, medido con muerte = −1) el agente aprende a
no saltar antes de aprender a cruzar el primer pozo y no gana nunca. Con `c = 0.05` fijo, en cambio, ganó ~93% de las fases con ~19 saltos.
La evolución de `c` queda en `jump_penalty.csv` (paso, `c`, tasa de éxito que motivó el cambio). Los éxitos que mira la rampa son de
episodios de **entrenamiento** (acciones muestreadas, con exploración), más bajos que en la evaluación determinista: por eso el umbral
por defecto es 80% y no 100%.

### Bonus por seguir vivo (opcional)
El cubo avanza solo a velocidad constante, así que no puede quedarse quieto: el bonus no premia "esperar", premia **cada
tramo del nivel que sobrevive**. Sirve para que no haya que retocar `death_penalty`/`win_bonus` al cambiar el largo del nivel:
el progreso suma 1 en todo el nivel (por tick vale menos cuanto más largo es), pero el costo de saltar crece con el largo; en un
nivel largo lo que queda por pagar en saltos puede superar lo que queda por ganar y al agente le conviene morir. Un bonus por
tick crece con el largo igual que los saltos y lo compensa.

| Parámetro | Qué hace |
|---|---|
| `--survive-bonus b --survive-interval N` | suma `b` cada `N` ticks vivo (no en el tick en que muere). Tasa por tick = `b / N` |
| `--survive-rate r --survive-interval N` | igual, pero con tasa por tick = `r × c` (`c` = `jump_penalty` vigente, también con `--ramp`). Necesita `--jump-penalty > 0`. Es alternativa a `--survive-bonus` (pasar las dos da error) |

Regla práctica: la tasa por tick tiene que superar `c × saltos_por_tick` del nivel (así cada tick vivo deja ganancia neta y morir
nunca conviene). En los niveles registrados el máximo es ~0.056 saltos por tick (`muy_dificil`: 15 saltos en 270 ticks; `level_1`: 13 en 520),
así que `--survive-rate 0.1` deja margen y no hay que recalcularlo por nivel. Ejemplo: `--jump-penalty 0.05 --survive-rate 0.1 --survive-interval 20`
da 0.1 × 0.05 × 20 = 0.1 cada 20 ticks. Si armás niveles con más de ~0.06 saltos por tick, subí `r`. Los valores se guardan en `config.json`
y el run queda con sufijo `_sr0.1_si20` (o `_sb…_si…`).

### Niveles registrados
Todos viven en `gd_env/levels.py` (registro `LEVELS`), miden 7 de alto y 60 de largo (salvo `level_1`, de 110) y arrancan en `x = 2.0`.
Los números salen de `python scripts/difficulty.py` (peor caso entre las 10 fases de arranque; ver "Dificultad de un nivel").

| Nivel | Autor | Qué tiene | Saltos mín. | Ventana mín. (ticks) | log10 P(ganar al azar) |
|---|---|---|---|---|---|
| `level_1` | TP | pinchos simples y dobles, escalón, pozos, escalera y combos | 13 | 2 | −4.6 |
| `nacho_facil` | Nacho | escalones de a uno y bajadas, pinchos sueltos | 6 | 7 | −3.7 |
| `nacho_medio` | Nacho | pozos anchos de 3 tiles e islas donde aterrizar | 8 | 5 | −7.1 |
| `nacho_dificil` | Nacho | mesetas de altura 3 con pinchos arriba y huecos entre paredes | 12 | 3 | −8.8 |
| `nacho_muy_dificil` | Nacho | ritmo: pinchos simples y dobles, pozos y una plataforma con pinchos | 14 | 1 | −12.8 |
| `facil` | Rugo | | 8 | 7 | −6.3 |
| `medio` | Rugo | | 9 | 3 | −6.8 |
| `dificil` | Rugo | | 14 | 2 | −8.2 |
| `muy_dificil` | Rugo | una fila de bloques arriba hace de techo (ver la nota de `features` en "Observaciones y CNN") | 15 | 1 | −10.3 |
| `muy_muy_dificil` | Rugo | "pixel perfect": ventanas de 1 tick | 12 | 1 | −33.1 |
| `franco_facil` | Franco | pinchos sueltos y separados, un escalón de 1 y un pozo de 2 | 5 | 8 | −2.0 |
| `franco_medio` | Franco | doble pincho, plataforma con pincho arriba, escalera, pozo de 3 y pincho al aterrizar | 8 | 3 | −5.1 |
| `franco_dificil` | Franco | dos dobles pinchos, plataforma entre pozos, escalera con pincho en la cima | 11 | 3 | −10.7 |
| `franco_techo` | Franco | dos escalones, una plataforma con **pinchos colgantes** (bajo ella saltar mata) y una torre con pincho | 5 | 3 | −17.3 |
| `franco_rutas` | Franco | **dos rutas**: escalera a una meseta alta con hueco de 3; se puede saltar y seguir arriba (7 saltos) o caer y pasar por el túnel de 1 tile de abajo (5 saltos) | 5 | 7 | −3.1 |
| `franco_tunel` | Franco | pinchos sueltos, plataforma con pinchos colgantes (hueco de 1 tile: saltar mata), escalón y pozo | 6 | 8 | −11.6 |
| `franco_pasillos` | Franco | escalera con pincho en la meseta, plataforma alta con pinchos colgantes (hueco de 2) y túnel bajo (hueco de 1), doble pincho y pozo | 8 | 3 | −15.7 |
| `franco_sin_saltar` | Franco | techo de pinchos colgantes en todo el nivel y piso liso (hueco de 1): se gana **sin saltar nunca**; cualquier salto mata | 0 | — | −80.7 |
| `santi_facil` | Santi | pinchos simples separados, dos plataformas bajas y un pozo de 2 | 6 | 8 | −3.4 |
| `santi_intermedio_1` | Santi | pinchos simples y dobles (`^^`), pozos de 2 y 3 | 7 | 1 | −6.7 |
| `santi_intermedio_2` | Santi | escalón a altura 2 y hasta 3 pinchos seguidos bajando de escalón (`##^^^`) | 9 | 1 | −10.9 |
| `santi_intermedio_dificil` | Santi | dos rachas de 3 pinchos tras una plataforma y pozos de 3 | 10 | 3 | −11.9 |
| `santi_dificil` | Santi | meseta de altura 3 con pincho arriba, caída sobre 3 pinchos y combos | 13 | 1 | −14.0 |

Para ver cualquiera en ASCII: `python -c "from gd_env import get_level, to_ascii; print(to_ascii(get_level('franco_techo')))"`.

**Pincho colgante (`v`).** Es un pincho que cuelga de un bloque y apunta hacia abajo; sirve para armar techos peligrosos como el de
`franco_techo`, `franco_tunel`, `franco_pasillos` y `franco_sin_saltar` (techo de pinchos en todo el nivel: el óptimo es 0 saltos, para probar si el agente aprende a no saltar): con el hueco de 1 tile que deja bajo la plataforma, el cubo pasa caminando pero muere si salta (la cabeza toca la hitbox).
Su hitbox es la del pincho común espejada (0.3×0.3, en la parte alta del tile) y, como en el resto de los pinchos, rozar la punta no mata.
En la grilla (`grid`/`dict`) los pinchos hacia arriba y los colgantes comparten el canal de pinchos, así la forma de la observación
sigue siendo (3, 7, 16) y los modelos ya entrenados se cargan igual; la posición dentro de la grilla los distingue. Los niveles que
no lo usan se comportan exactamente igual que antes. Se prueba en `tests/test_spike_down.py`.

### Varios niveles
`--levels` acepta nombres de niveles registrados en `gd_env.LEVELS` (`gd_env/levels.py`) y/o un archivo `.json`:
```bash
python scripts/train.py --algos ppo --levels niveles/ejemplo.json      # conjunto definido en un archivo
python scripts/train.py --algos ppo --levels franco_facil franco_medio  # nombres sueltos: peso 1 c/u, validación null
```
```json
{"niveles": {"franco_facil": 1.0, "franco_medio": 1.0, "franco_dificil": 2.0},
 "validacion": ["franco_techo"]}
```
- **`niveles`**: nivel → peso (probabilidad relativa de jugarlo en cada episodio; arriba `franco_dificil` sale el doble), o una lista
  de nombres (todos con peso 1). Un peso 0 deja el nivel cargado sin jugarse.
- **`validacion`**: niveles que **no** se usan para entrenar (held out):
  - un nombre o una lista de nombres: esos niveles;
  - un número `N`: sortea `N` de los de `niveles` y los saca del entrenamiento (reproducible con `--val-seed`, por defecto 0);
  - `null`: no reserva ninguno; valida en uno de los de entrenamiento elegido al azar (`resumen.md` lo aclara).
- Las claves que empiezan con `_` se ignoran (sirven de comentario). Ver `niveles/ejemplo.json`.
- `--train-phase fixed/random` se aplica igual a todos los niveles. Todos tienen que tener la misma altura (7) y arrancar en `x = 2.0`;
  `tests/test_env.py` comprueba con el oráculo que cada nivel registrado se gane desde las 10 fases de evaluación.
- **Evaluación**: cada modelo se evalúa en todos los niveles de entrenamiento × 10 fases (`eval_fases*.csv`, con columna `level`) y,
  aparte, en los de validación (`eval_fases*_val.csv`). Se graba un GIF por nivel (fase x0=2.0) más el de la peor falla; los de validación
  con prefijo `val_`. `resumen.md` trae tablas de entrenamiento y de validación.
- **Historial de validación**: las evaluaciones periódicas también miden los niveles de validación (columnas `val_success`,
  `val_progress`, `val_mean_jumps_won` de `eval_history.csv`), pero el **mejor modelo se elige sólo con los de entrenamiento**.
  Además hay una columna de éxito por nivel (`success_<nivel>`, `val_success_<nivel>`): sirve para ver si olvida alguno.
  `monitor/*.csv` registra en la columna `level` qué nivel se jugó en cada episodio de entrenamiento.
- `config.json` guarda lo pedido (`levels`, `val_seed`) y lo que resultó (`train_levels` con pesos, `val_levels`, `val_held_out`);
  `scripts/evaluate.py` lo usa para re-evaluar con los mismos niveles. Las corridas viejas (sin niveles) siguen funcionando con `level_1`.
- Los pesos se pueden cambiar durante el entrenamiento (`env_method("set_wrapper_attr", "level_weights", [...])`), igual que la
  penalización por salto: es lo que usa la progresión de niveles (abajo).

#### Progresión de niveles (introducirlos de a poco)
Opcional, en el mismo `.json` (en la literatura se llama *curriculum learning*). Ejemplo: `niveles/nacho_progresion.json`.
```json
{"niveles": ["nacho_facil", "nacho_medio", "nacho_dificil", "nacho_muy_dificil"],
 "validacion": "level_1",
 "progresion": {"etapas": [["nacho_facil"], ["nacho_medio"], ["nacho_dificil", "nacho_muy_dificil"]],
                "desbloquear": 0.8, "ventana": 50, "piso": 0.2}}
```
- **`etapas`**: cualquier cantidad, cada una con uno o más niveles; cada nivel de entrenamiento tiene que estar en exactamente una.
  Se arranca sólo con la primera; las demás tienen peso 0 hasta desbloquearse.
- **Desbloqueo**: cuando **todos** los niveles activos ganan al menos `desbloquear` (80%) de sus últimos `ventana` (50) episodios
  de entrenamiento, se suma la etapa siguiente.
- **Pesos según el desempeño**: cada nivel activo pesa `peso_base × max(piso, 1 − éxito_reciente)`. Se practica más lo que más falla;
  un nivel recién desbloqueado entra con el peso máximo; uno dominado baja hasta `piso` pero nunca a 0, así se sigue repasando, y si el
  agente lo olvida su peso vuelve a subir solo (evita el *olvido catastrófico*). Los pesos de `niveles` multiplican a este peso.
- `desbloquear`, `ventana` y `piso` son opcionales (valores por defecto los de arriba). Si `validacion` saca un nivel que estaba en una
  etapa, se lo quita de ella. El callback es `LevelProgression` (`agents/callbacks.py`); cada cambio de etapa (y, cada `ventana`
  episodios, los pesos y el éxito por nivel) queda en `progresion.csv`. El nombre de la corrida suma `_prog`.
- Con `--init-model` la progresión vuelve a empezar desde la primera etapa. Combinada con `--ramp`, cada nivel nuevo baja la tasa de
  éxito y la rampa puede aflojar `c` un rato.
- Se combina con los términos de la recompensa (`--rewards`), que valen igual en todas las etapas. Para armar una progresión con otros
  niveles (por ejemplo los de Santi) alcanza con copiar `niveles/nacho_progresion.json` y cambiar `niveles`, `validacion` y `etapas`.

### Qué se guarda
```
runs/<run-name>/
├── config.json              parámetros de la corrida (se escribe al empezar)
├── resumen.md               tablas entre seeds: modelo final y mejor modelo, en niveles de entrenamiento y de validación
└── <algo>/seed<N>/
    ├── model.zip            modelo al terminar el entrenamiento
    ├── best_model.zip       mejor modelo de las evaluaciones periódicas  ·  best_model.json: en qué paso y con qué métricas
    ├── checkpoints/         model_<pasos>_steps.zip cada --checkpoint-freq pasos (25k)
    ├── eval_history.csv     una fila por evaluación periódica (mismos pasos que los checkpoints); is_best=1 marca cuándo mejoró el récord; val_* = validación
    ├── eval_fases.csv       evaluación del modelo final  ·  eval_fases_best.csv  del mejor modelo  ·  *_val.csv: en los niveles de validación
    ├── monitor/             un CSV por entorno con cada episodio de entrenamiento
    ├── jump_penalty.csv     evolución de c (sólo con --ramp)
    ├── progresion.csv       etapas, pesos y éxito por nivel (sólo con "progresion" en el .json de niveles)
    └── episodio_<nivel>_x0_*.gif, *_recorrido.png  (final)  ·  mejor_episodio_*  (mejor)  ·  val_*  (niveles de validación)
```
- **Mejor modelo**: cada `--eval-freq` pasos (25k; `0` lo desactiva) y al terminar se evalúa el modelo de forma determinista en las
  10 fases. Gana el de **más fases ganadas**; a igualdad, más progreso; a igualdad, **menos saltos**. Hace falta porque PPO es muy
  inestable acá: reportar sólo el último modelo es una lotería.
- **Checkpoints**: como `--checkpoint-freq` y `--eval-freq` coinciden, cada fila de `eval_history.csv` tiene su `checkpoints/model_<paso>_steps.zip`
  (el mejor es el de `best_model.json["timesteps"]`, que además se copia como `best_model.zip`). Así se conservan todos los modelos y queda claro cuál fue el mejor.
- **Nunca se pisa una corrida**: si `runs/<run-name>/` ya existe, la nueva se guarda en `runs/<run-name>_<fecha-hora>/`.
- Re-evaluar los mejores modelos: `python scripts/evaluate.py --run runs/<run-name> --best [--gif]`. El tipo de observación (`features`/`grid`/`dict`) se lee del `config.json` de la corrida; para un `--model` suelto sin `config.json` hay que pasar `--obs`.

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
# con CNN: make_train_env("grid", ...) + build_model("ppo", env, seed=0, obs_mode="grid")   ("dict" igual)
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
Cada fila de `evaluate_policy_phases` tiene: `level, x0, success, progress, return, ticks, status, n_jumps, oracle_jumps, death_x`
(un episodio por nivel y fase; los niveles son los de `level=` en los kwargs, por defecto `level_1`).

## Observaciones y CNN
`GeometryDashEnv(obs_mode=...)` ofrece tres observaciones; `build_model` elige la política de SB3 y el extractor de features según cuál se use:

| `--obs` | Observación | Política SB3 | Extractor de features |
|---|---|---|---|
| `features` | vector `float32` de 28 números: `[y/H, vy/jump_v, en_suelo, frac(x)]` + por cada una de las 12 columnas siguientes `[altura/H, hay pincho]` | `MlpPolicy` | el de SB3 (aplanar) |
| `grid` | imagen `uint8` de forma **(3, 7, 16)**, canales bloques / pinchos / jugador, ventana de 16 columnas desde el cubo | `CnnPolicy` | `GridCNN` |
| `dict` | `{"grid": (3, 7, 16), "vec": (4,)}`; el vector trae `[y/H, vy/jump_v, en_suelo, frac(x)]` | `MultiInputPolicy` | `GridVecCNN` |

**Por qué hay una CNN propia.** La `CnnPolicy` por defecto de SB3 usa `NatureCNN` (convoluciones de 8×8/stride 4, 4×4/stride 2 y 3×3), pensada para imágenes de Atari de al menos 36×36. La grilla de 7×16 es más chica y con esos strides no entra, así que `agents/cnn.py` define extractores a medida y `build_model` los pasa con `policy_kwargs`.

**`GridCNN`** (observación `grid`): `Conv2d(3→32, 3×3, padding=1)` → ReLU → `Conv2d(32→64, 3×3, padding=1)` → ReLU → aplanar (64·7·16 = 7168) → `Linear(→128)` → ReLU. Con `padding=1` la grilla no se achica y no hay pooling ni strides: en una grilla tan chica cada celda importa (un pincho corrido una columna cambia el momento del salto). Después del extractor van las capas ocultas `[64, 64]` que ya usaban los demás modelos.

**`GridVecCNN`** (observación `dict`): la grilla pasa por la misma CNN (128 features) y el vector `vec` por `Linear(4→32)` + ReLU; las dos salidas se concatenan (160 features). El vector aporta lo que la grilla sola no tiene: la **velocidad vertical** (si el cubo está subiendo o cayendo) y la fase dentro del tile.

- Hiperparámetros de los algoritmos: los mismos que con `features` (cambia sólo el extractor). Cambiar el tamaño de la red: `policy_kwargs_for("grid", extra=dict(features_extractor_kwargs=dict(features_dim=64, channels=(16, 32))))` en `agents/common.py`.
- Los modelos con CNN se guardan igual (`model.zip`) y se recargan con `load_model`; hace falta poder importar `agents.cnn`, o sea correr desde la raíz del repo.
- Desde Python: `env = make_train_env("grid", n_envs=8, seed=0, random_phase=False)` y `model = build_model("ppo", env, seed=0, obs_mode="grid")` (con `"dict"` es igual). Evaluar: `evaluate_policy_phases(model, "grid")`.
- Con CNN el entrenamiento es más lento que con `features` (más parámetros por paso); en CPU, DQN/QR-DQN son los más lentos.
- **Limitación de `features` con techos.** De cada columna sólo ve la altura del bloque **más alto** y si hay un pincho (hacia arriba o
  colgante). Con un techo o una plataforma sobre el cubo (`muy_dificil`, `franco_techo`) esa altura es la del techo: no ve el piso de
  abajo, el hueco libre ni si el pincho cuelga o está en el piso. `grid` y `dict` sí ven todos los bloques. Que el agente aprenda o no
  esos niveles con cada observación hay que medirlo (misma seed y pasos, `features` contra `grid`); no está medido todavía.

## Decisiones de diseño
- **Un tick = un paso**; velocidad horizontal constante (0.2 tiles/tick), así el nivel dura 520 pasos.
- **Acción**: 0 = nada, 1 = saltar (sólo surte efecto apoyado; equivale a mantener apretado).
- **Muerte**: choque de costado con un bloque, pincho (hitbox 0.3×0.3; el colgante `v` la tiene espejada), o caer a un pozo. Golpear un bloque con la cabeza no mata: sólo frena la subida.
- **Observación `grid`**: (3, 7, 16) = bloques / pinchos / jugador, ventana ego-céntrica. No incluye la velocidad vertical; por eso existe `dict` (grilla + vector). Los pinchos colgantes van en el mismo canal que los demás.
- **Recompensa por defecto**: progreso (suma 1 en el nivel) + 1 al ganar − 10 al morir (`death_penalty`). Está hecha de términos (`progress`, `win`, `death`, `survive`, `jump_penalty`, `salto_preciso`) que se prenden y apagan con `--rewards`, o se reemplaza entera con `reward_fn(tr)` (ver "Recompensa y `train()`") para la curva de timing del profesor.
- `reset(options={"x0": ...})` cambia la fase de arranque: sirve para evaluar generalización sobre variaciones del nivel fijo. Probado con 20 fases: todas tienen solución.

## Oráculo
Como x depende sólo del tick, el espacio de estados es un DAG por capas; se recorre con DP guardando los saltos mínimos. Devuelve el plan con **menos saltos** (13 en el nivel 1, jugable en 520 ticks, retorno 2.0). `plan.jump_x` da dónde saltar (útil para el timing reward), y `Oracle.is_winnable(state)` indica si un estado todavía es salvable.

## Dificultad de un nivel
`scripts/difficulty.py` mide, con la física y sin entrenar, lo que acordamos como dificultad: **qué tan preciso tiene que ser
cada salto** y **cuántos saltos hay que encadenar**. Sirve para clasificar los niveles en fácil / medio / difícil con un criterio
común (y justificarlo en el informe), y para probar un nivel nuevo antes de entrenar con él.
```bash
python scripts/difficulty.py                          # todos los niveles registrados en gd_env.LEVELS
python scripts/difficulty.py level_1 --detalle        # + tabla salto por salto
python scripts/difficulty.py mi_nivel.txt --ascii     # un nivel dibujado en ASCII (sin registrarlo todavía)
python scripts/difficulty.py niveles/ejemplo.json     # los niveles de un .json (entrenamiento y validación)
```
```python
from gd_env import get_level, level_difficulty
d = level_difficulty(get_level("level_1"))      # d.summary() = la fila de la tabla; d.jumps = detalle por salto
```
**Cómo se calcula.** Como el oráculo, enumera todos los estados alcanzables (~6.500 en `level_1`, 0.1 s) y, de atrás para
adelante, calcula para cada uno los saltos mínimos para ganar y la probabilidad de ganar apretando al azar. Después recorre un
camino óptimo salto por salto (despegando en el centro de cada ventana) y mide:

| Columna | Qué mide |
|---|---|
| saltos mínimos | los mismos que el oráculo |
| ventana mín. / media (ticks) | cuántos ticks de despegue distintos siguen ganando **sin gastar saltos de más** (1 tick = 0.2 tiles). La mínima es el salto más preciso del nivel |
| saltos con trampas / ticks trampa | despegues desde los que el cubo aterriza vivo pero el nivel ya no se puede ganar: el error se paga después ("el primer salto determina los siguientes") |
| cadena más larga / saltos encadenados | dos saltos seguidos están encadenados si cuándo despegás en el primero (dentro de su ventana) cambia la ventana del segundo |
| log10 P(ganar al azar) | probabilidad de ganar apretando 50/50 en cada tick: cuánto le cuesta a un agente dar con la solución explorando (−3 = uno de cada mil episodios) |
| fases con solución | de cuántas fases de arranque se puede ganar |

Cada columna es el **peor caso** entre las 10 fases de arranque de la evaluación (`--fases 1`: sólo x0 = 2.0). Con `--detalle`, por
cada salto: dónde está la ventana, cuántas trampas tiene y, si está encadenado, entre cuánto y cuánto varía la ventana del siguiente.
Funciona con cualquier nivel dibujable en ASCII, también con techos, plataformas flotantes y pinchos colgantes (la física ya resuelve los cabezazos).

**`level_1`**: 13 saltos; ventanas de 2 a 10 ticks (media 6.3). Las más chicas: el salto desde la plataforma baja al pozo (x≈78,
2 ticks) y los pinchos dobles (x≈23 y x≈65, 3 ticks). Tiene 3 saltos con trampas (escalón x≈31, escalera x≈49, plataforma baja
x≈73), una cadena de 3 saltos (pozo → escalera, x≈45–52) y P(ganar al azar) ≈ 10^-4.6 en x0 = 2.0 (uno de cada ~40.000 episodios:
por eso la política aleatoria de los tests no gana nunca). Los cortes entre fácil / medio / difícil los tenemos que acordar;
`level_1` sirve de referencia.

## Nota para SB3
`CnnPolicy` por defecto (NatureCNN) exige imágenes de al menos 36×36; la grilla (7×16) es más chica. Por eso `grid` y `dict` usan los extractores de `agents/cnn.py` (ver "Observaciones y CNN"); `obs_mode="features"` con `MlpPolicy` queda como baseline.

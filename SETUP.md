# Guía de instalación y uso (paso a paso)

Para quien acaba de clonar el repo por primera vez. Todos los comandos se corren **desde la carpeta raíz del repo**
(la que contiene `scripts/`, `agents/` y `gd_env/`). Se recomienda **Python 3.10 o superior**.

## 1. Clonar el repo

```bash
git clone https://github.com/Khay150/Reinforcement-Learning-Geometry-Dash.git
cd Reinforcement-Learning-Geometry-Dash
```

## 2. Crear y activar un entorno virtual

> ⚠️ No llamar al entorno `gd_env`: ese es el nombre del paquete del entorno de Gymnasium dentro del repo y se pisarían.
> Usá `.venv`.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar.
En CMD: `.venv\Scripts\activate.bat`.

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Si el entorno quedó activo, el prompt muestra `(.venv)` al principio.

## 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Instala `gymnasium`, `numpy`, `pillow`, `pytest`, `stable-baselines3`, `sb3-contrib` (para QR-DQN) y `torch` (también lo usa la CNN de `agents/cnn.py`).
PyTorch pesa bastante (varios cientos de MB), puede tardar unos minutos. Alcanza con CPU; no hace falta GPU.

## 4. Chequear que todo funciona

```bash
# a) tests del entorno, los callbacks, la CNN, la dificultad y las recompensas (deberían pasar todos; si no está sb3-contrib se saltean 3, los de QR-DQN; tardan ~1 min)
pytest tests

# b) el oráculo resuelve el nivel y lo reproduce (debe terminar con status=win progreso=100%)
python scripts/run_oracle.py --ascii

# c) ver un nivel registrado en ASCII (# bloque, ^ pincho, v pincho colgante) y su dificultad
python -c "from gd_env import get_level, to_ascii; print(to_ascii(get_level('franco_techo')))"
python scripts/difficulty.py franco_techo
python scripts/difficulty.py santi_facil santi_dificil      # los niveles de Santi (todos los registrados: sin argumentos)
```

## 5. Prueba rápida de entrenamiento (~30 s)

```bash
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 --timesteps 12000 --run-name humo --no-gif
```
Si termina e imprime una tabla resumen, está todo bien. Deja resultados en `runs/humo/`.

Lo mismo con las observaciones que usan CNN (`grid` y `dict`):
```bash
python scripts/train.py --algos ppo dqn --obs grid --seeds 0 --timesteps 12000 --run-name humo_grid --no-gif
python scripts/train.py --algos ppo dqn --obs dict --seeds 0 --timesteps 12000 --run-name humo_dict --no-gif
```

## 6. Entrenamiento real

```bash
# PPO y DQN, 3 seeds cada uno (observación "features", nivel fijo)
python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000

# E1: comparación de algoritmos (PPO, A2C, DQN y QR-DQN), 3 seeds cada uno = 12 corridas
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos

# con penalización por salto efectivo (el run queda con sufijo _jp0.05)
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.05

# cambiar la recompensa por parámetros (si no se pasan, valen los valores por defecto: win 1, muerte 10, salto 0)
python scripts/train.py --algos ppo --seeds 0 --win-bonus 2 --death-penalty 5 --jump-penalty 0.05

# bonus por seguir vivo: cada N ticks vivo da survive_bonus (def. N = 1). Compensa que en niveles largos se acumulan más saltos penalizados
python scripts/train.py --algos ppo --seeds 0 --jump-penalty 0.05 --survive-bonus 0.1 --survive-interval 20
# o en proporción a jump_penalty (bonus por tick = RATE x c; sigue a la rampa y sirve igual en niveles cortos y largos)
python scripts/train.py --algos ppo --seeds 0 --jump-penalty 0.05 --survive-rate 0.1 --survive-interval 20

# elegir qué términos de la recompensa están prendidos (progress, win, death, survive, jump_penalty, salto_preciso).
# Sin --rewards queda la de siempre. Acá: sólo progreso, meta y muerte, más la recompensa de salto preciso de Santi
python scripts/train.py --algos ppo --seeds 0 --rewards progress win death salto_preciso
# la de siempre pero sin castigo por morir (los parámetros de un término apagado dan error, p. ej. --death-penalty 5)
python scripts/train.py --algos ppo --seeds 0 --rewards progress win survive jump_penalty --jump-penalty 0.05
# ojo: jump_penalty y salto_preciso son excluyentes (tampoco se puede usar --ramp con salto_preciso)

# recompensa propia: una función de un .py (ver ejemplos/reward_ejemplo.py); correr siempre desde la raíz del repo
python scripts/train.py --algos ppo --seeds 0 --reward-fn ejemplos/reward_ejemplo.py:penaliza_saltos

# penalización por salto con rampa: c arranca en 0 y sube de a poco, sólo cuando el agente ya gana
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.1 --ramp

# fine-tuning: seguir entrenando un modelo ya entrenado, ahora con penalización por salto
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --jump-penalty 0.05 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip

# con fase de arranque aleatoria en cada episodio (mide generalización)
python scripts/train.py --algos ppo dqn --train-phase random --run-name features_random

# con CNN: observación "grid" (imagen 3x7x16) o "dict" (grilla + vector); el run queda como grid_fixed / dict_fixed
python scripts/train.py --algos ppo dqn --obs grid --seeds 0 1 2 --timesteps 300000
python scripts/train.py --algos ppo dqn --obs dict --seeds 0 1 2 --timesteps 300000

# varios niveles: un .json con los niveles de entrenamiento (con peso) y los de validación (held out)
python scripts/train.py --algos ppo --levels niveles/ejemplo.json

# nivel con techo de pinchos colgantes: se gana sin saltar nunca (mira si el agente aprende a no saltar, p. ej. con --jump-penalty)
python scripts/train.py --algos ppo --seeds 0 --levels franco_sin_saltar --timesteps 100000

# los 5 niveles de Santi (pesos crecientes según dificultad; valida en level_1, held out), con su recompensa
python scripts/train.py --algos ppo --levels niveles/santi.json --rewards progress win death salto_preciso

# con CNN y varios niveles a la vez, nombres sueltos de gd_env.LEVELS (peso 1 c/u; tabla en "Niveles registrados" del README)
python scripts/train.py --algos ppo --obs grid --levels franco_facil franco_medio franco_dificil franco_techo

# progresión: los niveles se introducen de a poco, por etapas, y se juega más lo que más falla (ver "Progresión de niveles" en el README)
python scripts/train.py --algos ppo --obs grid --levels niveles/nacho_progresion.json
```
Opciones útiles: `--algos ppo` (solo uno; valores posibles: `ppo`, `a2c`, `dqn`, `qrdqn`), `--seeds 0` (una seed),
`--n-envs 8` (entornos en paralelo, solo PPO y A2C), `--jump-penalty 0.05` (penaliza cada salto efectivo),
`--win-bonus`, `--death-penalty`, `--progress-scale`, `--survive-bonus`, `--survive-interval`, `--survive-rate` (parámetros de la recompensa), `--rewards <términos>` (qué términos de la recompensa quedan prendidos), `--reward-fn archivo.py:funcion` (recompensa propia; no se combina con `--rewards`),
`--obs features|grid|dict` (tipo de observación: vector con MLP, o grilla / grilla + vector con CNN; ver el README),
`--ramp` (la penalización sube gradualmente), `--levels <nombres o .json>` (varios niveles, y con un bloque `progresion` en el `.json`, por etapas; ver "Varios niveles" y "Progresión de niveles" en el README), `--init-model <zip>` (sigue entrenando un modelo),
`--no-gif` (no graba GIFs), `--out runs` (carpeta de salida). Todas las opciones: `python scripts/train.py -h`.

Por cada agente se guarda en `runs/<run-name>/<algo>/seed<N>/`:
`model.zip` (final), `best_model.zip` (el mejor de las evaluaciones periódicas, ver el README), `checkpoints/`,
`monitor/` (curva de aprendizaje), `eval_history.csv`, `eval_fases.csv` (y `eval_fases_val.csv` en los niveles de validación),
`episodio_<nivel>_x0_<x0>.gif` y `*_recorrido.png` (los de validación con prefijo `val_`).
Con `--ramp` también `jump_penalty.csv`, y con un `.json` de niveles con `progresion`, `progresion.csv` (etapas, pesos y éxito por nivel).
La tabla comparativa entre seeds queda en `runs/<run-name>/resumen.md`. Si la carpeta de la corrida ya existe,
la nueva se guarda con fecha y hora en el nombre (no se pisa nada).

También se puede entrenar desde Python (notebook o script propio), con los mismos parámetros:
```python
from agents import train
train("ppo", seeds=[0], timesteps=200_000, win_bonus=2.0, death_penalty=5.0, jump_penalty=0.05)
train("ppo", reward_fn=mi_recompensa)      # función propia; ver "Recompensa y `train()`" en el README
train("ppo", levels="niveles/santi.json", rewards=["progress", "win", "death", "salto_preciso"])    # términos a elección
train("ppo", obs="grid", levels="niveles/nacho_progresion.json")      # niveles por etapas (progresión)
```

## 7. Re-evaluar modelos ya entrenados

```bash
# todos los modelos de una corrida, con 20 fases y regrabando los GIFs
python scripts/evaluate.py --run runs/features_fixed --n-phases 20 --gif

# el mejor modelo de cada seed en vez del final
python scripts/evaluate.py --run runs/features_fixed --best

# un modelo suelto (--algo: ppo, a2c, dqn o qrdqn)
python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo

# un modelo suelto entrenado con CNN, fuera de una carpeta de corrida (sin config.json): indicar --obs
python scripts/evaluate.py --model modelos/ppo_grid.zip --algo ppo --obs grid
```
Con `--run`, el tipo de observación (`features`/`grid`/`dict`) se lee del `config.json` de la corrida.

## 8. Grabar un GIF

```bash
# del oráculo (solución óptima)
python scripts/record.py --out videos/oraculo

# de un modelo entrenado
python scripts/record.py --model runs/features_fixed/ppo/seed0/model.zip --algo PPO --obs features --out videos/ppo

# de un modelo con CNN: --obs tiene que ser el mismo con el que se entrenó (grid o dict)
python scripts/record.py --model runs/grid_fixed/ppo/seed0/model.zip --algo PPO --obs grid --out videos/ppo_grid
```
`--algo` acepta PPO, A2C, DQN y QRDQN.
Genera `<out>.gif` y `<out>_recorrido.png`. Con `--x0 2.5` se cambia la fase de arranque (entre 2.0 y 2.95).

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError: gymnasium` / `stable_baselines3` | El entorno virtual no está activo o falta el paso 3. |
| `ModuleNotFoundError: gd_env` / `agents` | Correr los scripts desde la raíz del repo (`python scripts/train.py ...`), no desde dentro de `scripts/`. |
| `python` no se encuentra | En Linux/macOS usar `python3`; en Windows probar `py -3`. |
| `No module named 'torch'` tras instalar | Reintentar `pip install torch`; chequear que sea Python 3.10+ de 64 bits. |
| Entrenamiento muy lento | Bajar `--timesteps` o usar `--seeds 0` / `--algos ppo`. |
| `QR-DQN requiere sb3-contrib` | `pip install sb3-contrib` (o repetir el paso 3 con `requirements.txt` actualizado). |
| `ValueError: 'jump_penalty' y 'salto_preciso' son excluyentes` (o `pasaste ... pero el término ... no está en rewards`) | Con `--rewards` hay que elegir uno de los dos (`jump_penalty` o `salto_preciso`) y no pasar parámetros de un término apagado (p. ej. `--death-penalty` sin `death`). Ver "Recompensa y `train()`" en el README. |
| `ValueError: progresion: cada nivel de entrenamiento tiene que estar en exactamente una etapa` | En el `.json` con `progresion`, todos los niveles de `niveles` (menos los de `validacion`) tienen que aparecer en una sola etapa. Ver "Progresión de niveles" en el README. |
| `ValueError` / error de forma al cargar o evaluar un modelo con CNN | El `--obs` no coincide con el del entrenamiento (un modelo `grid` no acepta `features` ni `dict`). Con `evaluate.py --run` se toma solo del `config.json`; con un modelo suelto hay que indicarlo. |
| `ModuleNotFoundError: agents.cnn` al cargar un modelo con CNN | Correr desde la raíz del repo: el `.zip` guarda la clase del extractor y necesita poder importar `agents`. |
| Usar Colab | `!git clone <url>`, `%cd Reinforcement-Learning-Geometry-Dash`, `!pip install -r requirements.txt` y luego los mismos comandos con `!`. |

Más detalle del diseño del entorno, el oráculo y los hiperparámetros en el [README](README.md).

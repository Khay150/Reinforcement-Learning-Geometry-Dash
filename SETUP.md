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
Instala `gymnasium`, `numpy`, `pillow`, `pytest`, `stable-baselines3`, `sb3-contrib` (para QR-DQN) y `torch`.
PyTorch pesa bastante (varios cientos de MB), puede tardar unos minutos. Alcanza con CPU; no hace falta GPU.

## 4. Chequear que todo funciona

```bash
# a) tests del entorno (deberían pasar 17)
pytest tests

# b) el oráculo resuelve el nivel y lo reproduce (debe terminar con status=win progreso=100%)
python scripts/run_oracle.py --ascii
```

## 5. Prueba rápida de entrenamiento (~30 s)

```bash
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 --timesteps 12000 --run-name humo --no-gif
```
Si termina e imprime una tabla resumen, está todo bien. Deja resultados en `runs/humo/`.

## 6. Entrenamiento real

```bash
# PPO y DQN, 3 seeds cada uno (observación "features", nivel fijo)
python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000

# E1: comparación de algoritmos (PPO, A2C, DQN y QR-DQN), 3 seeds cada uno = 12 corridas
python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos

# con penalización por salto efectivo (el run queda con sufijo _jp0.05)
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.05

# penalización por salto con rampa: c arranca en 0 y sube de a poco, sólo cuando el agente ya gana
python scripts/train.py --algos ppo --seeds 0 1 2 --jump-penalty 0.1 --ramp

# fine-tuning: seguir entrenando un modelo ya entrenado, ahora con penalización por salto
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --jump-penalty 0.05 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip

# con fase de arranque aleatoria en cada episodio (mide generalización)
python scripts/train.py --algos ppo dqn --train-phase random --run-name features_random
```
Opciones útiles: `--algos ppo` (solo uno; valores posibles: `ppo`, `a2c`, `dqn`, `qrdqn`), `--seeds 0` (una seed),
`--n-envs 8` (entornos en paralelo, solo PPO y A2C), `--jump-penalty 0.05` (penaliza cada salto efectivo),
`--ramp` (la penalización sube gradualmente), `--init-model <zip>` (sigue entrenando un modelo),
`--no-gif` (no graba GIFs), `--out runs` (carpeta de salida). Todas las opciones: `python scripts/train.py -h`.

Por cada agente se guarda en `runs/<run-name>/<algo>/seed<N>/`:
`model.zip` (final), `best_model.zip` (el mejor de las evaluaciones periódicas, ver el README), `checkpoints/`,
`monitor/` (curva de aprendizaje), `eval_history.csv`, `eval_fases.csv`, `episodio_x0_<x0>.gif` y `*_recorrido.png`.
La tabla comparativa entre seeds queda en `runs/<run-name>/resumen.md`. Si la carpeta de la corrida ya existe,
la nueva se guarda con fecha y hora en el nombre (no se pisa nada).

## 7. Re-evaluar modelos ya entrenados

```bash
# todos los modelos de una corrida, con 20 fases y regrabando los GIFs
python scripts/evaluate.py --run runs/features_fixed --n-phases 20 --gif

# el mejor modelo de cada seed en vez del final
python scripts/evaluate.py --run runs/features_fixed --best

# un modelo suelto (--algo: ppo, a2c, dqn o qrdqn)
python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo
```

## 8. Grabar un GIF

```bash
# del oráculo (solución óptima)
python scripts/record.py --out videos/oraculo

# de un modelo entrenado
python scripts/record.py --model runs/features_fixed/ppo/seed0/model.zip --algo PPO --obs features --out videos/ppo
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
| Usar Colab | `!git clone <url>`, `%cd Reinforcement-Learning-Geometry-Dash`, `!pip install -r requirements.txt` y luego los mismos comandos con `!`. |

Más detalle del diseño del entorno, el oráculo y los hiperparámetros en el [README](README.md).

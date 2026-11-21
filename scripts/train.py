"""Entrena PPO / A2C / DQN / QR-DQN (Stable-Baselines3 y sb3-contrib) sobre GeometryDash-v0, evalúa en varias fases de arranque y graba GIFs.

Ejemplos
--------
  python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000
  python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos
  python scripts/train.py --algos ppo --seeds 0 --timesteps 20000 --run-name humo     # prueba rápida
  python scripts/train.py --algos ppo --jump-penalty 0.1 --ramp                       # penalización por salto con rampa
  python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --init-model runs/X/ppo/seed0/best_model.zip

Salida (por agente) en  runs/<run-name>/<algo>/seed<N>/ :
  model.zip, best_model.zip (+ .json), checkpoints/, monitor/, eval_history.csv, eval_fases*.csv, GIFs y *_recorrido.png
Resumen entre seeds: runs/<run-name>/resumen.md. Una corrida nunca pisa a otra. Detalle de las opciones y archivos: README.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.callbacks import CheckpointCallback

from agents import ALGOS, EVAL_PHASES, ON_POLICY, build_model, make_train_env, summarize
from agents.callbacks import BestModelCallback, PerformanceJumpPenaltyRamp
from agents.common import load_for_training, load_model
from agents.evaluate import describe
from agents.runs import evaluate_and_record, reward_kwargs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=["ppo", "dqn"], choices=list(ALGOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=300_000,
                    help="pasos de entrenamiento (con --init-model: pasos adicionales)")
    ap.add_argument("--obs", default="features", choices=["features"],
                    help="grid/dict requieren un extractor CNN propio (todavía no implementado)")
    ap.add_argument("--train-phase", default="fixed", choices=["fixed", "random"],
                    help="fixed: siempre x0=2.0 (nivel fijo). random: x0 aleatorio en cada episodio")
    ap.add_argument("--n-envs", type=int, default=8, help="entornos en paralelo (sólo PPO/A2C; DQN y QR-DQN usan 1)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-gif", action="store_true")

    g = ap.add_argument_group("recompensa")
    g.add_argument("--jump-penalty", type=float, default=0.0,
                   help="se resta en cada salto efectivo (p. ej. 0.05); 0 = sin penalización. Con --ramp es el valor máximo")
    g.add_argument("--death-penalty", type=float, default=None,
                   help="penalización al morir (por defecto, la de GeometryDashEnv)")

    g = ap.add_argument_group("rampa de la penalización por salto (opcional)")
    g.add_argument("--ramp", action="store_true",
                   help="c arranca en 0 y sube hasta --jump-penalty sólo cuando el agente ya gana")
    g.add_argument("--ramp-step", type=float, default=0.01, help="cuánto sube/baja c en cada decisión (def. 0.01)")
    g.add_argument("--ramp-window", type=int, default=20, help="episodios de entrenamiento por decisión (def. 20)")
    g.add_argument("--ramp-up", type=float, default=0.8, help="tasa de éxito para subir c (def. 0.8)")
    g.add_argument("--ramp-down", type=float, default=0.4, help="tasa de éxito para bajar c (def. 0.4)")

    g = ap.add_argument_group("guardado")
    g.add_argument("--checkpoint-freq", type=int, default=25_000, help="modelo intermedio cada N pasos (0 = no)")
    g.add_argument("--eval-freq", type=int, default=25_000, help="evalúa cada N pasos y guarda best_model.zip (0 = no)")
    g.add_argument("--init-model", default=None,
                   help="seguir entrenando este .zip en vez de empezar de cero (un solo algoritmo en --algos)")

    a = ap.parse_args()
    if a.ramp and not a.jump_penalty:
        ap.error("--ramp necesita --jump-penalty > 0 (el valor máximo de c)")
    if a.init_model and len(a.algos) != 1:
        ap.error("--init-model requiere un solo algoritmo en --algos (el del modelo)")
    if a.init_model and not Path(a.init_model).is_file():
        ap.error(f"no existe el modelo {a.init_model}")
    return a


def run_name(a) -> str:
    parts = [a.obs, a.train_phase]
    if a.jump_penalty:
        parts.append(f"jp{a.jump_penalty:g}")
    if a.ramp:
        parts.append("ramp")
    if a.death_penalty is not None:
        parts.append(f"dp{a.death_penalty:g}")
    if a.init_model:
        parts.append("continuado")
    return a.run_name or "_".join(parts)


def unique_dir(path: Path) -> Path:
    """No pisar corridas anteriores: si la carpeta ya existe, se agrega fecha y hora."""
    if not path.exists():
        return path
    new = path.with_name(f"{path.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[aviso] {path} ya existe: esta corrida se guarda en {new}", flush=True)
    return new


def make_callbacks(a, d: Path, n_envs: int, env_kwargs: dict) -> list:
    callbacks = []
    if a.checkpoint_freq > 0:          # save_freq cuenta llamadas al VecEnv: cada una son n_envs pasos
        callbacks.append(CheckpointCallback(max(a.checkpoint_freq // n_envs, 1), str(d / "checkpoints"), "model"))
    if a.ramp:
        callbacks.append(PerformanceJumpPenaltyRamp(a.jump_penalty, a.ramp_step, a.ramp_window, a.ramp_up,
                                                    a.ramp_down, log_path=d / "jump_penalty.csv"))
    if a.eval_freq > 0:                # después de la rampa: así registra el c vigente en cada evaluación
        callbacks.append(BestModelCallback(a.eval_freq, d, a.obs, **env_kwargs))
    return callbacks


def train_one(a, algo: str, seed: int, d: Path, env_kwargs: dict):
    """Entrena un agente, lo guarda y evalúa el modelo final y el mejor. Devuelve (filas_final, filas_mejor)."""
    n_envs = a.n_envs if algo in ON_POLICY else 1
    env = make_train_env(a.obs, n_envs, seed, random_phase=a.train_phase == "random",
                         monitor_dir=str(d / "monitor"), **env_kwargs)
    if a.init_model:
        model = load_for_training(algo, a.init_model, env, seed)
    else:
        model = build_model(algo, env, seed, a.obs)

    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, callback=make_callbacks(a, d, n_envs, env_kwargs))
    model.save(d / "model")
    env.close()

    final = evaluate_and_record(model, d / "model.zip", a.obs, env_kwargs, gif=not a.no_gif)
    print(f"[{algo} seed{seed}] {time.time() - t0:.0f}s  final: {describe(final)}", flush=True)
    if not (d / "best_model.zip").exists():
        return final, None
    best = evaluate_and_record(load_model(algo, str(d / "best_model.zip")), d / "best_model.zip", a.obs, env_kwargs,
                               gif=not a.no_gif)
    step = json.loads((d / "best_model.json").read_text())["timesteps"]
    print(f"[{algo} seed{seed}] mejor (paso {step}): {describe(best)}", flush=True)
    return final, best


def main():
    a = parse_args()
    env_kwargs = reward_kwargs(vars(a))
    root = unique_dir(Path(a.out) / run_name(a))
    root.mkdir(parents=True)
    # config al principio: si la corrida se corta, igual queda registrado con qué se entrenó
    (root / "config.json").write_text(json.dumps({**vars(a), "eval_phases": EVAL_PHASES}, indent=2))

    finals, bests = {}, {}
    for algo in a.algos:
        for seed in a.seeds:
            d = root / algo / f"seed{seed}"
            d.mkdir(parents=True)
            final, best = train_one(a, algo, seed, d, env_kwargs)
            name = f"{algo.upper()} ({root.name})"
            finals.setdefault(name, []).append(final)
            if best:
                bests.setdefault(name, []).append(best)

    text = "## Modelo final\n\n" + summarize(finals) + "\n"
    if bests:
        text += "\n## Mejor modelo (best_model.zip)\n\n" + summarize(bests) + "\n"
    (root / "resumen.md").write_text(text)
    print(f"\n{text}\nResultados en {root}")


if __name__ == "__main__":
    main()

"""Entrena PPO / A2C / DQN / QR-DQN (Stable-Baselines3 y sb3-contrib) sobre GeometryDash-v0, evalúa en varias fases de arranque y graba GIFs.

Ejemplos
--------
  python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000
  python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos
  python scripts/train.py --algos ppo --seeds 0 --timesteps 20000 --run-name humo     # prueba rápida

Salida (por agente) en  runs/<run-name>/<algo>/seed<N>/ :
  model.zip, monitor/ (curva de aprendizaje), eval_fases.csv, episodio_x0_<x0>.gif, *_recorrido.png
Resumen entre seeds: runs/<run-name>/resumen.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import ALGOS, EVAL_PHASES, ON_POLICY, build_model, evaluate_policy_phases, make_train_env, summarize
from agents.evaluate import save_rows
from gd_env import GeometryDashEnv, record_episode


def record_agent(model, obs_mode: str, out_dir: Path, rows: list[dict], frame_skip: int = 2):
    """GIF + mapa del recorrido: fase por defecto (x0=2.0) y, si falla alguna, la peor fase."""
    targets = [2.0]
    failed = [r for r in rows if not r["success"]]
    if failed:
        targets.append(min(failed, key=lambda r: r["progress"])["x0"])
    for x0 in targets:
        env = GeometryDashEnv(obs_mode=obs_mode)
        stem = out_dir / f"episodio_x0_{x0:.2f}"
        record_episode(env, model, f"{stem}.gif", f"{stem}_recorrido.png", frame_skip=frame_skip,
                       options={"x0": float(x0)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=["ppo", "dqn"], choices=list(ALGOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--obs", default="features", choices=["features"],
                    help="grid/dict requieren un extractor CNN propio (todavía no implementado)")
    ap.add_argument("--train-phase", default="fixed", choices=["fixed", "random"],
                    help="fixed: siempre x0=2.0 (nivel fijo). random: x0 aleatorio en cada episodio")
    ap.add_argument("--n-envs", type=int, default=8, help="entornos en paralelo (sólo PPO/A2C; DQN y QR-DQN usan 1)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--jump-penalty", type=float, default=0.0,
                    help="recompensa restada en cada salto efectivo (p. ej. 0.02); 0 = sin penalización")
    a = ap.parse_args()

    run_name = a.run_name or f"{a.obs}_{a.train_phase}" + (f"_jp{a.jump_penalty:g}" if a.jump_penalty else "")
    root = Path(a.out) / run_name
    results: dict[str, list[list[dict]]] = {}

    for algo in a.algos:
        for seed in a.seeds:
            d = root / algo / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            env = make_train_env(a.obs, n_envs=a.n_envs if algo in ON_POLICY else 1, seed=seed,
                                 random_phase=a.train_phase == "random", monitor_dir=str(d / "monitor"),
                                 jump_penalty=a.jump_penalty)
            model = build_model(algo, env, seed, a.obs)
            t0 = time.time()
            model.learn(total_timesteps=a.timesteps)
            model.save(d / "model")
            rows = evaluate_policy_phases(model, a.obs, jump_penalty=a.jump_penalty)
            save_rows(rows, d / "eval_fases.csv")
            if not a.no_gif:
                record_agent(model, a.obs, d, rows)
            ok = sum(r["success"] for r in rows)
            print(f"[{algo} seed{seed}] {time.time()-t0:5.0f}s  gana {ok}/{len(rows)} fases  "
                  f"progreso medio {100*sum(r['progress'] for r in rows)/len(rows):.0f}%  "
                  f"saltos {sum(r['n_jumps'] for r in rows)/len(rows):.1f} (oráculo {rows[0]['oracle_jumps']})", flush=True)
            jp = f", jp={a.jump_penalty:g}" if a.jump_penalty else ""
            results.setdefault(f"{algo.upper()} ({a.obs}, train={a.train_phase}{jp})", []).append(rows)
            env.close()

    table = summarize(results)
    (root / "config.json").write_text(json.dumps({**vars(a), "eval_phases": EVAL_PHASES}, indent=2))
    (root / "resumen.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()

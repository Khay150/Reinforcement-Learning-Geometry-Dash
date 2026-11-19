"""Graba un episodio a GIF + mapa del recorrido. Por defecto usa el oráculo.

Con un modelo de SB3:  python scripts/record.py --model modelos/ppo.zip --algo PPO --obs features
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gd_env import GeometryDashEnv, OraclePolicy, record_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="ruta a un modelo de Stable-Baselines3 (omitir = oráculo)")
    ap.add_argument("--algo", default="PPO", help="PPO, A2C, DQN o QRDQN (mayúsculas o minúsculas)")
    ap.add_argument("--obs", default="features", choices=["grid", "features", "dict"])
    ap.add_argument("--x0", type=float, default=None)
    ap.add_argument("--out", default="videos/episodio")
    ap.add_argument("--frame-skip", type=int, default=2)
    a = ap.parse_args()

    env = GeometryDashEnv(obs_mode=a.obs)
    if a.model:
        from agents.common import load_model
        policy = load_model(a.algo, a.model)
    else:
        policy = OraclePolicy(env)
    out = record_episode(env, policy, f"{a.out}.gif", f"{a.out}_recorrido.png",
                         frame_skip=a.frame_skip, options={"x0": a.x0} if a.x0 is not None else None)
    print(f"status={out['status']} progreso={out['progress']:.0%} retorno={out['return']:.2f} -> {a.out}.gif")


if __name__ == "__main__":
    main()

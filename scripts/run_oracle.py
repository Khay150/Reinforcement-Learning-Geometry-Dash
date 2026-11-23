"""Resuelve el nivel fijo con el oráculo y lo reproduce en el entorno Gymnasium."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importan gd_env y agents

import gd_env
from gd_env import GeometryDashEnv, Oracle, level_1, to_ascii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", type=float, default=None, help="x inicial alternativo (variación de fase)")
    ap.add_argument("--ascii", action="store_true", help="imprime el nivel")
    args = ap.parse_args()

    env = GeometryDashEnv(level=level_1(), obs_mode="features")
    if args.ascii:
        print(to_ascii(env.level), "\n")

    obs, info = env.reset(options={"x0": args.x0} if args.x0 is not None else None)
    oracle = Oracle(env.level, env.p)           # planificador exacto (programación dinámica)
    plan = oracle.plan(env.state)
    if plan is None:
        print("El nivel NO tiene solución con esta física."); return 1
    print(f"Plan: {len(plan.actions)} ticks, {plan.n_jumps} saltos")
    print("Saltos en x =", [round(x, 2) for x in plan.jump_x])

    total, done = 0.0, False                    # se reproduce el plan en el entorno para comprobar que de verdad gana
    while not done:
        a = oracle.action(env.state)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    print(f"status={info['status']} progreso={info['progress']:.0%} retorno={total:.3f} ticks={info['t']}")
    return 0 if info["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

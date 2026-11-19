"""Evaluación de una política en varias fases de arranque (x0) + resumen entre seeds."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gd_env import GeometryDashEnv, Oracle

from .common import EVAL_PHASES


def evaluate_policy_phases(policy, obs_mode: str = "features", phases=EVAL_PHASES, reward_fn=None,
                           jump_penalty: float = 0.0) -> list[dict]:
    """Corre un episodio determinista por cada x0. ``policy`` = cualquier objeto con ``.predict``."""
    env = GeometryDashEnv(obs_mode=obs_mode, reward_fn=reward_fn, jump_penalty=jump_penalty)
    oracle = Oracle(env.level, env.p)
    rows = []
    for x0 in phases:
        obs, info = env.reset(seed=0, options={"x0": float(x0)})
        plan = oracle.plan(env.state)
        done, ret = False, 0.0
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(np.asarray(a).reshape(-1)[0]))
            ret += r
            done = term or trunc
        rows.append({"x0": round(float(x0), 3), "success": int(info["success"]), "progress": round(info["progress"], 4),
                     "return": round(ret, 4), "ticks": info["t"], "status": info["status"],
                     "n_jumps": len(env._jumps), "oracle_jumps": plan.n_jumps if plan else -1,
                     "death_x": round(info["x"], 2) if not info["success"] else ""})
    return rows


def save_rows(rows: list[dict], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(results: dict[str, list[list[dict]]]) -> str:
    """results[nombre_agente] = lista (una por seed) de filas por fase. Devuelve una tabla markdown."""
    lines = ["| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) "
             "| seeds que ganan en todas las fases |",
             "|---|---|---|---|---|---|---|"]
    for name, per_seed in results.items():
        succ = [np.mean([r["success"] for r in rows]) for rows in per_seed]
        prog = [np.mean([r["progress"] for r in rows]) for rows in per_seed]
        ret = [np.mean([r["return"] for r in rows]) for rows in per_seed]
        won = [r for rows in per_seed for r in rows if r["success"]]
        ratio = f"{np.mean([r['n_jumps'] / r['oracle_jumps'] for r in won]):.2f}" if won else "-"
        full = sum(s == 1.0 for s in succ)
        lines.append(f"| {name} | {len(per_seed)} | {100*np.mean(succ):.0f} ± {100*np.std(succ):.0f} | "
                     f"{100*np.mean(prog):.0f}% ± {100*np.std(prog):.0f} | {np.mean(ret):.2f} ± {np.std(ret):.2f} | "
                     f"{ratio} | {full}/{len(per_seed)} |")
    return "\n".join(lines)

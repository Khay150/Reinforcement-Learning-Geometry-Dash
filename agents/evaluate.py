"""Evaluación de una política en varias fases de arranque (x0) + resumen entre seeds."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from gd_env import GeometryDashEnv, Oracle
from gd_env import physics as ph

from .common import EVAL_PHASES


@lru_cache(maxsize=None)
def oracle_jumps(x0: float) -> int:
    """Saltos mínimos del oráculo en el nivel por defecto arrancando en ``x0``.
    Se calcula una sola vez por fase (planificar tarda ~0.15 s y se evalúa muchas veces)."""
    env = GeometryDashEnv()
    plan = Oracle(env.level, env.p).plan(ph.initial_state(env.level, env.p, x0))
    return plan.n_jumps if plan else -1


def evaluate_policy_phases(policy, obs_mode: str = "features", phases=EVAL_PHASES, **env_kwargs) -> list[dict]:
    """Corre un episodio determinista por cada x0. ``policy`` = cualquier objeto con ``.predict``.
    ``env_kwargs`` va directo a ``GeometryDashEnv`` (p. ej. ``jump_penalty``, ``death_penalty``, ``reward_fn``)."""
    env = GeometryDashEnv(obs_mode=obs_mode, **env_kwargs)
    rows = []
    for x0 in phases:
        obs, info = env.reset(seed=0, options={"x0": float(x0)})
        done, ret = False, 0.0
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(np.asarray(a).reshape(-1)[0]))
            ret += r
            done = term or trunc
        rows.append({"x0": round(float(x0), 3), "success": int(info["success"]), "progress": round(info["progress"], 4),
                     "return": round(ret, 4), "ticks": info["t"], "status": info["status"],
                     "n_jumps": len(env._jumps), "oracle_jumps": oracle_jumps(float(x0)),
                     "death_x": round(info["x"], 2) if not info["success"] else ""})
    return rows


def metrics(rows: list[dict]) -> dict:
    """Resumen de una evaluación: fracción de fases ganadas, progreso medio y saltos medios en las ganadas."""
    won = [r["n_jumps"] for r in rows if r["success"]]
    return {"success": float(np.mean([r["success"] for r in rows])),
            "progress": float(np.mean([r["progress"] for r in rows])),
            "mean_jumps_won": float(np.mean(won)) if won else None}


def describe(rows: list[dict]) -> str:
    """Una línea legible con el resultado de una evaluación."""
    m = metrics(rows)
    jumps = f"{m['mean_jumps_won']:.1f}" if m["mean_jumps_won"] is not None else "-"
    return (f"gana {sum(r['success'] for r in rows)}/{len(rows)} fases  progreso {100 * m['progress']:.0f}%  "
            f"saltos {jumps} (oráculo {rows[0]['oracle_jumps']})")


def save_rows(rows: list[dict], path: str | Path):
    """Escribe una tabla completa (lista de dicts) a CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def append_row(row: dict, path: str | Path):
    """Agrega una fila a un CSV (escribe el encabezado si el archivo es nuevo)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def summarize(results: dict[str, list[list[dict]]]) -> str:
    """results[nombre_agente] = lista (una por seed) de filas por fase. Devuelve una tabla markdown."""
    lines = ["| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) "
             "| seeds que ganan en todas las fases |",
             "|---|---|---|---|---|---|---|"]
    for name, per_seed in results.items():
        succ = [metrics(rows)["success"] for rows in per_seed]
        prog = [metrics(rows)["progress"] for rows in per_seed]
        ret = [np.mean([r["return"] for r in rows]) for rows in per_seed]
        won = [r for rows in per_seed for r in rows if r["success"]]
        ratio = f"{np.mean([r['n_jumps'] / r['oracle_jumps'] for r in won]):.2f}" if won else "-"
        full = sum(s == 1.0 for s in succ)
        lines.append(f"| {name} | {len(per_seed)} | {100*np.mean(succ):.0f} ± {100*np.std(succ):.0f} | "
                     f"{100*np.mean(prog):.0f}% ± {100*np.std(prog):.0f} | {np.mean(ret):.2f} ± {np.std(ret):.2f} | "
                     f"{ratio} | {full}/{len(per_seed)} |")
    return "\n".join(lines)

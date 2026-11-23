"""Utilidades compartidas por ``scripts/train.py`` y ``scripts/evaluate.py``: evaluar un modelo y dejar sus archivos."""
from __future__ import annotations

import json
from pathlib import Path

from gd_env import GeometryDashEnv, load_reward_fn, record_episode

from .common import EVAL_PHASES
from .evaluate import evaluate_policy_phases, save_rows

# Nombre del modelo -> (sufijo del CSV de evaluación, prefijo de los GIFs). Así un modelo no pisa los archivos de otro.
OUTPUT_TAGS = {"model": ("", ""), "best_model": ("_best", "mejor_")}


REWARD_PARAMS = ("jump_penalty", "death_penalty", "win_bonus", "progress_scale", "survive_bonus")


def reward_kwargs(cfg: dict) -> dict:
    """Parámetros de la recompensa para ``GeometryDashEnv``, a partir de los argumentos (o de un ``config.json``).
    Los que están en None (o faltan) usan el valor por defecto del entorno; ``reward_fn`` puede ser una función o un
    texto ``'modulo:funcion'`` (como queda en config.json)."""
    kwargs = {"jump_penalty": cfg.get("jump_penalty") or 0.0}
    kwargs.update({k: cfg[k] for k in REWARD_PARAMS[1:] if cfg.get(k) is not None})
    if cfg.get("reward_fn") is not None:
        try:
            kwargs["reward_fn"] = load_reward_fn(cfg["reward_fn"])
        except Exception as e:      # p. ej. una lambda o una función definida en un notebook: no se puede recargar
            print(f"[aviso] no pude cargar reward_fn={cfg['reward_fn']!r} ({e}); se usa la recompensa por defecto "
                  f"(sólo cambia la columna 'return' de la evaluación)", flush=True)
    return kwargs


def run_config(model_path: Path) -> dict:
    """``config.json`` de la corrida a la que pertenece el modelo ({} si no hay)."""
    for folder in Path(model_path).parents:       # sube carpeta por carpeta: seed -> algo -> corrida
        if (folder / "config.json").exists():
            return json.loads((folder / "config.json").read_text())
    return {}


def record_agent(model, obs_mode: str, out_dir: Path, rows: list[dict], prefix: str = "", frame_skip: int = 2):
    """GIF + mapa del recorrido: fase por defecto (x0=2.0) y, si falla alguna, la peor fase."""
    targets = [2.0]                               # siempre se graba la fase por defecto
    failed = [r for r in rows if not r["success"]]
    if failed:
        targets.append(min(failed, key=lambda r: r["progress"])["x0"])
    for x0 in targets:
        stem = out_dir / f"{prefix}episodio_x0_{x0:.2f}"
        record_episode(GeometryDashEnv(obs_mode=obs_mode), model, f"{stem}.gif", f"{stem}_recorrido.png",
                       frame_skip=frame_skip, options={"x0": float(x0)})


def evaluate_and_record(model, model_path: Path, obs_mode: str, env_kwargs: dict, gif: bool = True,
                        phases=EVAL_PHASES) -> list[dict]:
    """Evalúa el modelo, guarda ``eval_fases<sufijo>.csv`` y (opcional) sus GIFs junto al ``.zip``."""
    # Sufijo/prefijo según el modelo (final, mejor u otro .zip) para que sus archivos no se pisen entre sí.
    suffix, prefix = OUTPUT_TAGS.get(model_path.stem, (f"_{model_path.stem}", f"{model_path.stem}_"))
    rows = evaluate_policy_phases(model, obs_mode, phases, **env_kwargs)
    save_rows(rows, model_path.parent / f"eval_fases{suffix}.csv")
    if gif:
        record_agent(model, obs_mode, model_path.parent, rows, prefix=prefix)
    return rows

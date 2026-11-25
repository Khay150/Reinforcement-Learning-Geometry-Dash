"""Utilidades compartidas por ``scripts/train.py`` y ``scripts/evaluate.py``: evaluar un modelo y dejar sus archivos."""
from __future__ import annotations

import json
from pathlib import Path

from gd_env import GeometryDashEnv, get_level, load_reward_fn, record_episode

from .common import EVAL_PHASES
from .evaluate import evaluate_policy_phases, save_rows

# Nombre del modelo -> (sufijo del CSV de evaluación, prefijo de los GIFs). Así un modelo no pisa los archivos de otro.
OUTPUT_TAGS = {"model": ("", ""), "best_model": ("_best", "mejor_")}


REWARD_PARAMS = ("jump_penalty", "death_penalty", "win_bonus", "progress_scale", "survive_bonus", "survive_interval", "survive_rate")


def reward_kwargs(cfg: dict) -> dict:
    """Parámetros de la recompensa para ``GeometryDashEnv``, a partir de los argumentos (o de un ``config.json``).
    Los que están en None (o faltan) usan el valor por defecto del entorno; ``reward_fn`` puede ser una función o un
    texto ``'modulo:funcion'`` (como queda en config.json)."""
    kwargs = {"jump_penalty": cfg.get("jump_penalty") or 0.0}
    kwargs.update({k: cfg[k] for k in REWARD_PARAMS[1:] if cfg.get(k) is not None})
    if cfg.get("rewards") is not None:      # términos prendidos (config.json de una corrida con --rewards)
        kwargs["rewards"] = list(cfg["rewards"])
    if cfg.get("reward_fn") is not None:
        try:
            kwargs["reward_fn"] = load_reward_fn(cfg["reward_fn"])
        except Exception as e:      # p. ej. una lambda o una función definida en un notebook: no se puede recargar
            print(f"[aviso] no pude cargar reward_fn={cfg['reward_fn']!r} ({e}); se usa la recompensa por defecto "
                  f"(sólo cambia la columna 'return' de la evaluación)", flush=True)
    return kwargs


def level_kwargs(cfg: dict, which: str = "train") -> dict | None:
    """Niveles para ``GeometryDashEnv`` a partir de la config: ``which="train"`` -> ``{"level", "level_weights"}``
    de los niveles de entrenamiento; ``which="val"`` -> ``{"level"}`` de los de validación (None si no hay).
    Una config vieja (sin niveles) entrena en el nivel por defecto y no tiene validación."""
    if which == "train":
        train = cfg.get("train_levels")
        if not train:
            return {}
        return {"level": [get_level(n) for n in train], "level_weights": list(train.values())}
    val = cfg.get("val_levels")
    return {"level": [get_level(n) for n in val]} if val else None


def run_config(model_path: Path) -> dict:
    """``config.json`` de la corrida a la que pertenece el modelo ({} si no hay)."""
    for folder in Path(model_path).parents:       # sube carpeta por carpeta: seed -> algo -> corrida
        if (folder / "config.json").exists():
            return json.loads((folder / "config.json").read_text())
    return {}


def record_agent(model, obs_mode: str, out_dir: Path, rows: list[dict], prefix: str = "", frame_skip: int = 2,
                 levels: dict | None = None):
    """GIF + mapa del recorrido: cada nivel en la fase por defecto (x0=2.0) y, si falla alguna, la peor (nivel, fase).
    ``levels`` = ``{"level": [...]}`` con los niveles evaluados (None = nivel por defecto)."""
    targets = list(dict.fromkeys((r["level"], 2.0) for r in rows))      # cada nivel, en la fase por defecto
    failed = [r for r in rows if not r["success"]]
    if failed:
        worst = min(failed, key=lambda r: r["progress"])
        if (worst["level"], worst["x0"]) not in targets:
            targets.append((worst["level"], worst["x0"]))
    env = GeometryDashEnv(obs_mode=obs_mode, **(levels or {}))
    for level, x0 in targets:
        stem = out_dir / f"{prefix}episodio_{level}_x0_{x0:.2f}"
        record_episode(env, model, f"{stem}.gif", f"{stem}_recorrido.png",
                       frame_skip=frame_skip, options={"x0": float(x0), "level": level})


def evaluate_and_record(model, model_path: Path, obs_mode: str, env_kwargs: dict, gif: bool = True,
                        phases=EVAL_PHASES, val: bool = False) -> list[dict]:
    """Evalúa el modelo en los niveles de ``env_kwargs``, guarda ``eval_fases<sufijo>.csv`` y (opcional) sus GIFs
    junto al ``.zip``. ``val=True``: son los niveles de validación (``..._val.csv`` y GIFs ``val_*``)."""
    # Sufijo/prefijo según el modelo (final, mejor u otro .zip) para que sus archivos no se pisen entre sí.
    suffix, prefix = OUTPUT_TAGS.get(model_path.stem, (f"_{model_path.stem}", f"{model_path.stem}_"))
    if val:
        suffix, prefix = suffix + "_val", "val_" + prefix
    rows = evaluate_policy_phases(model, obs_mode, phases, **env_kwargs)
    save_rows(rows, model_path.parent / f"eval_fases{suffix}.csv")
    if gif:
        record_agent(model, obs_mode, model_path.parent, rows, prefix=prefix,
                     levels={"level": env_kwargs["level"]} if "level" in env_kwargs else None)
    return rows

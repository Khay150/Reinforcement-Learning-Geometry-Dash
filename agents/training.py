"""``train(...)``: entrena uno o varios agentes con la recompensa que se le pase, sin pasar por la línea de comandos.

    from agents import train
    train("ppo", seeds=[0], timesteps=200_000, win_bonus=2.0, death_penalty=5.0, jump_penalty=0.05)
    train("ppo", reward_fn=mi_recompensa)        # recompensa propia: ver gd_env/rewards.py
    train("ppo", levels="niveles/ejemplo.json")  # varios niveles (con pesos) + niveles de validación

``scripts/train.py`` es sólo el parser de argumentos que llama a esta función.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from stable_baselines3.common.callbacks import CheckpointCallback

from gd_env import resolve_levels, reward_spec

from .callbacks import BestModelCallback, PerformanceJumpPenaltyRamp
from .common import EVAL_PHASES, ON_POLICY, build_model, load_for_training, load_model, make_train_env
from .evaluate import describe, summarize
from .runs import evaluate_and_record, level_kwargs, reward_kwargs


def run_name_from(cfg: dict) -> str:
    """Nombre de la carpeta de la corrida (p. ej. ``grid_fixed_jp0.05_ramp``) salvo que se pase ``run_name``."""
    parts = [cfg["obs"], cfg["train_phase"]]
    if cfg["levels"]:                       # p. ej. lv-ejemplo (de niveles/ejemplo.json) o lv-level_1+level_2
        items = [cfg["levels"]] if isinstance(cfg["levels"], str) else cfg["levels"]
        parts.append("lv-" + "+".join(Path(i).stem for i in items))
    if cfg["jump_penalty"]:
        parts.append(f"jp{cfg['jump_penalty']:g}")
    if cfg["ramp"]:
        parts.append("ramp")
    for key, tag in (("death_penalty", "dp"), ("win_bonus", "wb"), ("progress_scale", "ps"), ("survive_bonus", "sb")):
        if cfg[key] is not None:
            parts.append(f"{tag}{cfg[key]:g}")
    if cfg["reward_fn"] is not None:
        parts.append("rf_" + reward_spec(cfg["reward_fn"]).split(":")[-1].replace("<lambda>", "lambda"))
    if cfg["init_model"]:
        parts.append("continuado")
    return cfg["run_name"] or "_".join(parts)


def unique_dir(path: Path) -> Path:
    """No pisar corridas anteriores: si la carpeta ya existe, se agrega fecha y hora."""
    if not path.exists():
        return path
    new = path.with_name(f"{path.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[aviso] {path} ya existe: esta corrida se guarda en {new}", flush=True)
    return new


def _callbacks(cfg: dict, d: Path, n_envs: int, env_kwargs: dict, val_kwargs: dict) -> list:
    """Callbacks de SB3: checkpoints periódicos, rampa de la penalización por salto y mejor modelo."""
    callbacks = []
    if cfg["checkpoint_freq"] > 0:          # save_freq cuenta llamadas al VecEnv: cada una son n_envs pasos
        callbacks.append(CheckpointCallback(max(cfg["checkpoint_freq"] // n_envs, 1), str(d / "checkpoints"), "model"))
    if cfg["ramp"]:
        callbacks.append(PerformanceJumpPenaltyRamp(cfg["jump_penalty"], cfg["ramp_step"], cfg["ramp_window"],
                                                    cfg["ramp_up"], cfg["ramp_down"], log_path=d / "jump_penalty.csv"))
    if cfg["eval_freq"] > 0:                # después de la rampa: así registra el c vigente en cada evaluación
        callbacks.append(BestModelCallback(cfg["eval_freq"], d, cfg["obs"], val_env_kwargs=val_kwargs, **env_kwargs))
    return callbacks


def _train_one(cfg: dict, algo: str, seed: int, d: Path, env_kwargs: dict, val_kwargs: dict) -> dict:
    """Entrena un agente, lo guarda y evalúa el modelo final y el mejor, en los niveles de entrenamiento y en los de
    validación. Devuelve ``{"final", "best", "final_val", "best_val"}`` (filas por fase; las de "best" pueden ser None)."""
    n_envs = cfg["n_envs"] if algo in ON_POLICY else 1       # PPO/A2C: varios entornos; DQN/QR-DQN: uno
    env = make_train_env(cfg["obs"], n_envs, seed, random_phase=cfg["train_phase"] == "random",
                         monitor_dir=str(d / "monitor"), **env_kwargs)
    if cfg["init_model"]:                               # seguir entrenando un modelo ya guardado (fine-tuning)
        model = load_for_training(algo, cfg["init_model"], env, seed)
    else:                                               # modelo nuevo; obs decide MLP (features) o CNN (grid/dict)
        model = build_model(algo, env, seed, cfg["obs"])

    t0 = time.time()
    model.learn(total_timesteps=cfg["timesteps"], callback=_callbacks(cfg, d, n_envs, env_kwargs, val_kwargs))
    model.save(d / "model")                             # modelo final (puede no ser el mejor: ver best_model.zip)
    env.close()

    out = {"best": None, "best_val": None}
    out["final"] = evaluate_and_record(model, d / "model.zip", cfg["obs"], env_kwargs, gif=cfg["gif"])
    out["final_val"] = evaluate_and_record(model, d / "model.zip", cfg["obs"], val_kwargs, gif=cfg["gif"], val=True)
    print(f"[{algo} seed{seed}] {time.time() - t0:.0f}s  final: {describe(out['final'])}", flush=True)
    print(f"[{algo} seed{seed}]   validación: {describe(out['final_val'])}", flush=True)
    if not (d / "best_model.zip").exists():             # sin eval_freq no se guarda un mejor modelo
        return out
    best_model = load_model(algo, str(d / "best_model.zip"))
    out["best"] = evaluate_and_record(best_model, d / "best_model.zip", cfg["obs"], env_kwargs, gif=cfg["gif"])
    out["best_val"] = evaluate_and_record(best_model, d / "best_model.zip", cfg["obs"], val_kwargs, gif=cfg["gif"],
                                          val=True)
    step = json.loads((d / "best_model.json").read_text())["timesteps"]
    print(f"[{algo} seed{seed}] mejor (paso {step}): {describe(out['best'])}", flush=True)
    print(f"[{algo} seed{seed}]   validación: {describe(out['best_val'])}", flush=True)
    return out


def train(
    algos: str | list[str] = "ppo",
    *,
    seeds: list[int] = (0, 1, 2),
    timesteps: int = 300_000,                  # con init_model: pasos adicionales
    obs: str = "features",                     # "features" | "grid" | "dict"
    train_phase: str = "fixed",                # "fixed": x0 = 2.0 siempre; "random": x0 aleatorio en cada episodio
    n_envs: int = 8,                           # entornos en paralelo (sólo PPO/A2C)
    # ---- niveles: nombres de gd_env.LEVELS y/o un .json con pesos y validación (None = sólo level_1)
    levels: list[str] | str | None = None,
    val_seed: int = 0,                         # semilla del sorteo de niveles de validación
    # ---- recompensa: None = valor por defecto del entorno
    jump_penalty: float = 0.0,                 # se resta en cada salto efectivo; con ramp es el valor máximo
    death_penalty: float | None = None,        # castigo al morir (def. 10)
    win_bonus: float | None = None,            # premio al llegar a la meta (def. 1)
    progress_scale: float | None = None,       # lo que vale cruzar todo el nivel (def. 1)
    survive_bonus: float | None = None,        # bonus por tick vivo (def. 0)
    reward_fn: Callable | str | None = None,   # recompensa propia (reemplaza a la de por defecto), o 'archivo.py:funcion'
    # ---- rampa de la penalización por salto
    ramp: bool = False, ramp_step: float = 0.01, ramp_window: int = 20, ramp_up: float = 0.8, ramp_down: float = 0.4,
    # ---- guardado
    checkpoint_freq: int = 25_000, eval_freq: int = 25_000, init_model: str | None = None,
    run_name: str | None = None, out: str = "runs", gif: bool = True,
) -> dict:
    """Entrena ``algos`` x ``seeds`` y guarda todo en ``out/<run_name>/``. Devuelve
    ``{"root": carpeta, "final": {...}, "best": {...}, "final_val": {...}, "best_val": {...}}`` (evaluaciones por
    nivel y fase de cada agente, para ``summarize``; ``*_val`` son los niveles de validación)."""
    algos = [algos] if isinstance(algos, str) else list(algos)
    cfg = dict(algos=algos, seeds=list(seeds), timesteps=timesteps, obs=obs, train_phase=train_phase, n_envs=n_envs,
               levels=levels, val_seed=val_seed,
               jump_penalty=jump_penalty, death_penalty=death_penalty, win_bonus=win_bonus,
               progress_scale=progress_scale, survive_bonus=survive_bonus, reward_fn=reward_fn, ramp=ramp,
               ramp_step=ramp_step, ramp_window=ramp_window, ramp_up=ramp_up, ramp_down=ramp_down,
               checkpoint_freq=checkpoint_freq, eval_freq=eval_freq, init_model=init_model, run_name=run_name,
               out=out, gif=gif)
    # Combinaciones inválidas: se cortan acá, antes de gastar tiempo entrenando.
    if ramp and not jump_penalty:
        raise ValueError("ramp necesita jump_penalty > 0 (el valor máximo de c)")
    if init_model and len(algos) != 1:
        raise ValueError("init_model requiere un solo algoritmo (el del modelo)")
    if init_model and not Path(init_model).is_file():
        raise FileNotFoundError(f"no existe el modelo {init_model}")
    if isinstance(reward_fn, str):
        from gd_env import load_reward_fn
        cfg["reward_fn"] = reward_fn = load_reward_fn(reward_fn)     # falla ya si el texto está mal

    split = resolve_levels(levels, val_seed)             # niveles de entrenamiento (con peso) y de validación
    cfg.update(train_levels=split["train"], val_levels=split["val"], val_held_out=split["held_out"])
    reward = reward_kwargs(cfg)                          # parámetros de la recompensa que van al entorno
    env_kwargs = {**reward, **level_kwargs(cfg)}         # entrenamiento (y elección del mejor modelo)
    val_kwargs = {**reward, **level_kwargs(cfg, "val")}  # validación: sólo se mide
    root = unique_dir(Path(out) / run_name_from(cfg))
    root.mkdir(parents=True)
    # config al principio: si la corrida se corta, igual queda registrado con qué se entrenó
    (root / "config.json").write_text(json.dumps(
        {**cfg, "reward_fn": reward_spec(reward_fn), "eval_phases": EVAL_PHASES}, indent=2))

    # nombre del agente -> lista (una por seed) de evaluaciones, para cada modelo y conjunto de niveles
    results = {k: {} for k in ("final", "final_val", "best", "best_val")}
    for algo in algos:
        for seed in cfg["seeds"]:
            d = root / algo / f"seed{seed}"
            d.mkdir(parents=True)
            out = _train_one(cfg, algo, seed, d, env_kwargs, val_kwargs)
            name = f"{algo.upper()} ({root.name})"
            for k, rows in out.items():
                if rows:
                    results[k].setdefault(name, []).append(rows)

    # Tablas resumen entre seeds (media ± desvío): entrenamiento y validación, del modelo final y del mejor.
    train_names = ", ".join(f"{n} (peso {w:g})" for n, w in split["train"].items())
    val_note = "held out" if split["held_out"] else "no reservado: también se entrena en él"
    text = f"Niveles de entrenamiento: {train_names}\nValidación: {', '.join(split['val'])} ({val_note})\n"
    for key, title in (("final", "Modelo final"), ("best", "Mejor modelo (best_model.zip)")):
        if results[key]:
            text += f"\n## {title}: niveles de entrenamiento\n\n" + summarize(results[key]) + "\n"
            text += f"\n## {title}: validación\n\n" + summarize(results[f"{key}_val"]) + "\n"
    (root / "resumen.md").write_text(text)
    print(f"\n{text}\nResultados en {root}")
    return {"root": root, **results}

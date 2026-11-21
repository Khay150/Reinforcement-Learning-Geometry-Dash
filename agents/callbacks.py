"""Callbacks de entrenamiento (Stable-Baselines3).

* ``PerformanceJumpPenaltyRamp``: sube la penalización por salto ``c`` sólo cuando el agente ya gana.
* ``BestModelCallback``: evalúa cada N pasos y guarda el mejor modelo (no sólo el último).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .common import EVAL_PHASES
from .evaluate import append_row, evaluate_policy_phases, metrics


class PerformanceJumpPenaltyRamp(BaseCallback):
    """Rampa por desempeño: ``c`` empieza en 0 y cambia según el éxito de los episodios de entrenamiento.

    Cada ``window`` episodios mira la tasa de éxito: si es ``>= up`` sube ``c`` en ``step`` (hasta ``c_max``),
    si es ``<= down`` lo baja en ``step`` (hasta 0), y si no, lo deja igual. Tras cada cambio vacía la ventana,
    así la próxima decisión usa episodios jugados con el ``c`` nuevo.
    Cada cambio de ``c`` se registra en ``log_path`` (CSV): paso, ``c`` y la tasa de éxito que lo motivó.
    """

    def __init__(self, c_max: float, step: float = 0.01, window: int = 20, up: float = 0.8, down: float = 0.4,
                 log_path: str | Path | None = None):
        super().__init__()
        self.c_max, self.step, self.up, self.down = c_max, step, up, down
        self.log_path = log_path
        self.c = 0.0
        self.results: deque[float] = deque(maxlen=window)
        self.last_rate: float | None = None

    def observe(self, success: bool) -> bool:
        """Registra un episodio terminado. Devuelve True si cambió ``c``."""
        self.results.append(float(success))
        if len(self.results) < self.results.maxlen:
            return False
        self.last_rate = float(np.mean(self.results))
        old = self.c
        if self.last_rate >= self.up:
            self.c = min(self.c + self.step, self.c_max)
        elif self.last_rate <= self.down:
            self.c = max(self.c - self.step, 0.0)
        if self.c == old:
            return False
        self.results.clear()
        return True

    def _apply(self, success_rate=""):
        # set_wrapper_attr atraviesa los wrappers (Monitor, RandomPhase) hasta GeometryDashEnv.
        # Ojo: VecEnv.set_attr NO lo hace: dejaría el valor en el Monitor, sin efecto.
        self.training_env.env_method("set_wrapper_attr", "jump_penalty", self.c)
        if self.log_path:
            append_row({"timesteps": self.num_timesteps, "jump_penalty": round(self.c, 6),
                        "success_rate": success_rate}, self.log_path)

    def _on_training_start(self):
        self._apply()                      # arranca con c = 0, sea cual sea el valor con que se creó el entorno

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info and self.observe(info["success"]):   # "episode" = terminó un episodio
                self._apply(success_rate=round(self.last_rate, 3))
        self.logger.record("train/jump_penalty", self.c)
        return True


def _score(m: dict) -> tuple:
    """Cuál modelo es mejor: más éxito, luego más progreso, luego menos saltos."""
    jumps = m["mean_jumps_won"]
    return m["success"], m["progress"], -jumps if jumps is not None else -np.inf


class BestModelCallback(BaseCallback):
    """Cada ``eval_freq`` pasos (y al terminar) evalúa el modelo en ``phases`` y guarda el mejor.

    Deja en ``save_dir``: ``best_model.zip``, ``best_model.json`` (paso y métricas del mejor) y
    ``eval_history.csv`` (todas las evaluaciones, con ``is_best`` marcando cuándo mejoró el récord).
    ``env_kwargs`` va al entorno de evaluación.
    """

    def __init__(self, eval_freq: int, save_dir: str | Path, obs_mode: str = "features", phases=EVAL_PHASES,
                 **env_kwargs):
        super().__init__()
        self.eval_freq, self.save_dir, self.obs_mode, self.phases = eval_freq, Path(save_dir), obs_mode, phases
        self.env_kwargs = env_kwargs
        self.best_score = None
        self._last_eval = 0

    def _evaluate(self):
        self._last_eval = self.num_timesteps
        m = metrics(evaluate_policy_phases(self.model, self.obs_mode, self.phases, **self.env_kwargs))
        is_best = self.best_score is None or _score(m) > self.best_score
        info = {"timesteps": self.num_timesteps, **m,
                "train_jump_penalty": self.training_env.get_attr("jump_penalty", indices=[0])[0]}
        append_row({**info, "is_best": int(is_best)}, self.save_dir / "eval_history.csv")
        if is_best:
            self.best_score = _score(m)
            self.model.save(self.save_dir / "best_model")
            (self.save_dir / "best_model.json").write_text(json.dumps(info, indent=2))
        self.logger.record("eval/success", m["success"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last_eval:     # que el modelo final también compita
            self._evaluate()

"""Callbacks de entrenamiento (Stable-Baselines3).

* ``PerformanceJumpPenaltyRamp``: sube la penalización por salto ``c`` sólo cuando el agente ya gana.
* ``LevelProgression``: introduce los niveles de a poco, por etapas, y reparte los episodios según el desempeño.
* ``BestModelCallback``: evalúa cada N pasos y guarda el mejor modelo (no sólo el último).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .common import EVAL_PHASES
from .evaluate import append_row, evaluate_policy_phases, metrics, success_by_level


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
            return False                      # todavía no hay una ventana completa de episodios para decidir
        self.last_rate = float(np.mean(self.results))
        old = self.c
        if self.last_rate >= self.up:
            self.c = min(self.c + self.step, self.c_max)
        elif self.last_rate <= self.down:
            self.c = max(self.c - self.step, 0.0)
        if self.c == old:
            return False                      # zona neutra o tope alcanzado: la ventana sigue deslizando
        self.results.clear()
        return True

    def _apply(self, success_rate=""):
        # set_wrapper_attr atraviesa los wrappers (Monitor, RandomPhase) hasta GeometryDashEnv.
        # Ojo: VecEnv.set_attr NO lo hace: dejaría el valor en el Monitor, sin efecto.
        self.training_env.env_method("set_wrapper_attr", "jump_penalty", self.c)
        if self.log_path:
            append_row({"timesteps": self.num_timesteps, "jump_penalty": round(self.c, 6),
                        "success_rate": success_rate}, self.log_path)

    def _on_training_start(self):                # hook de SB3: corre una vez al empezar ``model.learn``
        self._apply()                      # arranca con c = 0, sea cual sea el valor con que se creó el entorno

    def _on_step(self) -> bool:                  # hook de SB3: corre después de cada paso de los entornos
        for info in self.locals["infos"]:
            if "episode" in info and self.observe(info["success"]):   # "episode" = terminó un episodio
                self._apply(success_rate=round(self.last_rate, 3))
        self.logger.record("train/jump_penalty", self.c)
        return True


class LevelProgression(BaseCallback):
    """Progresión de niveles (en la literatura, *curriculum learning*): los niveles se introducen de a poco, por etapas.

    Qué hace, paso a paso:

    1. Arranca con sólo la primera etapa activa; los niveles de las etapas siguientes tienen peso 0 (no se juegan).
    2. Cada vez que termina un episodio de entrenamiento, anota si ganó en una ventana con los últimos ``window``
       episodios de *ese* nivel (el nivel viene en ``info["level"]``).
    3. Recalcula el peso de cada nivel (la probabilidad relativa de que salga en el próximo episodio):
         * bloqueado (etapa todavía no alcanzada): 0;
         * activo: ``peso_base * max(floor, 1 - éxito_reciente)``. Se practica más lo que más falla. Un nivel recién
           desbloqueado (sin episodios) cuenta como éxito 0, así que entra con el peso máximo. Uno ya dominado baja
           hasta ``floor`` pero nunca a 0: se lo sigue repasando. Si el agente lo empieza a olvidar, su éxito baja y su
           peso vuelve a subir solo (así se evita el *olvido catastrófico*).
    4. Cuando **todos** los niveles activos tienen la ventana llena con éxito >= ``unlock``, desbloquea la etapa siguiente.
    5. Pasa los pesos a los entornos en paralelo (``set_wrapper_attr``, igual que la rampa de la penalización por
       salto) y registra en ``log_path`` (CSV) cada cambio de etapa y, cada ``window`` episodios, los pesos y el éxito
       de cada nivel.

    El éxito es el de los episodios de entrenamiento (con exploración), por eso el umbral por defecto es 0.8 y no 1.
    ``levels`` y ``base_weights`` van en el mismo orden que los niveles del entorno (``GeometryDashEnv.levels``).
    """

    def __init__(self, levels: list[str], base_weights: list[float], stages: list[list[str]], unlock: float = 0.8,
                 window: int = 50, floor: float = 0.2, log_path: str | Path | None = None):
        super().__init__()
        self.levels, self.base = list(levels), list(base_weights)
        self.stage_of = {name: i for i, stage in enumerate(stages) for name in stage}
        assert sorted(self.stage_of) == sorted(self.levels), "cada nivel tiene que estar en exactamente una etapa"
        self.n_stages, self.unlock, self.floor, self.log_path = len(stages), unlock, floor, log_path
        self.window = window
        self.stage = 0
        self.results = {name: deque(maxlen=window) for name in self.levels}   # últimos episodios de cada nivel
        self.episodes = 0

    def active(self) -> list[str]:
        return [n for n in self.levels if self.stage_of[n] <= self.stage]

    def success(self, name: str) -> float:
        """Éxito reciente del nivel (0 si todavía no se jugó)."""
        return float(np.mean(self.results[name])) if self.results[name] else 0.0

    def weights(self) -> list[float]:
        """Peso de cada nivel, en el orden de ``levels`` (paso 3 de arriba)."""
        return [b * max(self.floor, 1.0 - self.success(n)) if self.stage_of[n] <= self.stage else 0.0
                for n, b in zip(self.levels, self.base)]

    def observe(self, name: str, success: bool) -> bool:
        """Registra un episodio terminado. Devuelve True si se desbloqueó una etapa nueva (paso 4)."""
        self.results[name].append(float(success))
        self.episodes += 1
        ready = all(len(self.results[n]) == self.window and self.success(n) >= self.unlock for n in self.active())
        if ready and self.stage < self.n_stages - 1:
            self.stage += 1
            return True
        return False

    def _apply(self, event: str):
        """Pasa los pesos actuales a los entornos y anota una fila en el CSV."""
        self.training_env.env_method("set_wrapper_attr", "level_weights", self.weights())
        if self.log_path:
            append_row({"timesteps": self.num_timesteps, "event": event, "stage": self.stage,
                        **{f"w_{n}": round(w, 4) for n, w in zip(self.levels, self.weights())},
                        **{f"success_{n}": round(self.success(n), 3) for n in self.levels}}, self.log_path)

    def _on_training_start(self):
        self._apply("inicio")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:                    # sólo cuentan los episodios terminados
                continue
            if self.observe(info["level"], info["success"]):
                self._apply(f"etapa {self.stage + 1}/{self.n_stages}")           # pesos nuevos + registro
            elif self.episodes % self.window == 0:
                self._apply("pesos")                                             # registro periódico
            else:
                self.training_env.env_method("set_wrapper_attr", "level_weights", self.weights())
        self.logger.record("train/etapa", self.stage)
        return True


def _score(m: dict) -> tuple:
    """Cuál modelo es mejor: más éxito, luego más progreso, luego menos saltos."""
    jumps = m["mean_jumps_won"]
    # Tuplas se comparan en orden: primero éxito, después progreso, y por último menos saltos (de ahí el signo menos).
    return m["success"], m["progress"], -jumps if jumps is not None else -np.inf


class BestModelCallback(BaseCallback):
    """Cada ``eval_freq`` pasos (y al terminar) evalúa el modelo en ``phases`` y guarda el mejor.

    Deja en ``save_dir``: ``best_model.zip``, ``best_model.json`` (paso y métricas del mejor) y
    ``eval_history.csv`` (todas las evaluaciones, con ``is_best`` marcando cuándo mejoró el récord).
    ``env_kwargs`` va al entorno de evaluación (niveles de entrenamiento + recompensa): con eso se elige el mejor.
    ``val_env_kwargs`` (opcional): niveles de validación. Se evalúan y se anotan en el historial (columnas ``val_*``),
    pero **no** cuentan para elegir el mejor modelo.
    """

    def __init__(self, eval_freq: int, save_dir: str | Path, obs_mode: str = "features", phases=EVAL_PHASES,
                 val_env_kwargs: dict | None = None, **env_kwargs):
        super().__init__()
        # eval_freq en pasos del entrenamiento; obs_mode debe coincidir con el del entorno de entrenamiento.
        self.eval_freq, self.save_dir, self.obs_mode, self.phases = eval_freq, Path(save_dir), obs_mode, phases
        self.env_kwargs, self.val_env_kwargs = env_kwargs, val_env_kwargs
        self.best_score = None
        self._last_eval = 0

    def _evaluate(self):
        """Evalúa el modelo actual en las fases, lo registra en el historial y, si es el mejor hasta ahora, lo guarda."""
        self._last_eval = self.num_timesteps
        rows = evaluate_policy_phases(self.model, self.obs_mode, self.phases, **self.env_kwargs)
        m = metrics(rows)
        is_best = self.best_score is None or _score(m) > self.best_score
        info = {"timesteps": self.num_timesteps, **m, **success_by_level(rows),
                "train_jump_penalty": self.training_env.get_attr("jump_penalty", indices=[0])[0]}
        if self.val_env_kwargs is not None:          # validación: sólo se registra
            vrows = evaluate_policy_phases(self.model, self.obs_mode, self.phases, **self.val_env_kwargs)
            info.update({f"val_{k}": v for k, v in {**metrics(vrows), **success_by_level(vrows)}.items()})
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

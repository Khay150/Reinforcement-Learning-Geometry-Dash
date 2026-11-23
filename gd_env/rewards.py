"""Recompensa del entorno: la que viene por defecto y cómo enchufar una propia.

Una recompensa propia es una función que recibe un ``Transition`` (lo que pasó en este tick) y devuelve un float::

    from gd_env import Transition, default_reward

    def mi_recompensa(tr: Transition) -> float:
        r = default_reward(tr, death_penalty=5.0)      # se puede partir de la de por defecto...
        if tr.jumped:
            r -= 0.02                                  # ...y agregarle lo que haga falta
        return r

    env = GeometryDashEnv(reward_fn=mi_recompensa)                  # directo en el entorno
    train("ppo", reward_fn=mi_recompensa)                           # o al entrenar (agents.train)
    python scripts/train.py --reward-fn mi_archivo.py:mi_recompensa     # o desde la línea de comandos

También se acepta la firma vieja ``reward_fn(prev, new, status, level)``.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .physics import DEAD, WIN, Level, State


@dataclass(frozen=True)
class Transition:
    """Lo que pasó en un tick; es lo único que ve una ``reward_fn``."""
    prev: State          # estado antes del tick
    new: State           # estado después del tick
    status: str          # ok / crash / spike / fell / win
    level: Level
    jumped: bool         # True si en este tick hubo un salto efectivo (acción 1 con el cubo apoyado)

    @property
    def dead(self) -> bool:
        return self.status in DEAD

    @property
    def won(self) -> bool:
        return self.status == WIN

    @property
    def dx(self) -> float:
        """Distancia recorrida en el tick (en columnas)."""
        return self.new.x - self.prev.x

    @property
    def d_progress(self) -> float:
        """Fracción del nivel recorrida en este tick (sumada en todo el nivel da ~1)."""
        return self.dx / (self.level.goal_x - self.level.start_x)

    @property
    def progress(self) -> float:
        """Fracción del nivel recorrida hasta ahora, en [0, 1]."""
        span = self.level.goal_x - self.level.start_x
        return float(np.clip((self.new.x - self.level.start_x) / span, 0, 1))


# Una recompensa propia: Transition -> float (o la firma vieja: prev, new, status, level -> float).
RewardFn = Callable[[Transition], float]


def default_reward(tr: Transition, *, progress_scale: float = 1.0, win_bonus: float = 1.0,
                   death_penalty: float = 10.0, survive_bonus: float = 0.0) -> float:
    """Recompensa por defecto: avanzar (vale ``progress_scale`` cruzar todo el nivel) + ``win_bonus`` al ganar
    - ``death_penalty`` al morir + ``survive_bonus`` por tick vivo. La penalización por salto la aplica el entorno aparte."""
    r = progress_scale * tr.d_progress + survive_bonus
    if tr.dead:
        r -= death_penalty
    elif tr.won:
        r += win_bonus
    return r


def adapt_reward_fn(fn: Callable | None) -> RewardFn | None:
    """Acepta ``fn(tr)`` o la firma vieja ``fn(prev, new, status, level)`` y devuelve siempre algo de la forma ``fn(tr)``."""
    if fn is None:
        return None
    try:
        n = sum(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):       # callables sin firma inspeccionable: se asume la nueva
        n = 1
    if n == 4:
        return lambda tr: fn(tr.prev, tr.new, tr.status, tr.level)
    if n == 1:
        return fn
    raise TypeError("reward_fn tiene que recibir un argumento (Transition) o cuatro (prev, new, status, level)")


# ---------------------------------------------------------------- por nombre (CLI y config.json)
def reward_spec(fn: Callable | None) -> str | None:
    """Texto ``modulo:funcion`` (o ``archivo.py:funcion``) que identifica la función; es lo que se guarda en config.json."""
    if fn is None:
        return None
    if isinstance(fn, str):
        return fn
    if getattr(fn, "_gd_spec", None):          # cargada con load_reward_fn: se guarda el texto original
        return fn._gd_spec
    return f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', repr(fn))}"


def load_reward_fn(spec: str | Callable | None) -> Callable | None:
    """Carga una recompensa desde ``'paquete.modulo:funcion'`` o ``'ruta/archivo.py:funcion'``."""
    if spec is None or callable(spec):
        return spec
    mod_name, _, attr = spec.rpartition(":")
    if not mod_name or not attr:
        raise ValueError(f"reward_fn debe ser 'modulo:funcion' o 'archivo.py:funcion' (recibí {spec!r})")
    if mod_name.endswith(".py"):
        path = Path(mod_name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no existe {path}")
        s = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(s)
        s.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_name)
    try:
        fn = getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{mod_name} no tiene una función {attr!r}") from None
    try:
        fn._gd_spec = spec                     # para que config.json guarde 'archivo.py:funcion' y se pueda recargar
    except AttributeError:
        pass
    return fn

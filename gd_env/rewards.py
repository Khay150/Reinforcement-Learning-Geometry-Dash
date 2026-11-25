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
import math
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


# ---------------------------------------------------------------- términos de la recompensa (se prenden y apagan)
# Cada término es una pieza de la recompensa. Se eligen por entrenamiento con ``rewards=[...]`` (``--rewards`` en la CLI).
#   progress       avanzar: ``progress_scale`` por cruzar todo el nivel
#   win            ``win_bonus`` al llegar a la meta
#   death          ``-death_penalty`` al morir
#   survive        ``survive_bonus`` cada ``survive_interval`` ticks vivo (sólo si survive_bonus / survive_rate > 0)
#   jump_penalty   ``-jump_penalty`` en cada salto efectivo (lo resta el entorno; con ``--ramp`` va subiendo)
#   salto_preciso  (Santi) saltar sobre pinchos/pozos sale gratis y premia según cuántos cubre; saltar sin obstáculo castiga
# ``jump_penalty`` y ``salto_preciso`` son excluyentes: los dos regulan el costo de saltar y se pisarían.
REWARD_TERMS = ("progress", "win", "death", "survive", "jump_penalty", "salto_preciso")
DEFAULT_TERMS = ("progress", "win", "death", "survive", "jump_penalty")      # la recompensa de siempre
EXCLUSIVE_TERMS = ("jump_penalty", "salto_preciso")
# Parámetros que sólo tienen sentido con su término prendido (``train`` avisa si se pasan con el término apagado).
TERM_PARAMS = {"progress": ("progress_scale",), "win": ("win_bonus",), "death": ("death_penalty",),
               "survive": ("survive_bonus", "survive_interval", "survive_rate"), "jump_penalty": ("jump_penalty", "ramp"),
               "salto_preciso": ()}

JUMP_SPAN = 3.2   # largo horizontal de un salto en el eje X (16 ticks * 0.2 tiles/tick)


def validate_terms(terms) -> tuple[str, ...]:
    """Normaliza la lista de términos (None = ``DEFAULT_TERMS``) y falla con un mensaje claro si hay uno desconocido,
    si no queda ninguno o si están prendidos a la vez ``jump_penalty`` y ``salto_preciso``."""
    if terms is None:
        return DEFAULT_TERMS
    terms = tuple(dict.fromkeys([terms] if isinstance(terms, str) else terms))      # sin repetidos, mismo orden
    unknown = [t for t in terms if t not in REWARD_TERMS]
    if unknown:
        raise ValueError(f"término(s) de recompensa desconocido(s): {', '.join(map(repr, unknown))} "
                         f"(disponibles: {', '.join(REWARD_TERMS)})")
    if not terms:
        raise ValueError("rewards está vacío: tiene que haber al menos un término de recompensa prendido")
    if all(t in terms for t in EXCLUSIVE_TERMS):
        raise ValueError("'jump_penalty' y 'salto_preciso' son excluyentes (los dos fijan cuánto cuesta saltar): "
                         "elegí uno. 'salto_preciso' ya castiga el salto sin sentido y deja gratis el que sirve")
    return terms


def salto_preciso_term(tr: Transition, *, penalty: float = 0.06, bonus_per_hazard: float = 0.03) -> float:
    """Término de Santi: sólo el *ajuste* por saltar (no incluye progreso, meta ni muerte). Es 0 si no hubo salto.

    Mira los ``JUMP_SPAN`` = 3.2 bloques que cubre el salto desde donde despega el cubo:
    * sin ningún pincho, pozo ni escalón en ese tramo: salto innecesario -> ``-penalty``;
    * con 2 o más peligros (pinchos o celdas de pozo, p. ej. ``^^`` o un pozo ancho): ``+bonus_per_hazard`` por cada uno;
    * con 1 peligro o sólo un escalón: gratis (0).
    """
    if not tr.jumped:
        return 0.0
    lv = tr.level
    ground_h = round(tr.prev.y)
    c_start = int(math.floor(tr.prev.x))
    c_end = min(int(math.floor(tr.prev.x + JUMP_SPAN)) + 1, lv.length)
    hazards, step = 0, False
    for c in range(c_start, c_end):
        top = lv.column_top(c)
        if lv.has_spike(c) or top == 0:       # pincho o pozo
            hazards += 1
        elif top != ground_h:                 # subida o bajada de bloque
            step = True
    if hazards == 0 and not step:
        return -penalty
    if hazards >= 2:
        return bonus_per_hazard * hazards
    return 0.0


def reward_terms(tr: Transition, terms=DEFAULT_TERMS, *, progress_scale: float = 1.0, win_bonus: float = 1.0,
                 death_penalty: float = 10.0, survive_bonus: float = 0.0, survive_interval: int = 1) -> float:
    """Suma de los términos prendidos en ``terms`` para este tick. ``jump_penalty`` no se calcula acá: lo resta el
    entorno (``GeometryDashEnv.step``) porque su valor cambia durante el entrenamiento con la rampa."""
    r = 0.0
    if "progress" in terms:
        r += progress_scale * tr.d_progress
    if "survive" in terms and survive_bonus and not tr.dead and tr.new.t % max(int(survive_interval), 1) == 0:
        r += survive_bonus
    if tr.dead:
        if "death" in terms:
            r -= death_penalty
    elif tr.won and "win" in terms:
        r += win_bonus
    if "salto_preciso" in terms:
        r += salto_preciso_term(tr)
    return r


def default_reward(tr: Transition, *, progress_scale: float = 1.0, win_bonus: float = 1.0,
                   death_penalty: float = 10.0, survive_bonus: float = 0.0, survive_interval: int = 1) -> float:
    """Recompensa por defecto: avanzar (vale ``progress_scale`` cruzar todo el nivel) + ``win_bonus`` al ganar
    - ``death_penalty`` al morir + ``survive_bonus`` cada ``survive_interval`` ticks vivo (1 = en todos los ticks).
    El bonus se da en los ticks t = interval, 2*interval, ... en los que el cubo sigue vivo (no en el tick en que muere).
    La penalización por salto la aplica el entorno aparte."""
    return reward_terms(tr, ("progress", "survive", "death", "win"), progress_scale=progress_scale, win_bonus=win_bonus,
                        death_penalty=death_penalty, survive_bonus=survive_bonus, survive_interval=survive_interval)


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

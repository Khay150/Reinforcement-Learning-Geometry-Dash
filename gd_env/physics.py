"""Física determinista del "cubo" de Geometry Dash, como funciones puras.

Todo el simulador vive acá (sin Gymnasium) para que el entorno y el oráculo
usen *exactamente* la misma dinámica.

Convenciones
------------
* Unidad de longitud = 1 tile. El cubo mide 1x1.
* Un tick = un paso de simulación. El cubo avanza a velocidad horizontal
  constante ``vx`` tiles/tick (como en el juego: no se controla).
* ``y`` = borde inferior del cubo. El piso es la fila 0 del nivel (su cara
  superior está en y = 1). Un pozo es una columna sin tile en la fila 0.
* El cubo avanza SIEMPRE a ``vx`` y no hay acción para frenarlo: x depende sólo
  del tick. Únicamente se detiene al morir (choque, pincho o caída), y ese
  estado es terminal: queda congelado donde murió. Ganar también termina el episodio.
* Única acción: saltar (1) / no hacer nada (0). El salto sólo tiene efecto si
  el cubo está apoyado. Equivale a "mantener apretado" en el juego: al caer
  vuelve a saltar solo si seguís pidiendo 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

EMPTY, BLOCK, SPIKE = 0, 1, 2

# Estados terminales posibles de un tick.
OK, CRASH, SPIKE_HIT, FELL, WIN = "ok", "crash", "spike", "fell", "win"
DEAD = (CRASH, SPIKE_HIT, FELL)


@dataclass(frozen=True)
class Physics:
    vx: float = 0.2          # tiles/tick (avance horizontal constante)
    gravity: float = 0.05    # tiles/tick^2
    jump_v: float = 0.4      # velocidad vertical inicial del salto
    max_fall: float = 0.6    # velocidad de caída máxima (< 1 evita atravesar tiles)
    spike_w: float = 0.3     # ancho de la hitbox del pincho (centrada en el tile)
    spike_h: float = 0.3     # alto de la hitbox del pincho
    eps: float = 1e-9


class State(NamedTuple):
    t: int            # ticks transcurridos
    x0: float         # x inicial (x(t) = x0 + vx * t, no hay deriva numérica)
    x: float          # borde izquierdo del cubo
    y: float          # borde inferior del cubo
    vy: float
    on_ground: bool

    def key(self) -> tuple:
        """Clave hashable para deduplicar estados en la búsqueda del oráculo."""
        return (self.t, round(self.y, 7), round(self.vy, 7), self.on_ground)


class Level:
    """Nivel fijo: grilla ``grid[fila, col]`` con fila 0 = piso (abajo)."""

    def __init__(self, grid: np.ndarray, goal_x: float, start_x: float = 2.0, name: str = "level"):
        self.grid = np.asarray(grid, dtype=np.uint8)
        self.height, self.length = self.grid.shape
        self.goal_x = float(goal_x)
        self.start_x = float(start_x)
        self.name = name
        self._spikes = {(int(r), int(c)) for r, c in zip(*np.nonzero(self.grid == SPIKE))}

    def tile(self, row: int, col: int) -> int:
        if 0 <= row < self.height and 0 <= col < self.length:
            return int(self.grid[row, col])
        return EMPTY

    def column_top(self, col: int) -> int:
        """Altura (en tiles) del bloque sólido más alto de la columna; 0 si no hay."""
        if not 0 <= col < self.length:
            return 0
        rows = np.nonzero(self.grid[:, col] == BLOCK)[0]
        return int(rows.max()) + 1 if rows.size else 0

    def has_spike(self, col: int) -> bool:
        return 0 <= col < self.length and bool((self.grid[:, col] == SPIKE).any())


def initial_state(level: Level, p: Physics, x0: float | None = None) -> State:
    x0 = level.start_x if x0 is None else float(x0)
    return State(t=0, x0=x0, x=x0, y=1.0, vy=0.0, on_ground=True)


def _solid_tiles(level: Level, p: Physics, x: float, y: float) -> list[tuple[int, int]]:
    """(fila, col) de los bloques que se solapan (área > 0) con el cubo en (x, y)."""
    e = p.eps
    c0, c1 = math.floor(x + e), math.floor(x + 1 - e)
    r0, r1 = math.floor(y + e), math.floor(y + 1 - e)
    out = []
    for r in range(max(r0, 0), r1 + 1):
        for c in range(c0, c1 + 1):
            if level.tile(r, c) == BLOCK:
                out.append((r, c))
    return out


def _spike_hit(level: Level, p: Physics, x: float, y: float) -> bool:
    e = p.eps
    lo, hi = (1 - p.spike_w) / 2, (1 + p.spike_w) / 2
    for c in range(math.floor(x), math.floor(x + 1) + 1):
        for r in range(max(math.floor(y), 0), math.floor(y + 1) + 1):
            if (r, c) in level._spikes:
                if x < c + hi - e and x + 1 > c + lo + e and y < r + p.spike_h - e and y + 1 > r + e:
                    return True
    return False


def step(level: Level, p: Physics, s: State, action: int) -> tuple[State, str]:
    """Avanza un tick. Devuelve (nuevo_estado, status)."""
    vy = p.jump_v if (action == 1 and s.on_ground) else s.vy
    vy = max(vy - p.gravity, -p.max_fall)

    t = s.t + 1
    x = s.x0 + p.vx * t

    # 1) Avance horizontal: chocar de costado contra un bloque mata.
    hit_side = _solid_tiles(level, p, x, s.y)
    if hit_side:
        # El cubo se frena contra la cara del bloque (no se mete adentro).
        x_stop = max(s.x, min(c for _, c in hit_side) - 1.0)
        return State(t, s.x0, x_stop, s.y, 0.0, s.on_ground), CRASH

    # 2) Movimiento vertical y resolución de colisiones.
    y, on_ground = s.y + vy, False
    hit = _solid_tiles(level, p, x, y)
    if hit:
        if vy <= 0:                                   # aterriza sobre el tile más alto
            y = max(r + 1 for r, _ in hit)
            on_ground = True
        else:                                         # golpea el techo
            y = min(r for r, _ in hit) - 1.0
        vy = 0.0

    new = State(t, s.x0, x, y, vy, on_ground)
    if _spike_hit(level, p, x, y):
        return new, SPIKE_HIT
    if y < -1.0:
        return new, FELL
    if x >= level.goal_x:
        return new, WIN
    return new, OK

"""Nivel fijo del TP, definido de forma determinista y exportable a ASCII.

Leyenda: '#' bloque sólido, '^' pincho, '.' vacío. La última línea es el piso
(fila 0); un '.' en esa línea es un pozo.
"""
from __future__ import annotations

import numpy as np

from .physics import BLOCK, EMPTY, SPIKE, Level

_CHARS = {".": EMPTY, "#": BLOCK, "^": SPIKE}


def parse_level(ascii_map: str, start_x: float = 2.0, goal_margin: int = 4, name: str = "ascii") -> Level:
    rows = [r for r in ascii_map.strip("\n").splitlines()]
    width = max(len(r) for r in rows)
    grid = np.zeros((len(rows), width), dtype=np.uint8)
    for i, line in enumerate(rows):
        for j, ch in enumerate(line):
            grid[len(rows) - 1 - i, j] = _CHARS[ch]       # fila 0 = última línea
    return Level(grid, goal_x=width - goal_margin, start_x=start_x, name=name)


def to_ascii(level: Level) -> str:
    inv = {v: k for k, v in _CHARS.items()}
    return "\n".join("".join(inv[int(v)] for v in level.grid[r]) for r in range(level.height - 1, -1, -1))


def build_level(length: int, height: int, spikes=(), blocks=(), pits=(), name: str = "built") -> Level:
    """spikes: columnas; blocks: (col, alto_en_tiles) con alto >=1 apilado desde el piso;
    pits: (col_inicio, ancho)."""
    g = np.zeros((height, length), dtype=np.uint8)
    g[0, :] = BLOCK
    for c, w in pits:
        g[0, c:c + w] = EMPTY
    for c, h in blocks:
        g[1:1 + h, c] = BLOCK
    for c in spikes:
        g[1, c] = SPIKE
    return Level(g, goal_x=length - 4, start_x=2.0, name=name)


def level_1() -> Level:
    """Nivel 1: piso plano con pincho simple, doble, escalón, pozo, escalera y combos."""
    return build_level(
        length=110, height=7, name="level_1",
        spikes=[14, 24, 25, 40, 66, 67, 86, 96],
        blocks=[
            (34, 1), (35, 1), (36, 1),                      # escalón de 1
            (52, 1), (53, 1), (54, 2), (55, 2), (56, 2),    # escalera 1 -> 2
            (75, 1), (76, 1), (77, 1),                      # plataforma baja
            (101, 1),                                       # último bloque
        ],
        pits=[(46, 2), (80, 2)],
    )

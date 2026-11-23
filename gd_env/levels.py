"""Niveles del TP (deterministas, exportables a ASCII) y conjuntos de niveles para entrenar.

Leyenda: '#' bloque sólido, '^' pincho, '.' vacío. La última línea es el piso
(fila 0); un '.' en esa línea es un pozo.

Para entrenar con varios niveles, ``resolve_levels`` traduce lo que se pasa a ``--levels`` (nombres de ``LEVELS``
y/o un archivo ``.json``) en los niveles de entrenamiento con su peso y los de validación. Formato del ``.json``::

    {"niveles": {"level_1": 1.0, "level_2": 2.0},     # o una lista de nombres (todos con peso 1)
     "validacion": ["level_3"]}                        # nombre(s), un número (cuántos sortear) o null
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .physics import BLOCK, EMPTY, SPIKE, Level

_CHARS = {".": EMPTY, "#": BLOCK, "^": SPIKE}      # carácter ASCII -> tipo de tile


def parse_level(ascii_map: str, start_x: float = 2.0, goal_margin: int = 4, name: str = "ascii") -> Level:
    """Arma un ``Level`` desde un dibujo ASCII. La meta queda ``goal_margin`` columnas antes del final del mapa."""
    rows = [r for r in ascii_map.strip("\n").splitlines()]
    width = max(len(r) for r in rows)
    grid = np.zeros((len(rows), width), dtype=np.uint8)
    for i, line in enumerate(rows):
        for j, ch in enumerate(line):
            grid[len(rows) - 1 - i, j] = _CHARS[ch]       # fila 0 = última línea
    return Level(grid, goal_x=width - goal_margin, start_x=start_x, name=name)


def to_ascii(level: Level) -> str:
    """Inversa de ``parse_level``: dibuja el nivel en texto (útil para imprimirlo o guardarlo)."""
    inv = {v: k for k, v in _CHARS.items()}
    return "\n".join("".join(inv[int(v)] for v in level.grid[r]) for r in range(level.height - 1, -1, -1))


def build_level(length: int, height: int, spikes=(), blocks=(), pits=(), name: str = "built") -> Level:
    """spikes: columnas; blocks: (col, alto_en_tiles) con alto >=1 apilado desde el piso;
    pits: (col_inicio, ancho)."""
    g = np.zeros((height, length), dtype=np.uint8)
    g[0, :] = BLOCK                       # piso continuo...
    for c, w in pits:
        g[0, c:c + w] = EMPTY             # ...con pozos donde se indica
    for c, h in blocks:
        g[1:1 + h, c] = BLOCK             # columnas de bloques apiladas sobre el piso
    for c in spikes:
        g[1, c] = SPIKE                   # pinchos apoyados sobre el piso
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


# Niveles disponibles por nombre (lo que se escribe en --levels o en el .json de niveles).
# Todos tienen que tener la misma altura (la forma de la observación depende de ella) y arrancar en x = 2.0.
LEVELS = {
    "level_1": level_1,
}


def get_level(name: str) -> Level:
    """Construye un nivel registrado en ``LEVELS``."""
    try:
        return LEVELS[name]()
    except KeyError:
        raise ValueError(f"nivel desconocido: {name!r} (disponibles: {', '.join(LEVELS)})") from None


def read_level_file(path: str | Path) -> tuple[dict[str, float], object]:
    """Lee un .json de niveles. Devuelve ({nombre: peso}, validacion). Las claves que empiezan con "_" se ignoran."""
    data = json.loads(Path(path).read_text())
    levels = data["niveles"]
    if isinstance(levels, list):
        levels = {name: 1.0 for name in levels}
    return {k: float(v) for k, v in levels.items()}, data.get("validacion")


def resolve_levels(spec: list[str] | str | None, val_seed: int = 0) -> dict:
    """Resuelve ``--levels`` en niveles de entrenamiento (con peso) y de validación.

    ``spec``: nombres de ``LEVELS`` y/o rutas a ``.json`` (se pueden mezclar; los nombres sueltos pesan 1).
    ``validacion`` del .json:
      * nombre o lista de nombres -> esos niveles quedan fuera del entrenamiento (held out);
      * un número n               -> se sortean n niveles de entrenamiento y se sacan del entrenamiento;
      * null (o sin .json)        -> no se reserva ninguno: se valida en uno de entrenamiento elegido al azar.
    El sorteo usa ``val_seed``, así es reproducible. Devuelve
    ``{"train": {nombre: peso}, "val": [nombres], "held_out": bool}``.
    """
    spec = ["level_1"] if not spec else [spec] if isinstance(spec, str) else list(spec)
    train, val = {}, None
    for item in spec:
        if item.endswith(".json"):
            weights, v = read_level_file(item)
            train.update(weights)
            if v is not None:
                if val is not None:
                    raise ValueError("más de un .json define 'validacion'")
                val = v
        else:
            train[item] = 1.0
    rng = np.random.default_rng(val_seed)
    if val is None:                                       # no reservado: uno de los de entrenamiento
        val_names, held_out = [str(rng.choice(sorted(train)))], False
    elif isinstance(val, int) and not isinstance(val, bool):
        if not 0 < val < len(train):
            raise ValueError(f"validacion={val}: tiene que quedar al menos un nivel de entrenamiento ({len(train)} niveles)")
        val_names, held_out = sorted(str(n) for n in rng.choice(sorted(train), size=val, replace=False)), True
    else:
        val_names, held_out = ([val] if isinstance(val, str) else list(val)), True
    if held_out:
        train = {k: w for k, w in train.items() if k not in val_names}
    for name in [*train, *val_names]:
        get_level(name)                                   # falla ya si algún nombre no existe
    if not train or sum(train.values()) <= 0:
        raise ValueError("no quedan niveles de entrenamiento con peso > 0")
    return {"train": train, "val": val_names, "held_out": held_out}

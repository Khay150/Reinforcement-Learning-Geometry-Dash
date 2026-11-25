"""Mide la dificultad de uno o varios niveles con la física, sin entrenar: precisión de cada salto, trampas, saltos
encadenados y probabilidad de ganar apretando al azar. Las definiciones están en gd_env/difficulty.py.

Ejemplos
--------
  python scripts/difficulty.py                          # todos los niveles registrados (gd_env.LEVELS)
  python scripts/difficulty.py level_1 --detalle        # + tabla salto por salto (en la fase por defecto, x0 = 2.0)
  python scripts/difficulty.py mi_nivel.txt --ascii     # un nivel dibujado en ASCII, sin registrarlo (ver gd_env/levels.py)
  python scripts/difficulty.py niveles/ejemplo.json     # los niveles de un .json (entrenamiento y validación)
  python scripts/difficulty.py level_1 --fases 1        # sólo la fase por defecto, no las 10 de la evaluación

Cada columna del resumen es el peor caso entre las fases de arranque (por defecto, las 10 de la evaluación).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importan gd_env y agents

import numpy as np

from gd_env import LEVELS, Level, difficulty_over_phases, get_level, level_difficulty, parse_level, to_ascii
from gd_env.levels import read_level_file

# Columnas del resumen: clave de difficulty_over_phases -> encabezado.
COLUMNS = {
    "level": "nivel",
    "solvable_phases": "fases con solución",
    "n_jumps": "saltos mínimos",
    "min_window": "ventana mín. (ticks)",
    "mean_window": "ventana media (ticks)",
    "jumps_with_traps": "saltos con trampas",
    "trap_ticks": "ticks trampa",
    "max_chain": "cadena más larga",
    "linked_jumps": "saltos encadenados",
    "log10_p_random": "log10 P(ganar al azar)",
}

LEGEND = """\
- ventana: ticks de despegue distintos que siguen ganando sin gastar saltos de más (1 tick = 0.2 tiles). Chica = salto preciso.
- trampas: despegues desde los que el cubo aterriza vivo pero el nivel ya no se puede ganar (el error se paga más adelante).
- cadena: saltos seguidos en los que cuándo despegás en uno cambia la ventana del siguiente. "saltos encadenados" = cuántos
  saltos forman parte de alguna cadena.
- log10 P(ganar al azar): -3 = gana uno de cada mil episodios apretando 50/50 en cada tick (cuánto cuesta explorar)."""


def eval_phases(level: Level, n: int) -> list[float]:
    """``n`` fases de arranque como las de la evaluación (``EVAL_PHASES`` de agents/common.py para start_x = 2.0)."""
    return [level.start_x] if n <= 1 else [float(x) for x in level.start_x + np.linspace(0.0, 0.95, n)]


def load_levels(items: list[str] | None) -> list[Level]:
    """Nombres de ``gd_env.LEVELS``, archivos ``.txt`` con un nivel en ASCII y/o ``.json`` de niveles."""
    levels = []
    for item in items or list(LEVELS):
        if item.endswith(".txt"):
            path = Path(item)
            levels.append(parse_level(path.read_text(), name=path.stem))
        elif item.endswith(".json"):
            weights, val = read_level_file(item)
            names = list(weights) + ([val] if isinstance(val, str) else val if isinstance(val, list) else [])
            levels += [get_level(n) for n in dict.fromkeys(names)]
        else:
            levels.append(get_level(item))
    return levels


def _fmt(v) -> str:
    return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)


def summary_table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS.values()) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(_fmt(r.get(k)) for k in COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def detail_table(level: Level) -> str:
    """Salto por salto, en la fase por defecto."""
    d = level_difficulty(level)
    if d is None:
        return f"{level.name}: no tiene solución desde x0 = {level.start_x:g}"
    lines = [f"### {level.name} (x0 = {d.x0:g}), saltos mínimos: {d.n_jumps}",
             "",
             "| # | despegue en x | ventana (ticks) | se puede despegar entre x | trampas (ticks) "
             "| encadenado con el siguiente | ventana del siguiente según cuándo despegues |",
             "|---|---|---|---|---|---|---|"]
    for i, j in enumerate(d.jumps, 1):
        nxt = "-" if not j.linked_next else f"{j.next_windows[0]}–{j.next_windows[1]} ticks"
        lines.append(f"| {i} | {j.x:.1f} | {j.window} | {j.x_from:.1f} – {j.x_to:.1f} | {j.traps} "
                     f"| {'sí' if j.linked_next else 'no'} | {nxt} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="*", metavar="NIVEL|ARCHIVO.txt|ARCHIVO.json",
                    help="niveles a medir (por defecto, todos los de gd_env.LEVELS)")
    ap.add_argument("--fases", type=int, default=10,
                    help="fases de arranque a evaluar (def. 10, las de la evaluación; 1 = sólo x0 = start_x)")
    ap.add_argument("--detalle", action="store_true", help="tabla salto por salto de cada nivel (fase por defecto)")
    ap.add_argument("--ascii", action="store_true", help="imprime cada nivel en ASCII")
    args = ap.parse_args()

    try:
        levels = load_levels(args.levels)
    except (ValueError, FileNotFoundError, KeyError) as e:
        sys.exit(f"error: {e}")

    if args.ascii:
        for lv in levels:
            print(f"{lv.name}:\n{to_ascii(lv)}\n")
    rows = [difficulty_over_phases(lv, eval_phases(lv, args.fases)) for lv in levels]
    print(summary_table(rows))
    print(f"\nPeor caso entre {args.fases} fase(s) de arranque.\n{LEGEND}")
    if args.detalle:
        for lv in levels:
            print("\n" + detail_table(lv))


if __name__ == "__main__":
    main()

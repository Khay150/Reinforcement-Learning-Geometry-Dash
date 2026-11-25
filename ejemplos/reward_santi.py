"""Recompensa de salto preciso (Santi) para GeometryDashEnv.

Mira la ventana horizontal de un salto en el eje X (3.2 bloques = 16 ticks * 0.2 tiles/tick):
- Si el cubo salta cuando no hay ningún obstáculo (ni pincho, ni pozo, ni cambio de altura de bloque)
  en el alcance del salto, penaliza el salto innecesario (-0.06).
- Si salta sobre obstáculos reales, el salto es gratis (no se cobra) y además otorga un bonus
  proporcional a la cantidad de columnas de peligro (pinchos seguidos ^^ / ^^^ o pozos anchos)
  que sobrevuela el salto.

Uso:
  python scripts/train.py --algos ppo --levels niveles/santi.json --reward-fn ejemplos/reward_santi.py:salto_preciso
"""
from __future__ import annotations

import math

from gd_env import Transition, default_reward

JUMP_SPAN = 3.2  # largo horizontal de un salto en el eje X (16 ticks * 0.2 tiles/tick)


def salto_preciso(tr: Transition) -> float:
    """No cobra los saltos sobre obstáculos, premia según cuántos bloques de peligro (pinchos/pozos)
    cubre el salto de 3.2 tiles, y castiga saltar en piso plano cuando no viene ningún obstáculo."""
    r = default_reward(tr, progress_scale=1.0, win_bonus=2.0, death_penalty=10.0)
    if not tr.jumped:
        return r

    lv = tr.level
    x0 = tr.prev.x
    ground_h = round(tr.prev.y)
    c_start = int(math.floor(x0))
    c_end = min(int(math.floor(x0 + JUMP_SPAN)) + 1, lv.length)

    peligros = 0       # cantidad de pinchos o celdas de pozo que sobrevuela este salto
    hay_escalon = False
    for c in range(c_start, c_end):
        top = lv.column_top(c)
        if lv.has_spike(c) or top == 0:   # pincho o pozo
            peligros += 1
        elif top != ground_h:             # subida o bajada de bloque
            hay_escalon = True

    if peligros == 0 and not hay_escalon:
        # Saltó en piso liso sin ningún obstáculo en todo el rango del salto (3.2 bloques)
        r -= 0.06
    elif peligros >= 2:
        # Salto ajustado sobre múltiples pinchos seguidos (^^ / ^^^) o pozo ancho: premia la precisión
        r += 0.03 * peligros
    return r

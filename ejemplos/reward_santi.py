"""Recompensa de salto preciso (Santi) para GeometryDashEnv.

Mira la ventana horizontal de un salto en el eje X (3.2 bloques = 16 ticks * 0.2 tiles/tick):
- Si el cubo salta cuando no hay ningún obstáculo (ni pincho, ni pozo, ni cambio de altura de bloque)
  en el alcance del salto, penaliza el salto innecesario (-0.06).
- Si salta sobre obstáculos reales, el salto es gratis (no se cobra) y además otorga un bonus
  proporcional a la cantidad de columnas de peligro (pinchos seguidos ^^ / ^^^ o pozos anchos)
  que sobrevuela el salto.

Ahora vive como un término de la recompensa (``salto_preciso`` en ``gd_env/rewards.py``), que se prende con ``--rewards``
y se puede combinar con los demás (progress, win, death, survive). Es excluyente con ``jump_penalty``.

Uso (recomendado, con el término):
  python scripts/train.py --algos ppo --levels niveles/santi.json --rewards progress win death salto_preciso

Esta función suelta (la versión original de Santi, con win_bonus=2) sigue sirviendo como ``--reward-fn``:
  python scripts/train.py --algos ppo --levels niveles/santi.json --reward-fn ejemplos/reward_santi.py:salto_preciso
"""
from __future__ import annotations

from gd_env import Transition, default_reward, salto_preciso_term


def salto_preciso(tr: Transition) -> float:
    """La de por defecto (win_bonus=2) más el término ``salto_preciso``. Como reemplaza a todo, no se combina con ``--rewards``."""
    return default_reward(tr, progress_scale=1.0, win_bonus=2.0, death_penalty=10.0) + salto_preciso_term(tr)

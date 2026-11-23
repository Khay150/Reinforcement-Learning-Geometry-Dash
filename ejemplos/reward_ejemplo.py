"""Ejemplos de recompensas propias para GeometryDashEnv.

Una recompensa es una función ``fn(tr) -> float`` donde ``tr`` (``gd_env.Transition``) tiene:
  tr.prev / tr.new   estados antes y después del tick (x, y, vy, on_ground, t)
  tr.status          "ok", "crash", "spike", "fell" o "win"   (tr.dead / tr.won son atajos)
  tr.jumped          hubo un salto efectivo en este tick
  tr.d_progress      fracción del nivel recorrida en el tick (sumada en todo el nivel da ~1)
  tr.progress        fracción del nivel recorrida hasta ahora, en [0, 1]
  tr.level           el nivel (por si hace falta mirar bloques o pinchos)

Uso:
  python scripts/train.py --algos ppo --reward-fn ejemplos/reward_ejemplo.py:penaliza_saltos
  from agents import train; from ejemplos.reward_ejemplo import penaliza_saltos; train("ppo", reward_fn=penaliza_saltos)
"""
from gd_env import Transition, default_reward


def penaliza_saltos(tr: Transition) -> float:
    """La de por defecto, con otros valores, más un costo por saltar (equivale a --jump-penalty 0.05)."""
    r = default_reward(tr, win_bonus=2.0, death_penalty=5.0)
    return r - 0.05 if tr.jumped else r


def desde_cero(tr: Transition) -> float:
    """Una recompensa totalmente propia, sin partir de la de por defecto."""
    if tr.won:
        return 10.0
    if tr.dead:
        return -5.0 - 5.0 * (1 - tr.progress)       # morir cerca de la meta duele menos
    return 10.0 * tr.d_progress

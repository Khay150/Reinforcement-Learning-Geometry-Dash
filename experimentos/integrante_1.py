"""Integrante 1: E2 con grilla (CNN) + la corrida base de PPO + E3 (penalización por salto fija vs. rampa) + 2 niveles de E7.

Correr desde cualquier lado (el script se para solo en la raíz del repo):
    python experimentos/integrante_1.py --dry-run     # ver qué va a correr y cuánto tarda
    python experimentos/integrante_1.py --subir       # entrenar y subir cada corrida al terminar (un commit por seed)
Se puede cortar y volver a lanzar: saltea lo que ya terminó. Ver experimentos/README.md.
"""
from comun import E7, N, Corrida, correr, e7

CORRIDAS = [
    # E2: misma configuración que base_ppo, cambiando la observación (grilla 3x7x16 + CNN)
    Corrida("e2_obs_grid", ["--algos", "ppo", "--obs", "grid", *N], minutos=51),
    # base de E1-E4: PPO con features. Es "PPO" en E1, "features" en E2, "sin castigo" en E3 y "fase fija" en E4
    Corrida("base_ppo", ["--algos", "ppo", *N], minutos=14),
    # E3: castigo por salto c = 0.1 fijo vs. rampa (arranca en 0 y sube hasta 0.1 sólo cuando ya gana)
    Corrida("e3_fijo", ["--algos", "ppo", *N, "--jump-penalty", "0.1"], minutos=14),
    Corrida("e3_rampa", ["--algos", "ppo", *N, "--jump-penalty", "0.1", "--ramp"], minutos=14),
    *e7(E7[0:2]),
]

if __name__ == "__main__":
    correr("Integrante 1: E2 grid, base PPO, E3 y E7 (2 niveles)", CORRIDAS)

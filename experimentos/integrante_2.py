"""Integrante 2: E2 con dict (CNN + vector) + E4 (fase aleatoria) + E5 (salto_preciso en los niveles de Santi) + 2 niveles de E7.

Correr desde cualquier lado (el script se para solo en la raíz del repo):
    python experimentos/integrante_2.py --dry-run     # ver qué va a correr y cuánto tarda
    python experimentos/integrante_2.py --subir       # entrenar y subir cada corrida al terminar (un commit por seed)
Se puede cortar y volver a lanzar: saltea lo que ya terminó. Ver experimentos/README.md.
"""
from comun import E7, N, Corrida, correr, e7

CORRIDAS = [
    # E2: como base_ppo (integrante 1) pero con observación dict (grilla + [y, vy, en_suelo, frac(x)])
    Corrida("e2_obs_dict", ["--algos", "ppo", "--obs", "dict", *N], minutos=51),
    # E4: como base_ppo pero con x0 sorteado en cada episodio (base_ppo = fase fija)
    Corrida("e4_random", ["--algos", "ppo", *N, "--train-phase", "random"], minutos=14),
    # E5: 5 niveles de Santi (validación level_1), castigo fijo por salto vs. la recompensa salto_preciso
    Corrida("e5_santi_jp", ["--algos", "ppo", *N, "--levels", "niveles/santi.json", "--jump-penalty", "0.05"],
            minutos=14),
    Corrida("e5_santi_salto_preciso", ["--algos", "ppo", *N, "--levels", "niveles/santi.json",
                                       "--rewards", "progress", "win", "death", "salto_preciso"], minutos=14),
    *e7(E7[2:4]),
]

if __name__ == "__main__":
    correr("Integrante 2: E2 dict, E4, E5 y E7 (2 niveles)", CORRIDAS)

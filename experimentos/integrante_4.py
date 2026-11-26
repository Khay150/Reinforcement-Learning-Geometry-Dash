"""Integrante 4: E8, generalización. PPO con observación dict entrenado en 20 niveles y validado en 3 que nunca ve
(santi_facil, level_1 y nacho_dificil: uno fácil, uno medio y uno difícil). Ver niveles/generalizacion.json.

Es dict y no features porque entre los 20 hay niveles con techo o bloques flotantes que features no puede representar
(muestra sólo el bloque más alto de cada columna: un techo se ve como una pared).

Correr desde cualquier lado (el script se para solo en la raíz del repo):
    python experimentos/integrante_4.py --dry-run     # ver qué va a correr y cuánto tarda
    python experimentos/integrante_4.py --subir       # entrenar y subir cada corrida al terminar (un commit por seed)
Se puede cortar y volver a lanzar: saltea lo que ya terminó. Ver experimentos/README.md.
"""
from comun import Corrida, correr

CORRIDAS = [
    # 600k pasos = ~30k por nivel de entrenamiento; castigo por salto 0.05 como en E5 y E6
    Corrida("e8_generalizacion", ["--algos", "ppo", "--obs", "dict", "--seeds", "0", "1", "2", "--timesteps", "600000",
                                  "--levels", "niveles/generalizacion.json", "--jump-penalty", "0.05"], minutos=135),
]

if __name__ == "__main__":
    correr("Integrante 4: E8 generalización", CORRIDAS)

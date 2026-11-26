"""Integrante 3: E1 (QR-DQN; PPO es base_ppo del integrante 1) + los 8 niveles más difíciles de E7.

QR-DQN necesita sb3-contrib (pip install -r requirements.txt); el script lo chequea antes de empezar.
Correr desde cualquier lado (el script se para solo en la raíz del repo):
    python experimentos/integrante_3.py --dry-run     # ver qué va a correr y cuánto tarda
    python experimentos/integrante_3.py --subir       # entrenar y subir cada corrida al terminar (un commit por seed)
Se puede cortar y volver a lanzar: saltea lo que ya terminó. Ver experimentos/README.md.
"""
from comun import E7, N, Corrida, correr, e7

CORRIDAS = [
    # E1: misma configuración que base_ppo (PPO), cambiando el algoritmo
    Corrida("e1_qrdqn", ["--algos", "qrdqn", *N], minutos=27),
    *e7(E7[4:]),
]

if __name__ == "__main__":
    correr("Integrante 3: E1 (QR-DQN) y E7 (8 niveles)", CORRIDAS)

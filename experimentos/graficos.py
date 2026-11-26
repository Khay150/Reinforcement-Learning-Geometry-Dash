"""Genera todas las figuras de la presentación en figuras/e1 ... figuras/e8, con las corridas que haya en runs/.

Correr después de `git pull` (cuando los 4 integrantes subieron sus corridas):
    python experimentos/graficos.py
Las corridas que todavía no están se saltean con un aviso, así que se puede correr antes para ver cómo viene.
"""
import os
import subprocess
import sys

from comun import RAIZ, RUNS

# figura -> [(corrida, etiqueta)], en el orden en que aparecen en la leyenda
COMPARACIONES = {
    "e1": [("base_ppo", "PPO"), ("e1_qrdqn", "QR-DQN")],
    "e2": [("base_ppo", "features (MLP)"), ("e2_obs_grid", "grid (CNN)"), ("e2_obs_dict", "dict (CNN + vector)")],
    "e3": [("base_ppo", "sin castigo"), ("e3_fijo", "c = 0.1 fijo"), ("e3_rampa", "c = 0.1 con rampa")],
    "e4": [("base_ppo", "fase fija"), ("e4_random", "fase aleatoria")],
    "e5": [("e5_santi_jp", "castigo fijo c = 0.05"), ("e5_santi_salto_preciso", "salto_preciso")],
    "e6": [("dict_fixed_lv-nacho_jp0.05", "sin progresión"),
           ("dict_fixed_lv-nacho_progresion_prog_jp0.05", "con progresión"),
           ("dict_fixed_lv-nacho_jp0.2", "sin progresión, c = 0.2")],
    "e8": [("e8_generalizacion", "20 niveles de entrenamiento")],
}


def main():
    os.chdir(RAIZ)
    for fig, series in COMPARACIONES.items():
        hay = [(c, e) for c, e in series if (RUNS / c / "resumen.md").exists()]
        faltan = [c for c, _ in series if (c, _) not in hay]
        if faltan:
            print(f"[{fig}] faltan: {', '.join(faltan)}")
        if not hay:
            continue
        cmd = [sys.executable, "scripts/plot.py", *[f"runs/{c}" for c, _ in hay],
               "--labels", *[e for _, e in hay], "--out", f"figuras/{fig}"]
        print(f"[{fig}] {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd)

    e7 = sorted(RUNS.glob("e7_nivel_*"))
    print(f"[e7] {len(e7)} niveles", flush=True)
    if len(e7) >= 3:          # con menos niveles la correlación no dice nada
        subprocess.run([sys.executable, "scripts/difficulty_vs_performance.py", "runs/e7_nivel_*", "--out", "figuras/e7"])


if __name__ == "__main__":
    main()

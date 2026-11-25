"""Entrena PPO / A2C / DQN / QR-DQN (Stable-Baselines3 y sb3-contrib) sobre GeometryDash-v0, evalúa en varias fases de arranque y graba GIFs.

Ejemplos
--------
  python scripts/train.py --algos ppo dqn --seeds 0 1 2 --timesteps 300000
  python scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2 --timesteps 300000 --run-name e1_algoritmos
  python scripts/train.py --algos ppo --seeds 0 --timesteps 20000 --run-name humo     # prueba rápida
  python scripts/train.py --algos ppo dqn --obs grid --seeds 0 1 2                    # CNN sobre la grilla 3x7x16
  python scripts/train.py --algos ppo --obs dict --seeds 0                            # CNN (grilla) + MLP (vector)
  python scripts/train.py --algos ppo --jump-penalty 0.1 --ramp                       # penalización por salto con rampa
  python scripts/train.py --algos ppo --win-bonus 2 --death-penalty 5 --jump-penalty 0.05   # parámetros de la recompensa
  python scripts/train.py --algos ppo --reward-fn ejemplos/reward_ejemplo.py:penaliza_saltos   # recompensa propia
  python scripts/train.py --algos ppo --rewards progress win death salto_preciso --levels niveles/santi.json
                                                                                     # términos de la recompensa a elección
  python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 --init-model runs/X/ppo/seed0/best_model.zip
  python scripts/train.py --algos ppo --levels niveles/ejemplo.json                   # varios niveles + validación
  python scripts/train.py --algos ppo --levels level_1 level_2                        # nombres sueltos (peso 1 c/u)

Salida (por agente) en  runs/<run-name>/<algo>/seed<N>/ :
  model.zip, best_model.zip (+ .json), checkpoints/, monitor/, eval_history.csv, eval_fases*.csv, GIFs y *_recorrido.png
Resumen entre seeds: runs/<run-name>/resumen.md. Una corrida nunca pisa a otra. Detalle de las opciones y archivos: README.
Desde Python: ``from agents import train; train("ppo", win_bonus=2, death_penalty=5, reward_fn=mi_fn)``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importan gd_env y agents

from agents import ALGOS, train
from gd_env import REWARD_TERMS


def parse_args():
    """Define y valida las opciones de línea de comandos (ver ``python scripts/train.py -h``)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=["ppo", "dqn"], choices=list(ALGOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=300_000,
                    help="pasos de entrenamiento (con --init-model: pasos adicionales)")
    ap.add_argument("--obs", default="features", choices=["features", "grid", "dict"],
                    help="observación: features (vector + MlpPolicy), grid (imagen 3x7x16 + CNN propia) "
                         "o dict (grilla + vector, CNN + MLP). Ver agents/cnn.py")
    ap.add_argument("--train-phase", default="fixed", choices=["fixed", "random"],
                    help="fixed: siempre x0=2.0 (nivel fijo). random: x0 aleatorio en cada episodio")
    ap.add_argument("--n-envs", type=int, default=8, help="entornos en paralelo (sólo PPO/A2C; DQN y QR-DQN usan 1)")
    ap.add_argument("--levels", nargs="+", default=None, metavar="NIVEL|ARCHIVO.json",
                    help="niveles de entrenamiento: nombres (gd_env.LEVELS) y/o un .json con pesos y niveles de "
                         "validación (ver niveles/ejemplo.json). Por defecto, sólo level_1")
    ap.add_argument("--val-seed", type=int, default=0,
                    help="semilla del sorteo de niveles de validación ('validacion': N o null en el .json)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-gif", action="store_true")

    g = ap.add_argument_group("recompensa (si no se pasan, valen los valores por defecto de GeometryDashEnv)")
    g.add_argument("--jump-penalty", type=float, default=0.0,
                   help="se resta en cada salto efectivo (p. ej. 0.05); 0 = sin penalización. Con --ramp es el valor máximo")
    g.add_argument("--death-penalty", type=float, default=None, help="penalización al morir (def. 10)")
    g.add_argument("--win-bonus", type=float, default=None, help="premio al llegar a la meta (def. 1)")
    g.add_argument("--progress-scale", type=float, default=None, help="lo que vale cruzar todo el nivel (def. 1)")
    g.add_argument("--survive-bonus", type=float, default=None, help="bonus por supervivencia (def. 0)")
    g.add_argument("--survive-interval", type=int, default=None,
                   help="el bonus se da cada N ticks vivo (def. 1 = todos los ticks); p. ej. --survive-bonus 0.01 --survive-interval 20")
    g.add_argument("--survive-rate", type=float, default=None,
                   help="alternativa a --survive-bonus: bonus por tick = RATE x jump-penalty (p. ej. 0.1); sigue a la rampa y "
                        "no hay que recalcularlo al cambiar de nivel o de --jump-penalty")
    g.add_argument("--rewards", nargs="+", default=None, choices=list(REWARD_TERMS), metavar="TERMINO",
                   help="términos de la recompensa que quedan prendidos (def.: progress win death survive jump_penalty). "
                        f"Opciones: {', '.join(REWARD_TERMS)}. 'jump_penalty' y 'salto_preciso' son excluyentes. "
                        "No se combina con --reward-fn")
    g.add_argument("--reward-fn", default=None, metavar="ARCHIVO.py:FUNCION",
                   help="recompensa propia, p. ej. ejemplos/reward_ejemplo.py:penaliza_saltos (o paquete.modulo:funcion). "
                        "Reemplaza a la de por defecto (--jump-penalty se sigue sumando). Ver gd_env/rewards.py")

    g = ap.add_argument_group("rampa de la penalización por salto (opcional)")
    g.add_argument("--ramp", action="store_true",
                   help="c arranca en 0 y sube hasta --jump-penalty sólo cuando el agente ya gana")
    g.add_argument("--ramp-step", type=float, default=0.01, help="cuánto sube/baja c en cada decisión (def. 0.01)")
    g.add_argument("--ramp-window", type=int, default=20, help="episodios de entrenamiento por decisión (def. 20)")
    g.add_argument("--ramp-up", type=float, default=0.8, help="tasa de éxito para subir c (def. 0.8)")
    g.add_argument("--ramp-down", type=float, default=0.4, help="tasa de éxito para bajar c (def. 0.4)")

    g = ap.add_argument_group("guardado")
    g.add_argument("--checkpoint-freq", type=int, default=25_000, help="modelo intermedio cada N pasos (0 = no)")
    g.add_argument("--eval-freq", type=int, default=25_000, help="evalúa cada N pasos y guarda best_model.zip (0 = no)")
    g.add_argument("--init-model", default=None,
                   help="seguir entrenando este .zip en vez de empezar de cero (un solo algoritmo en --algos)")
    return ap.parse_args()


def main():
    a = vars(parse_args())
    a["gif"] = not a.pop("no_gif")
    try:
        train(**a)
    except (ValueError, FileNotFoundError) as e:       # combinaciones inválidas / archivos que no existen
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

"""Re-evalúa modelos guardados en varias fases de arranque y (opcional) graba sus GIFs.

  python scripts/evaluate.py --run runs/features_fixed                 # todos los modelos del run
  python scripts/evaluate.py --run runs/features_fixed --gif           # además regraba los GIFs
  python scripts/evaluate.py --run runs/features_fixed --best          # usa best_model.zip en vez de model.zip
  python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo

La recompensa (jump_penalty, death_penalty) y el tipo de observación (--obs: features, grid o dict) se toman del
config.json de la corrida, para que el retorno coincida con el que calculó train.py y el modelo reciba la observación
con la que se entrenó. Con --model suelto sin config.json hay que pasar --obs si no es "features".
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importan gd_env y agents

from agents import ALGOS, EVAL_PHASES, summarize
from agents.common import load_model
from agents.evaluate import describe
from agents.runs import evaluate_and_record, reward_kwargs, run_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="carpeta de un run de train.py (busca <algo>/seed*/model.zip)")
    ap.add_argument("--model", help="un modelo suelto (.zip)")
    ap.add_argument("--algo", choices=list(ALGOS), help="algoritmo del --model")
    ap.add_argument("--obs", default=None, choices=["features", "grid", "dict"],
                    help="tipo de observación del modelo (por defecto: el del config.json de la corrida, o features)")
    ap.add_argument("--n-phases", type=int, default=len(EVAL_PHASES), help="cantidad de fases x0 en [2.0, 2.95]")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--best", action="store_true", help="con --run: evalúa best_model.zip en vez de model.zip")
    a = ap.parse_args()

    phases = [float(x) for x in np.linspace(2.0, 2.95, a.n_phases)]     # fases de arranque equiespaciadas en [2.0, 2.95]
    jobs = []                                     # (algo, ruta del modelo)
    if a.run:
        name = "best_model" if a.best else "model"
        jobs += [(m.parent.parent.name, m) for m in sorted(Path(a.run).glob(f"*/seed*/{name}.zip"))]
    if a.model:
        assert a.algo, "con --model hay que indicar --algo"
        jobs.append((a.algo, Path(a.model)))
    assert jobs, "no se encontró ningún modelo"

    results = {}                                  # algoritmo -> lista (una por modelo/seed) de filas por fase
    for algo, path in jobs:
        cfg = run_config(path)                                       # config.json de la corrida ({} si no hay)
        obs_mode = a.obs or cfg.get("obs", "features")               # --obs manda; si no, el de la corrida
        rows = evaluate_and_record(load_model(algo, str(path)), path, obs_mode, reward_kwargs(cfg),
                                   gif=a.gif, phases=phases)
        results.setdefault(algo.upper(), []).append(rows)
        print(f"{path}: {describe(rows)}")
    print("\n" + summarize(results))


if __name__ == "__main__":
    main()

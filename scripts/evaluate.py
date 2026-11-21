"""Re-evalúa modelos guardados en varias fases de arranque y (opcional) graba sus GIFs.

  python scripts/evaluate.py --run runs/features_fixed                 # todos los modelos del run
  python scripts/evaluate.py --run runs/features_fixed --gif           # además regraba los GIFs
  python scripts/evaluate.py --run runs/features_fixed --best          # usa best_model.zip en vez de model.zip
  python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo

La recompensa (jump_penalty, death_penalty) se toma del config.json de la corrida, para que el retorno
coincida con el que calculó train.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import ALGOS, EVAL_PHASES, summarize
from agents.common import load_model
from agents.evaluate import describe
from agents.runs import evaluate_and_record, reward_kwargs, run_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="carpeta de un run de train.py (busca <algo>/seed*/model.zip)")
    ap.add_argument("--model", help="un modelo suelto (.zip)")
    ap.add_argument("--algo", choices=list(ALGOS), help="algoritmo del --model")
    ap.add_argument("--obs", default="features", choices=["features"])
    ap.add_argument("--n-phases", type=int, default=len(EVAL_PHASES), help="cantidad de fases x0 en [2.0, 2.95]")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--best", action="store_true", help="con --run: evalúa best_model.zip en vez de model.zip")
    a = ap.parse_args()

    phases = [float(x) for x in np.linspace(2.0, 2.95, a.n_phases)]
    jobs = []                                     # (algo, ruta del modelo)
    if a.run:
        name = "best_model" if a.best else "model"
        jobs += [(m.parent.parent.name, m) for m in sorted(Path(a.run).glob(f"*/seed*/{name}.zip"))]
    if a.model:
        assert a.algo, "con --model hay que indicar --algo"
        jobs.append((a.algo, Path(a.model)))
    assert jobs, "no se encontró ningún modelo"

    results = {}
    for algo, path in jobs:
        rows = evaluate_and_record(load_model(algo, str(path)), path, a.obs, reward_kwargs(run_config(path)),
                                   gif=a.gif, phases=phases)
        results.setdefault(algo.upper(), []).append(rows)
        print(f"{path}: {describe(rows)}")
    print("\n" + summarize(results))


if __name__ == "__main__":
    main()

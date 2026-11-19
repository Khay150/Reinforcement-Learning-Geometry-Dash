"""Re-evalúa modelos guardados en varias fases de arranque y (opcional) graba sus GIFs.

  python scripts/evaluate.py --run runs/features_fixed                 # todos los modelos del run
  python scripts/evaluate.py --run runs/features_fixed --gif           # además regraba los GIFs
  python scripts/evaluate.py --model runs/features_fixed/ppo/seed0/model.zip --algo ppo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import ALGOS, EVAL_PHASES, evaluate_policy_phases, summarize
from agents.common import load_model
from agents.evaluate import save_rows
from train import record_agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="carpeta de un run de train.py (busca <algo>/seed*/model.zip)")
    ap.add_argument("--model", help="un modelo suelto (.zip)")
    ap.add_argument("--algo", choices=list(ALGOS), help="algoritmo del --model")
    ap.add_argument("--obs", default="features", choices=["features"])
    ap.add_argument("--n-phases", type=int, default=len(EVAL_PHASES), help="cantidad de fases x0 en [2.0, 2.95]")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    import numpy as np
    phases = [float(x) for x in np.linspace(2.0, 2.95, a.n_phases)]
    jobs = []                                     # (nombre, algo, ruta del modelo)
    if a.run:
        for m in sorted(Path(a.run).glob("*/seed*/model.zip")):
            jobs.append((m.parent.parent.name.upper(), m.parent.parent.name, m))
    if a.model:
        assert a.algo, "con --model hay que indicar --algo"
        jobs.append((a.algo.upper(), a.algo, Path(a.model)))
    assert jobs, "no se encontró ningún modelo"

    results = {}
    for name, algo, path in jobs:
        model = load_model(algo, str(path))
        rows = evaluate_policy_phases(model, a.obs, phases)
        save_rows(rows, path.parent / "eval_fases.csv")
        if a.gif:
            record_agent(model, a.obs, path.parent, rows)
        results.setdefault(name, []).append(rows)
        print(f"{path.parent}: gana {sum(r['success'] for r in rows)}/{len(rows)} fases")
    print("\n" + summarize(results))


if __name__ == "__main__":
    main()

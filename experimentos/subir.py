"""Sube corridas terminadas al repo: un commit por seed y un push después de cada uno (las seeds pesan ~15-160 MB).

    python experimentos/subir.py runs/e2_obs_grid runs/base_ppo
Lo que ya está commiteado se saltea, así que se puede repetir sin problema.
"""
import sys
from pathlib import Path

from comun import subir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    subir([Path(p) for p in sys.argv[1:]])

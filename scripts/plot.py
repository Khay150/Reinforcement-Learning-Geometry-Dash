"""Gráficos de una o varias corridas de train.py, listos para las diapositivas (PNG, 200 dpi).

Ejemplos
--------
  python scripts/plot.py runs/features_fixed                                  # todo, en figuras/features_fixed/
  python scripts/plot.py runs/features_fixed runs/features_fixed_jp0.05 --labels "sin castigo" "c = 0.05"
  python scripts/plot.py runs/e1_algoritmos runs/e2_obs_grid --out figuras/e1 --tipo curvas barras
  python scripts/plot.py runs/grid_fixed_lv-nacho_progresion_prog_jp0.05 --tipo progresion

Cada corrida es una carpeta runs/<nombre>/ con <algo>/seed<N>/ adentro. Cada par (corrida, algoritmo) es una serie
con su color; las seeds se promedian (banda: mín–máx entre seeds). Qué figura sale de qué archivo:

  curvas        curva_{retorno,exito,progreso}.png   monitor/*.monitor.csv (episodios de entrenamiento)
  eval          eval_{exito,progreso}.png            eval_history.csv (evaluación periódica; punteada = validación)
  barras        barras_{entrenamiento,validacion}.png eval_fases*.csv (modelo final y mejor modelo)
  niveles       barras_niveles.png                   eval_fases*.csv, fases ganadas en cada nivel (si hay varios)
  progresion    progresion_<corrida>_<algo>.png      progresion.csv (etapas, éxito y probabilidad de cada nivel)
  penalizacion  penalizacion_salto.png               jump_penalty.csv (sólo corridas con --ramp)

Lo que falta (una seed sin terminar, una corrida vieja sin eval_history.csv, ...) se saltea con un aviso. Sirve con
corridas en curso: las curvas llegan hasta donde llegó cada seed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importa agents

from agents.plots import TIPOS, graficar, nombre_salida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corridas", nargs="+", help="carpetas runs/<nombre> (cada una con <algo>/seed*/ adentro)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="un nombre por corrida para las leyendas (por defecto, el nombre de la carpeta)")
    ap.add_argument("--out", default=None,
                    help="carpeta de salida (por defecto figuras/<corrida>, o figuras/comparacion_<primera>_+N)")
    ap.add_argument("--tipo", nargs="+", choices=TIPOS, default=list(TIPOS),
                    help="qué figuras hacer (por defecto, todas las que tengan datos)")
    ap.add_argument("--suavizado", type=float, default=0.04,
                    help="ancho de la media móvil de las curvas, como fracción de los pasos totales (def. 0.04)")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path("figuras") / nombre_salida(a.corridas)
    print(f"Figuras en {out}/")
    paths = graficar(a.corridas, out, a.labels, a.tipo, a.suavizado)
    if not paths:
        sys.exit("no se generó ninguna figura (ver los avisos de arriba)")
    print(f"{len(paths)} figuras.")


if __name__ == "__main__":
    main()

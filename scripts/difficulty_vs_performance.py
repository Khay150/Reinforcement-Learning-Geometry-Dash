"""¿La dificultad que medimos con la física (gd_env/difficulty.py) predice cuánto le cuesta un nivel a un agente de RL?

Toma corridas de ``scripts/train.py`` entrenadas en **un solo nivel** cada una (p. ej. las de E7, ``runs/e7_nivel_*``),
mide la dificultad de ese nivel con ``difficulty_over_phases`` (lo mismo que ``scripts/difficulty.py``) y la cruza con
el desempeño de cada seed:

* **pasos hasta la 1ª victoria**: primer episodio de *entrenamiento* ganado (columna ``success`` de ``monitor/*.csv``).
  Los entornos en paralelo avanzan juntos (un paso de cada uno por paso del VecEnv), así que un episodio que termina
  cuando su entorno lleva ``k`` pasos corresponde a ``k x n_entornos`` pasos totales. Una seed que nunca ganó queda
  **censurada** en los pasos totales de la corrida (se marca y cuenta como "al menos eso").
* **pasos hasta aprenderlo**: primer momento en que la tasa de éxito de los últimos ``--ventana`` (50) episodios de
  entrenamiento (de todos los entornos, en orden) llega a ``--umbral`` (50%). La 1ª victoria mide sobre todo cuánto
  cuesta *encontrar* la solución explorando (en un nivel fácil puede salir en el primer episodio, de suerte); ésta mide
  cuánto cuesta *aprenderla*. También se censura si no llega.
* **pasos hasta ganar una evaluación**: primera fila de ``eval_history.csv`` con éxito > 0 (política determinista, sin
  exploración). Si un monitor no tiene la columna ``success`` (corridas muy viejas), se usa esto como 1ª victoria.
* **éxito / progreso final**: ``eval_fases.csv`` (modelo final, 10 fases); **del mejor modelo**: ``eval_fases_best.csv``
  (o ``best_model.json``).

Sale una tabla por nivel (media ± desvío entre seeds), las correlaciones de Spearman entre cada medida de dificultad y
cada medida de desempeño (con valor p por permutaciones, sin scipy) y gráficos de dispersión para las diapositivas.

Ejemplos
--------
  python scripts/difficulty_vs_performance.py 'runs/e7_nivel_*'            # E7 (las comillas: el glob lo expande el script)
  python scripts/difficulty_vs_performance.py runs/e7_nivel_level_1 runs/e7_nivel_franco_facil --out figuras/prueba
  python scripts/difficulty_vs_performance.py 'runs/e7_nivel_*' --incompletas   # incluye seeds que todavía entrenan
  python scripts/difficulty_vs_performance.py 'runs/e7_nivel_*' --pares n_jumps:exito_final max_chain:pasos_1a_victoria
  python scripts/difficulty_vs_performance.py 'runs/e7_nivel_*' --fases 1       # dificultad sólo en x0 = 2.0

Salida (por defecto en figuras/dificultad_vs_desempeno/):
  tabla.md (tabla + correlaciones), tabla.csv (una fila por nivel), semillas.csv (una fila por seed),
  <dificultad>_vs_<desempeño>.png (un gráfico por par) y panel.png (los gráficos por defecto juntos).
"""
from __future__ import annotations

import argparse
import glob
import io
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # raíz del repo en el path: así se importa gd_env

import numpy as np
import pandas as pd

from gd_env import difficulty_over_phases, get_level, resolve_levels

# Medidas de dificultad (claves de difficulty_over_phases) -> (etiqueta, sentido): +1 = más alto es más difícil,
# -1 = más bajo es más difícil. En los gráficos el eje x se da vuelta si hace falta: más difícil siempre a la derecha.
DIFFICULTY = {
    "n_jumps": ("saltos mínimos", +1),
    "min_window": ("ventana mínima (ticks)", -1),
    "mean_window": ("ventana media (ticks)", -1),
    "jumps_with_traps": ("saltos con trampas", +1),
    "trap_ticks": ("ticks trampa", +1),
    "max_chain": ("cadena más larga (saltos)", +1),
    "linked_jumps": ("saltos encadenados", +1),
    "log10_p_random": ("log10 P(ganar al azar)", -1),
}
# Medidas de desempeño (por seed) -> (etiqueta, es_pasos). Éxito y progreso en %.
PERFORMANCE = {
    "pasos_1a_victoria": ("pasos hasta la 1ª victoria", True),
    "pasos_aprende": ("pasos hasta ganar el 50% en entrenamiento", True),    # la etiqueta sigue a --umbral
    "pasos_eval_gana": ("pasos hasta ganar una evaluación", True),
    "exito_final": ("éxito final (%)", False),
    "progreso_final": ("progreso final (%)", False),
    "exito_mejor": ("éxito del mejor modelo (%)", False),
    "progreso_mejor": ("progreso del mejor modelo (%)", False),
}
CENSORED = {"pasos_1a_victoria": "censurada_1a_victoria", "pasos_aprende": "censurada_aprende",
            "pasos_eval_gana": "censurada_eval_gana"}
DEFAULT_PAIRS = ["log10_p_random:pasos_1a_victoria", "log10_p_random:pasos_aprende",
                 "log10_p_random:progreso_final", "min_window:exito_final"]

# Okabe-Ito (seguro para daltonismo). La forma también distingue: círculo lleno = ganó, triángulo hueco = nunca ganó.
BLUE, VERMILLION, GREY, INK = "#0072B2", "#D55E00", "#8C8C8C", "#222222"


# ----------------------------------------------------------------------------------------------------------- corridas
def expand_runs(items: list[str]) -> list[Path]:
    """Carpetas de corridas (con ``config.json``) a partir de rutas y/o globs. Una carpeta sin ``config.json`` (p. ej.
    ``runs/``) aporta las corridas que tenga adentro."""
    out = []
    for item in items:
        matches = sorted(glob.glob(item)) or [item]
        for m in map(Path, matches):
            if (m / "config.json").is_file():
                out.append(m)
            elif m.is_dir():
                out += sorted(p.parent for p in m.glob("*/config.json"))
            elif not m.is_file() or len(matches) == 1:    # un .log que entró en el glob no es un error
                print(f"[aviso] {m}: no existe o no es una corrida (falta config.json)", file=sys.stderr)
    return list(dict.fromkeys(out))


def train_level_names(cfg: dict) -> list[str]:
    """Niveles de entrenamiento según ``config.json``: ``train_levels`` (corridas nuevas) o, si falta, lo que resulte de
    ``levels`` (las corridas viejas sin niveles entrenaban en ``level_1``)."""
    if cfg.get("train_levels"):
        return list(cfg["train_levels"])
    return list(resolve_levels(cfg.get("levels"), cfg.get("val_seed", 0))["train"])


def _as_bool(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def read_monitors(seed_dir: Path) -> list[pd.DataFrame]:
    """Un DataFrame por entorno (``monitor/<i>.monitor.csv``). Tolera la última línea a medio escribir de una corrida
    que sigue entrenando."""
    out = []
    for f in sorted((seed_dir / "monitor").glob("*.monitor.csv")):
        text = f.read_text()
        if not text.endswith("\n"):             # la última línea todavía se está escribiendo: se descarta
            text = text[:text.rfind("\n") + 1]
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=1, on_bad_lines="skip")
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame(columns=["r", "l", "t"])
        df["l"] = pd.to_numeric(df.get("l"), errors="coerce")
        df = df.dropna(subset=["l"]).reset_index(drop=True)
        out.append(df)
    return out


def train_episodes(monitors: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Episodios de entrenamiento de todos los entornos, ordenados por el paso total del VecEnv en que terminaron
    (columnas ``pasos`` y ``gano``). Los ``n`` entornos avanzan juntos: si un episodio termina cuando su entorno lleva
    ``k`` pasos, el VecEnv lleva ``k * n``. None si los monitores no tienen la columna ``success``."""
    n = len(monitors)
    if n == 0 or not all("success" in m.columns for m in monitors if len(m)):
        return None
    parts = [pd.DataFrame({"pasos": n * m["l"].cumsum().astype(int), "gano": _as_bool(m["success"]), "env": i})
             for i, m in enumerate(monitors) if len(m)]
    if not parts:
        return pd.DataFrame({"pasos": [], "gano": [], "env": []})
    return pd.concat(parts).sort_values(["pasos", "env"], kind="stable").reset_index(drop=True)


def first_train_win(episodes: pd.DataFrame | None) -> int | None:
    """Pasos totales hasta el primer episodio de entrenamiento ganado (None si nunca ganó o no se sabe)."""
    if episodes is None or not episodes["gano"].any():
        return None
    return int(episodes.loc[episodes["gano"], "pasos"].iloc[0])


def learned_at(episodes: pd.DataFrame | None, window: int = 50, threshold: float = 0.5) -> int | None:
    """Pasos totales hasta que el éxito de los últimos ``window`` episodios de entrenamiento llega a ``threshold``."""
    if episodes is None or len(episodes) < window:
        return None
    rate = episodes["gano"].astype(float).rolling(window).mean()
    hit = episodes.loc[rate >= threshold - 1e-9, "pasos"]
    return int(hit.iloc[0]) if len(hit) else None


def first_eval_win(seed_dir: Path) -> int | None:
    """Pasos de la primera evaluación periódica (``eval_history.csv``) que ganó al menos una fase."""
    f = seed_dir / "eval_history.csv"
    if not f.is_file():
        return None
    h = pd.read_csv(f, on_bad_lines="skip")
    s = pd.to_numeric(h.get("success"), errors="coerce")
    won = h.loc[s > 0, "timesteps"]
    return int(won.iloc[0]) if len(won) else None


def _eval_means(path: Path, level: str) -> tuple[float, float]:
    """(éxito %, progreso %) medios de un ``eval_fases*.csv`` en ``level`` (NaN si no existe)."""
    if not path.is_file():
        return math.nan, math.nan
    df = pd.read_csv(path)
    if "level" in df.columns:
        df = df[df["level"] == level]
    if not len(df):
        return math.nan, math.nan
    return 100 * float(df["success"].mean()), 100 * float(df["progress"].mean())


def seed_performance(seed_dir: Path, level: str, total_timesteps: int, window: int = 50,
                     threshold: float = 0.5) -> dict:
    """Desempeño de una seed (ver el docstring del módulo). Censura: si nunca ganó, los pasos quedan en el total de la
    corrida (o en los observados hasta ahora, si sigue entrenando) y ``censurada_* = True``."""
    complete = (seed_dir / "model.zip").is_file()
    monitors = read_monitors(seed_dir)
    episodes = train_episodes(monitors)
    observed = len(monitors) * int(max((m["l"].sum() for m in monitors), default=0))
    train_win, learned = first_train_win(episodes), learned_at(episodes, window, threshold)
    eval_win = first_eval_win(seed_dir)
    limit = total_timesteps if complete else observed
    if episodes is None:                        # monitor sin 'success': la 1ª victoria sale del historial de evaluación
        train_win, source = eval_win, "eval_history"
    else:
        source = "monitor"
    row = {"completa": complete, "pasos_totales": total_timesteps, "pasos_observados": observed,
           "fuente_1a_victoria": source}
    for key, value in (("pasos_1a_victoria", train_win), ("pasos_aprende", learned), ("pasos_eval_gana", eval_win)):
        row[key] = value if value is not None else limit
        row[CENSORED[key]] = value is None
    row["exito_final"], row["progreso_final"] = _eval_means(seed_dir / "eval_fases.csv", level)
    row["exito_mejor"], row["progreso_mejor"] = _eval_means(seed_dir / "eval_fases_best.csv", level)
    if math.isnan(row["exito_mejor"]) and (seed_dir / "best_model.json").is_file():
        best = json.loads((seed_dir / "best_model.json").read_text())
        row["exito_mejor"] = 100 * best.get(f"success_{level}", best.get("success", math.nan))
        row["progreso_mejor"] = 100 * best.get("progress", math.nan)
    return row


def eval_phases(start_x: float, n: int) -> list[float]:
    """Las ``n`` fases de arranque de la evaluación (igual que ``scripts/difficulty.py``)."""
    return [start_x] if n <= 1 else [float(x) for x in start_x + np.linspace(0.0, 0.95, n)]


def collect(run_dirs: list[Path], n_phases: int = 10, include_incomplete: bool = False, window: int = 50,
            threshold: float = 0.5) -> pd.DataFrame:
    """Una fila por (corrida, algoritmo, seed) con la dificultad del nivel y el desempeño de la seed."""
    cache: dict[str, dict] = {}
    rows = []
    for run in run_dirs:
        cfg = json.loads((run / "config.json").read_text())
        try:
            levels = train_level_names(cfg)
        except Exception as e:                  # p. ej. un .json de niveles que ya no está
            print(f"[aviso] {run}: no pude saber en qué nivel entrenó ({e}); se saltea", file=sys.stderr)
            continue
        if len(levels) != 1:
            print(f"[aviso] {run}: entrena en {len(levels)} niveles ({', '.join(levels)}); este análisis es de "
                  f"corridas de un solo nivel, se saltea", file=sys.stderr)
            continue
        level = levels[0]
        if level not in cache:
            lv = get_level(level)
            cache[level] = difficulty_over_phases(lv, eval_phases(lv.start_x, n_phases))
        seed_dirs = sorted(run.glob("*/seed*"), key=lambda p: (p.parent.name, int(p.name[4:]) if p.name[4:].isdigit() else 0))
        if not seed_dirs:
            print(f"[aviso] {run}: todavía no tiene seeds", file=sys.stderr)
        for sd in seed_dirs:
            perf = seed_performance(sd, level, int(cfg.get("timesteps", 0)), window, threshold)
            if not perf["completa"] and not include_incomplete:
                print(f"[aviso] {sd}: sigue entrenando (no hay model.zip); se saltea (--incompletas la incluye)",
                      file=sys.stderr)
                continue
            rows.append({"nivel": level, "corrida": run.name, "algo": sd.parent.name, "seed": sd.name[4:],
                         **{k: cache[level].get(k) for k in ("solvable_phases", *DIFFICULTY)}, **perf})
    return pd.DataFrame(rows)


def aggregate(seeds: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (corrida, algoritmo): dificultad + media y desvío entre seeds de cada medida de desempeño."""
    out = []
    multi_algo = seeds["algo"].nunique() > 1
    for (run, algo), g in seeds.groupby(["corrida", "algo"], sort=False):
        row = {"nivel": g["nivel"].iloc[0], "corrida": run, "algo": algo, "seeds": len(g),
               **{k: g[k].iloc[0] for k in ("solvable_phases", *DIFFICULTY)}}
        for k in PERFORMANCE:
            v = pd.to_numeric(g[k], errors="coerce")
            row[f"{k}_media"] = float(v.mean()) if v.notna().any() else math.nan
            row[f"{k}_desvio"] = float(v.std(ddof=0)) if v.notna().any() else math.nan
        for k, c in CENSORED.items():
            row[f"{k}_ganaron"] = int((~g[c].astype(bool)).sum())
        out.append(row)
    agg = pd.DataFrame(out)
    if len(agg):
        # Etiqueta para gráficos y tabla: el nivel; si se repite (dos corridas o algoritmos en el mismo), se aclara.
        dup = agg["nivel"].duplicated(keep=False)
        agg["etiqueta"] = agg["nivel"]
        if multi_algo:
            agg["etiqueta"] = agg["nivel"] + " (" + agg["algo"].str.upper() + ")"
        else:
            agg.loc[dup, "etiqueta"] = agg["nivel"] + " (" + agg["corrida"] + ")"
    return agg


# ----------------------------------------------------------------------------------------------------------- Spearman
def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else math.nan


def spearman(x, y, n_perm: int = 20_000, seed: int = 0) -> tuple[float, float, int]:
    """(ρ de Spearman, valor p a dos colas por permutaciones, n). Los empates se rankean con el promedio (así una seed
    censurada en el total de pasos empata con las otras censuradas). El valor p es exacto (todas las permutaciones)
    hasta n = 8 y Monte Carlo con ``n_perm`` permutaciones más allá. NaN si hay menos de 3 pares o una variable es
    constante."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return math.nan, math.nan, n
    rx, ry = _ranks(x), _ranks(y)
    rho = _pearson(rx, ry)
    if math.isnan(rho):
        return math.nan, math.nan, n
    if n <= 8:
        perms = np.array(list(itertools.permutations(ry)))
    else:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
    a = rx - rx.mean()
    b = perms - perms.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum() * (b * b).sum(axis=1))
    null = (b @ a) / den
    p = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    return rho, p, n


def correlations(agg: pd.DataFrame) -> pd.DataFrame:
    """ρ de Spearman (entre niveles, con la media entre seeds) de cada medida de dificultad con cada una de desempeño."""
    rows = []
    for d in DIFFICULTY:
        for k in PERFORMANCE:
            rho, p, n = spearman(agg[d], agg[f"{k}_media"])
            rows.append({"dificultad": d, "desempeno": k, "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------- tablas
def _num(v, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float) and v.is_integer() and nd == 0:
        return str(int(v))
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _steps(v: float) -> str:
    return "-" if math.isnan(v) else f"{v / 1000:.1f}k"


def markdown(agg: pd.DataFrame, corr: pd.DataFrame, n_phases: int) -> str:
    agg = agg.sort_values("log10_p_random", ascending=False, na_position="last")
    cols = ["nivel", "saltos mín.", "ventana mín.", "ventana media", "saltos c/ trampas", "cadena máx.",
            "log10 P(azar)", "seeds", "ganaron (entren.)", "pasos 1ª victoria", "pasos hasta aprender", "pasos 1ª eval. ganada",
            "éxito final %", "progreso final %", "éxito mejor %"]
    lines = ["## Dificultad vs. desempeño", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in agg.iterrows():
        def pm(k, fmt=lambda v: f"{v:.0f}"):
            m, s = r[f"{k}_media"], r[f"{k}_desvio"]
            return "-" if math.isnan(m) else f"{fmt(m)} ± {fmt(s)}"

        def steps(k):
            won, n = r[f"{k}_ganaron"], r["seeds"]
            if won == 0:                        # todas censuradas: no llegó en ninguna seed
                return f"nunca (≥ {_steps(r[f'{k}_media'])})"
            txt = pm(k, _steps)
            return txt if won == n else f"≥ {txt} ({won}/{n} llegaron)"
        lines.append("| " + " | ".join([
            r["etiqueta"], _num(r["n_jumps"], 0), _num(r["min_window"], 0), _num(r["mean_window"], 2),
            _num(r["jumps_with_traps"], 0), _num(r["max_chain"], 0), _num(r["log10_p_random"], 1),
            str(r["seeds"]), f"{r['pasos_1a_victoria_ganaron']}/{r['seeds']}", steps("pasos_1a_victoria"),
            steps("pasos_aprende"), steps("pasos_eval_gana"), pm("exito_final"), pm("progreso_final"), pm("exito_mejor"),
        ]) + " |")
    lines += ["", f"Dificultad: peor caso entre {n_phases} fase(s) de arranque (como `scripts/difficulty.py`). Desempeño: "
              "media ± desvío entre seeds. \"≥\": alguna seed no llegó y cuenta con el total de pasos (censurada); \"pasos hasta "
              "aprender\": primer momento en que el éxito de los últimos episodios de entrenamiento llega al umbral.", "",
              "## Correlación de Spearman entre niveles (ρ, valor p por permutaciones)", "",
              "| dificultad | " + " | ".join(PERFORMANCE[k][0] for k in PERFORMANCE) + " |",
              "|---|" + "---|" * len(PERFORMANCE)]
    for d in DIFFICULTY:
        cells = []
        for k in PERFORMANCE:
            c = corr[(corr["dificultad"] == d) & (corr["desempeno"] == k)].iloc[0]
            cells.append("-" if math.isnan(c["rho"]) else f"{c['rho']:+.2f} ({_p(c['p'])})")
        lines.append(f"| {DIFFICULTY[d][0]} | " + " | ".join(cells) + " |")
    n = int(corr["n"].max()) if len(corr) else 0
    lines += ["", f"n = {n} niveles. Con pocos niveles el valor p es grueso: con n = 5 el mínimo posible es 0.017."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------- gráficos
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.grid": True,
                         "grid.color": "#E3E3E3", "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "savefig.bbox": "tight", "savefig.dpi": 200, "figure.facecolor": "white"})
    return plt


def _place_labels(ax, xs, ys, labels, marks=()):
    """Nombres de los niveles junto a cada punto: prueba posiciones alrededor hasta encontrar una que no pise otro
    nombre, ningún marcador (``marks``: puntos (x, y) en coordenadas de datos) ni se salga del gráfico."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    r = 7 * fig.dpi / 72                                    # radio de un marcador, en píxeles
    obstacles = [Bbox.from_extents(px - r, py - r, px + r, py + r) for px, py in
                 (to_px((mx, my)) for mx, my in marks if np.isfinite(mx) and np.isfinite(my))]
    frame = ax.get_window_extent(renderer)
    # Candidatos: en espiral alrededor del punto (cerca primero); de lejos, con una línea fina hasta el punto.
    angles = [45, -45, 135, -135, 90, -90, 0, 180, 20, -20, 160, -160, 65, -65, 115, -115]
    candidates = [(rad, ang) for rad in (10, 18, 28, 40, 55, 72, 92, 115) for ang in angles]
    style = dict(textcoords="offset points", fontsize=12, color=INK, zorder=5,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    def put(x, y, lab, rad, ang):
        dx, dy = rad * math.cos(math.radians(ang)), rad * math.sin(math.radians(ang))
        ha = "left" if dx > 2 else "right" if dx < -2 else "center"
        va = "bottom" if dy > 2 else "top" if dy < -2 else "center"
        arrow = dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=0, shrinkB=5) if rad >= 28 else None
        t = ax.annotate(lab, (x, y), xytext=(dx, dy), ha=ha, va=va, arrowprops=arrow, **style)
        t.update_positions(renderer)            # si no, Text.get_window_extent no sabe todavía dónde quedó
        return t

    for x, y, lab in sorted(zip(xs, ys, labels), key=lambda t: (t[0], t[1])):
        for rad, ang in candidates:
            t = put(x, y, lab, rad, ang)
            bb = Text.get_window_extent(t, renderer)     # sólo el texto (sin la línea hasta el punto)
            inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
            if inside and not any(bb.overlaps(o) for o in obstacles):
                break
            t.remove()
        else:                                   # no hubo lugar libre: queda en la primera posición
            t = put(x, y, lab, *candidates[0])
            bb = Text.get_window_extent(t, renderer)
        obstacles.append(bb.expanded(1.02, 1.08))


# Nombres cortos para los títulos de los gráficos.
SHORT = {"n_jumps": "saltos mínimos", "min_window": "ventana mínima", "mean_window": "ventana media",
         "jumps_with_traps": "saltos con trampas", "trap_ticks": "ticks trampa", "max_chain": "cadena más larga",
         "linked_jumps": "saltos encadenados", "log10_p_random": "P(ganar al azar)",
         "pasos_1a_victoria": "1ª victoria", "pasos_aprende": "aprender el nivel", "pasos_eval_gana": "1ª evaluación ganada",
         "exito_final": "éxito final", "progreso_final": "progreso final", "exito_mejor": "éxito del mejor modelo",
         "progreso_mejor": "progreso del mejor modelo"}


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.0095 else f"p = {p:.2f}"


def spearman_text(x, y) -> str:
    rho, p, n = spearman(x, y)
    if n < 3:
        return f"ρ de Spearman: n/d (hay {n} nivel{'es' if n != 1 else ''}, hacen falta 3)"
    if math.isnan(rho):
        return f"ρ de Spearman: n/d (todos los niveles empatan; n = {n})"
    return f"ρ de Spearman = {rho:+.2f}   ({_p(p)}, n = {n} niveles)"


def plot_pair(ax, seeds: pd.DataFrame, agg: pd.DataFrame, xkey: str, ykey: str, total_steps: float | None):
    """Un par dificultad (x) vs. desempeño (y): una columna de puntos por nivel (uno por seed) y la media marcada.
    Seeds censuradas (nunca llegaron): triángulo hueco en el total de pasos, sobre la línea punteada."""
    xlab, harder = DIFFICULTY[xkey]
    ylab, is_steps = PERFORMANCE[ykey]
    cens = CENSORED.get(ykey)
    a = agg.dropna(subset=[xkey, f"{ykey}_media"])
    xs = a[xkey].astype(float).to_numpy()
    span = (xs.max() - xs.min()) if len(xs) > 1 and xs.max() > xs.min() else 1.0
    marks = []                                  # dónde hay marcadores: los nombres de los niveles los esquivan
    # Niveles con el mismo valor de x (p. ej. ventana mínima entera): se corren un poco para que no se pisen.
    xpos = {}
    for _, grp in a.groupby(xkey, sort=False):
        for i, idx in enumerate(grp.index):
            xpos[idx] = float(grp.loc[idx, xkey]) + (i - (len(grp) - 1) / 2) * 0.035 * span
    for idx, r in a.iterrows():
        x0 = xpos[idx]
        g = seeds[(seeds["corrida"] == r["corrida"]) & (seeds["algo"] == r["algo"])]
        gy = pd.to_numeric(g[ykey], errors="coerce").to_numpy()
        jitter = (np.arange(len(g)) - (len(g) - 1) / 2) * 0.012 * span
        gx = x0 + jitter
        ax.plot([x0] * 2, [np.nanmin(gy), np.nanmax(gy)], color=GREY, lw=2, zorder=1, solid_capstyle="round")
        won = ~g[cens].astype(bool).to_numpy() if cens else np.ones(len(g), bool)
        ax.scatter(gx[won], gy[won], s=70, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(gx[~won], gy[~won], s=90, marker="^", facecolor="white", edgecolor=VERMILLION, linewidth=2,
                   zorder=3)
        ax.scatter([x0], [r[f"{ykey}_media"]], s=260, marker="_", color=INK, linewidth=2.5, zorder=4)
        marks += list(zip(gx, gy)) + [(x0, float(r[f"{ykey}_media"]))]
    if is_steps:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        ax.set_yscale("log")
        vals = pd.to_numeric(seeds[ykey], errors="coerce")
        top = max(total_steps or 0, float(vals.max()))
        if total_steps:
            ax.axhline(total_steps, color=VERMILLION, ls="--", lw=1.5, zorder=0)
        ax.set_ylim(max(float(vals.min()) / 2, 100), top * 1.6)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(-5, 108)
    if harder < 0:
        ax.invert_xaxis()
    ax.set_xlabel(f"{xlab}   (más difícil →)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{SHORT[ykey].capitalize()} según {SHORT[xkey]}\n{spearman_text(a[xkey], a[f'{ykey}_media'])}",
                 loc="left")
    # Los nombres se ubican al final (``place_labels()``), con la figura ya armada: dependen de dónde quedó cada punto.
    return lambda: _place_labels(ax, [xpos[i] for i in a.index], a[f"{ykey}_media"].astype(float).tolist(),
                                 a["etiqueta"].tolist(), marks)


def _legend(censored: bool, total_steps: float | None):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", markersize=10,
                      label="una seed"),
               Line2D([], [], marker="_", ls="", color=INK, markersize=18, markeredgewidth=2.5,
                      label="media entre seeds")]
    if censored:
        handles.insert(1, Line2D([], [], marker="^", ls="", markerfacecolor="white", markeredgecolor=VERMILLION,
                                 markeredgewidth=2, markersize=10, label="no llegó (censurada)"))
        if total_steps:
            handles.append(Line2D([], [], ls="--", color=VERMILLION, lw=1.5,
                                  label=f"fin del entrenamiento ({total_steps / 1000:g}k pasos)"))
    return handles


def make_plots(seeds: pd.DataFrame, agg: pd.DataFrame, pairs: list[tuple[str, str]], out: Path,
               panel: bool = True) -> list[Path]:
    plt = _style()
    total = float(pd.to_numeric(seeds["pasos_totales"], errors="coerce").max()) or None  # línea "fin del entrenamiento"
    paths = []
    for xk, yk in pairs:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        place_labels = plot_pair(ax, seeds, agg, xk, yk, total)
        hs = _legend(yk in CENSORED, total)
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(len(hs), 2), frameon=False)
        place_labels()
        p = out / f"{xk}_vs_{yk}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    if panel and len(pairs) > 1:
        ncols = 2 if len(pairs) == 4 else min(len(pairs), 3)
        nrows = math.ceil(len(pairs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7 * nrows), squeeze=False)
        placers = [plot_pair(ax, seeds, agg, xk, yk, total) for ax, (xk, yk) in zip(axes.flat, pairs)]
        for ax in list(axes.flat)[len(pairs):]:
            ax.set_visible(False)
        fig.tight_layout(h_pad=3, w_pad=3)
        for place_labels in placers:
            place_labels()
        hs = _legend(any(yk in CENSORED for _, yk in pairs), total)
        fig.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False, bbox_to_anchor=(0.5, 0.0))
        p = out / "panel.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------------------------------------------- CLI
def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for it in items:
        x, sep, y = it.partition(":")
        if not sep or x not in DIFFICULTY or y not in PERFORMANCE:
            raise ValueError(f"par inválido {it!r}: tiene que ser DIFICULTAD:DESEMPEÑO con DIFICULTAD en "
                             f"{', '.join(DIFFICULTY)} y DESEMPEÑO en {', '.join(PERFORMANCE)}")
        pairs.append((x, y))
    return pairs


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="CORRIDA|GLOB",
                    help="carpetas de corridas (con config.json) o globs entre comillas, p. ej. 'runs/e7_nivel_*'")
    ap.add_argument("--out", default="figuras/dificultad_vs_desempeno", help="carpeta de salida")
    ap.add_argument("--fases", type=int, default=10,
                    help="fases de arranque para medir la dificultad (def. 10, peor caso como scripts/difficulty.py; "
                         "1 = sólo x0 = 2.0, la fase en la que se entrena con --train-phase fixed)")
    ap.add_argument("--pares", nargs="+", default=DEFAULT_PAIRS, metavar="DIFICULTAD:DESEMPEÑO",
                    help=f"gráficos a hacer (def. {' '.join(DEFAULT_PAIRS)}). Dificultad: {', '.join(DIFFICULTY)}. "
                         f"Desempeño: {', '.join(PERFORMANCE)}")
    ap.add_argument("--incompletas", action="store_true",
                    help="incluir seeds que siguen entrenando (sin model.zip): se censuran en los pasos que llevan")
    ap.add_argument("--ventana", type=int, default=50,
                    help="episodios de entrenamiento de la ventana de 'pasos_aprende' (def. 50)")
    ap.add_argument("--umbral", type=float, default=0.5,
                    help="éxito en esa ventana para considerar el nivel aprendido (def. 0.5)")
    ap.add_argument("--sin-panel", action="store_true", help="no armar panel.png con todos los gráficos juntos")
    args = ap.parse_args(argv)
    try:
        pairs = parse_pairs(args.pares)
    except ValueError as e:
        sys.exit(f"error: {e}")

    run_dirs = expand_runs(args.runs)
    if not run_dirs:
        sys.exit("error: no encontré ninguna corrida (carpeta con config.json)")
    PERFORMANCE["pasos_aprende"] = (f"pasos hasta ganar el {100 * args.umbral:g}% en entrenamiento", True)
    seeds = collect(run_dirs, args.fases, args.incompletas, args.ventana, args.umbral)
    if seeds.empty:
        sys.exit("error: ninguna seed terminada todavía (con --incompletas se incluyen las que siguen entrenando)")
    agg = aggregate(seeds)
    corr = correlations(agg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(out / "semillas.csv", index=False)
    agg.to_csv(out / "tabla.csv", index=False)
    corr.to_csv(out / "correlaciones.csv", index=False)
    md = markdown(agg, corr, args.fases)
    (out / "tabla.md").write_text(md)
    figs = make_plots(seeds, agg, pairs, out, panel=not args.sin_panel)
    print(md)
    print(f"Resultados en {out}: tabla.md, tabla.csv, semillas.csv, correlaciones.csv, "
          + ", ".join(p.name for p in figs))
    return {"seeds": seeds, "agg": agg, "corr": corr, "figs": figs}


if __name__ == "__main__":
    main()

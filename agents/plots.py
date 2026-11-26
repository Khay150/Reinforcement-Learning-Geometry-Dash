"""Gráficos de una o varias corridas de ``train.py`` (para informes y diapositivas). Lo usa ``scripts/plot.py``.

Lee sólo lo que cada corrida deja en disco (ver "Qué se guarda" en el README), así que sirve para corridas viejas,
corridas con una sola seed y corridas que todavía se están entrenando (monitores a medio escribir):

* ``<algo>/seed<N>/monitor/*.monitor.csv``  -> curvas de aprendizaje (retorno, éxito y progreso de los episodios)
* ``<algo>/seed<N>/eval_history.csv``       -> curva de la evaluación periódica (BestModelCallback)
* ``<algo>/seed<N>/eval_fases*.csv``        -> barras finales (modelo final / mejor modelo, entrenamiento / validación)
* ``<algo>/seed<N>/progresion.csv``         -> etapas de la progresión de niveles
* ``<algo>/seed<N>/jump_penalty.csv``       -> penalización por salto en el tiempo (sólo con --ramp)

Cada "serie" es un par (corrida, algoritmo); sus seeds se promedian y la banda va del mínimo al máximo entre seeds.
Lo que falta se saltea con un aviso: nunca se corta todo por un archivo que no está.
"""
from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                    # sin ventana: sólo se escriben PNGs (también anda por ssh / en tests)
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Estilo: letra grande para diapositivas, paleta apta para daltónicos (orden fijo: la serie i siempre tiene el color i),
# grilla tenue y sin bordes de más.
# ---------------------------------------------------------------------------------------------------------------------
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA_2, GRILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
ETAPAS = ["#f3f2ef", "#e6e5e0"]          # fondo de las etapas de la progresión (se alternan)
TIPOS = ("curvas", "eval", "barras", "niveles", "progresion", "penalizacion")

ESTILO = {
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "figure.titlesize": 19,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRILLA, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "lines.linewidth": 2.2, "legend.frameon": False, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "savefig.facecolor": FONDO, "savefig.dpi": 200, "figure.dpi": 100,
}


def _redondo(x: float) -> float:
    """Redondea a 2 cifras significativas (para textos: "media móvil de 12k pasos", no 11.9776k)."""
    return float(f"{x:.2g}") if x > 0 else x


def _pasos(x, _pos=None) -> str:
    """Formato del eje de pasos: 0, 50k, 1.5M."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    return f"{x / 1e3:g}k" if abs(x) >= 1e3 else f"{x:g}"


def _aviso(msg: str):
    print(f"[aviso] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Lectura de archivos (tolerante: filas cortadas o vacías se saltean)
# ---------------------------------------------------------------------------------------------------------------------
def leer_csv(path: Path) -> list[dict]:
    """Filas de un CSV como dicts. Saltea las incompletas (p. ej. la última línea de una corrida en curso)."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        first = f.readline()
        if not first.startswith("#"):          # los monitores de SB3 empiezan con una línea "#{json}"
            f.seek(0)
        rows = list(csv.DictReader(f))
    return [r for r in rows if None not in r.values() and None not in r]


def _num(v) -> float:
    """Texto -> float (True/False -> 1/0; vacío o roto -> nan)."""
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("True", "true"):
        return 1.0
    if s in ("False", "false"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def columna(rows: list[dict], name: str) -> np.ndarray:
    return np.array([_num(r.get(name)) for r in rows], dtype=float)


def leer_monitor(seed_dir: Path) -> dict[str, np.ndarray] | None:
    """Junta los CSV de monitor de todos los entornos de una seed en una sola historia de episodios.

    Cada entorno escribe su propio archivo con ``t`` = segundos desde su ``t_start``. Se ordena por tiempo absoluto
    (``t_start + t``) y los pasos acumulados son la suma de los largos (``l``) de los episodios terminados hasta ahí.
    Devuelve ``{"pasos", "retorno", "exito", "progreso", "nivel"}`` o None si no hay episodios.
    """
    eps = []
    for f in sorted((Path(seed_dir) / "monitor").glob("*.monitor.csv")):
        try:
            with open(f) as fh:
                head = fh.readline()
            t0 = float(json.loads(head[1:]).get("t_start", 0.0)) if head.startswith("#") else 0.0
        except (OSError, ValueError):
            t0 = 0.0
        for r in leer_csv(f):
            t, rew, ln = _num(r.get("t")), _num(r.get("r")), _num(r.get("l"))
            if np.isnan(t) or np.isnan(rew) or np.isnan(ln):
                continue
            eps.append((t0 + t, rew, ln, _num(r.get("success")), _num(r.get("progress")), r.get("level") or ""))
    if not eps:
        return None
    eps.sort(key=lambda e: e[0])
    _, rew, ln, suc, prog, lvl = zip(*eps)
    return {"pasos": np.cumsum(ln), "retorno": np.array(rew), "exito": np.array(suc), "progreso": np.array(prog),
            "nivel": np.array(lvl)}


def media_movil(pasos: np.ndarray, valores: np.ndarray, grilla: np.ndarray, ancho: float) -> np.ndarray:
    """Promedio de los episodios que terminaron en la ventana ``(x - ancho, x]`` para cada ``x`` de ``grilla``.

    Se suaviza por pasos (no por cantidad de episodios) para que algoritmos con distinta cantidad de entornos o
    episodios de distinto largo queden comparables. Donde no hay episodios (antes del primero o después del último:
    corrida en curso) da nan, así la curva no se inventa datos.
    """
    ok = ~np.isnan(valores)
    pasos, valores = pasos[ok], valores[ok]
    out = np.full(len(grilla), np.nan)
    if len(pasos) == 0:
        return out
    acum = np.concatenate([[0.0], np.cumsum(valores)])
    hi = np.searchsorted(pasos, grilla, side="right")
    lo = np.searchsorted(pasos, grilla - ancho, side="right")
    n = hi - lo
    dentro = (n > 0) & (grilla <= pasos[-1])
    out[dentro] = (acum[hi] - acum[lo])[dentro] / n[dentro]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Series: (corrida, algoritmo) con sus seeds
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Serie:
    etiqueta: str
    corrida: Path
    algo: str
    seeds: list[Path]
    config: dict = field(default_factory=dict)
    color: str = COLORES[0]

    def progresion(self) -> dict | None:
        return self.config.get("progression") or None


def _seed_num(p: Path) -> int:
    m = re.search(r"\d+", p.name)
    return int(m.group()) if m else 0


def buscar_series(corridas: list[str | Path], etiquetas: list[str] | None = None) -> list[Serie]:
    """Una serie por cada ``<corrida>/<algo>/`` que tenga carpetas ``seed*``.

    Si una corrida tiene varios algoritmos, la etiqueta queda ``"<etiqueta> · PPO"``; si se compara una sola corrida,
    alcanza con el algoritmo. Sin ``etiquetas`` se usa el nombre de la carpeta.
    """
    corridas = [Path(c) for c in corridas]
    if etiquetas and len(etiquetas) != len(corridas):
        raise ValueError(f"--labels tiene {len(etiquetas)} etiquetas pero hay {len(corridas)} corridas")
    series = []
    for i, run in enumerate(corridas):
        if not run.is_dir():
            _aviso(f"{run} no existe: la salteo")
            continue
        try:
            cfg = json.loads((run / "config.json").read_text())
        except (OSError, ValueError):
            cfg = {}
        algos = sorted(d for d in run.iterdir() if d.is_dir() and any(s.is_dir() for s in d.glob("seed*")))
        if not algos:
            _aviso(f"{run} no tiene carpetas <algo>/seed*/: la salteo")
            continue
        base = etiquetas[i] if etiquetas else run.name
        for d in algos:
            if len(algos) == 1:
                label = base if (etiquetas or len(corridas) > 1) else f"{d.name.upper()} ({run.name})"
            else:
                label = f"{base} · {d.name.upper()}" if (etiquetas or len(corridas) > 1) else d.name.upper()
            seeds = sorted((s for s in d.glob("seed*") if s.is_dir()), key=_seed_num)
            series.append(Serie(label, run, d.name, seeds, cfg))
    if len(series) > len(COLORES):
        _aviso(f"hay {len(series)} series y la paleta tiene {len(COLORES)} colores: se repiten. "
               f"Conviene partir la comparación en varias figuras")
    for i, s in enumerate(series):
        s.color = COLORES[i % len(COLORES)]
    return series


def nombre_salida(corridas: list[str | Path]) -> str:
    """Carpeta por defecto dentro de figuras/: el nombre de la corrida, o ``comparacion_<primera>_+N``."""
    nombres = [Path(c).name for c in corridas]
    return nombres[0] if len(nombres) == 1 else f"comparacion_{nombres[0]}_+{len(nombres) - 1}"


# ---------------------------------------------------------------------------------------------------------------------
# Ayudas de dibujo
# ---------------------------------------------------------------------------------------------------------------------
def _figura(nrows=1, ncols=1, size=(12, 6.2), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, layout="constrained", squeeze=False, **kw)
    return fig, axes


def _encabezado(fig, titulo: str, nota: str | None = None):
    """Título arriba (negrita, alineado a la izquierda) y, abajo, una nota gris con la letra chica de cómo se calculó.
    Se usan suptitle/supxlabel para que constrained_layout les reserve el lugar."""
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold")
    if nota:
        fig.supxlabel(nota, x=0.01, ha="left", fontsize=12, color=TINTA_2, linespacing=1.4)


def _leyenda(ax, handles, ncol=None) -> int:
    """Leyenda en una fila arriba del gráfico (debajo del título), sin marco. Devuelve cuántas filas ocupa."""
    if not handles:
        return 0
    largo = max(len(h.get_label()) for h in handles)
    ncol = ncol or max(1, min(len(handles), 5 if largo < 16 else 3 if largo < 30 else 2 if largo < 50 else 1))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.03), ncol=ncol, handlelength=1.8,
              columnspacing=1.6, borderaxespad=0, borderpad=0)
    return -(-len(handles) // ncol)


def _guardar(fig, out: Path, nombre: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / nombre
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}", flush=True)
    return path


def _eje_pasos(ax, xmax: float | None = None):
    ax.xaxis.set_major_formatter(FuncFormatter(_pasos))
    ax.set_xlabel("pasos de entrenamiento")
    ax.set_xlim(0, xmax)


def _agregar(curvas: list[np.ndarray]):
    """Media, mínimo, máximo y cantidad de seeds en cada punto (ignorando nan: seeds que todavía no llegaron)."""
    m = np.vstack(curvas)
    n = np.sum(~np.isnan(m), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # columnas sin ninguna seed: nan
        return np.nanmean(m, axis=0), np.nanmin(m, axis=0), np.nanmax(m, axis=0), n


def _cambios_de_etapa(serie: Serie, seed_dir: Path) -> list[tuple[float, int]]:
    """(paso, etapa) de cada cambio de etapa registrado en progresion.csv (incluye el inicio, etapa 0)."""
    rows = leer_csv(seed_dir / "progresion.csv")
    out = []
    for r in rows:
        ev = str(r.get("event", ""))
        if ev == "inicio" or ev.startswith("etapa"):
            st = _num(r.get("stage"))
            if not np.isnan(st) and (not out or int(st) != out[-1][1]):
                out.append((_num(r.get("timesteps")), int(st)))
    return out


def _sombrear_etapas(ax, cambios: list[tuple[float, int]], xmax: float, nombres: list[str] | None = None,
                     rotulo=True):
    """Fondo alternado por etapa + rótulo "etapa k" arriba a la izquierda de cada región."""
    for i, (x0, st) in enumerate(cambios):
        x1 = cambios[i + 1][0] if i + 1 < len(cambios) else xmax
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=ETAPAS[st % 2], zorder=0, lw=0)
        if i > 0:
            ax.axvline(x0, color=TINTA_2, lw=1, ls=(0, (3, 3)), zorder=1)
        if rotulo:
            txt = f"etapa {st + 1}"
            if nombres and st < len(nombres):
                txt += f"\n{nombres[st]}"
            ax.text(x0 + 0.01 * xmax, 0.98, txt, transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=11, color=TINTA_2, zorder=5, clip_on=True)


def _nombres_etapas(cfg: dict) -> list[str] | None:
    p = cfg.get("progression") or {}
    return [", ".join(st) for st in p.get("etapas", [])] or None


# ---------------------------------------------------------------------------------------------------------------------
# 1. Curvas de aprendizaje (monitor)
# ---------------------------------------------------------------------------------------------------------------------
METRICAS_MONITOR = {
    "retorno": ("Retorno por episodio (entrenamiento)", "retorno", None),
    "exito": ("Episodios ganados (entrenamiento)", "episodios ganados", 100),
    "progreso": ("Progreso por episodio (entrenamiento)", "progreso en el nivel", 100),
}


def curvas(series: list[Serie], out: Path, suavizado: float = 0.04, puntos: int = 400) -> list[Path]:
    """``curva_retorno.png``, ``curva_exito.png`` y ``curva_progreso.png``: media entre seeds + banda mín–máx.

    Con una sola serie que tenga progresión de niveles, se sombrean detrás las etapas (paso promedio entre seeds).
    """
    datos = {}                                                    # serie -> lista de monitores (uno por seed)
    for s in series:
        mons = [m for m in (leer_monitor(d) for d in s.seeds) if m is not None]
        if mons:
            datos[s.etiqueta] = (s, mons)
        else:
            _aviso(f"{s.etiqueta}: no hay episodios en monitor/ (¿todavía no arrancó?)")
    if not datos:
        return []
    xmax = max(m["pasos"][-1] for _, mons in datos.values() for m in mons)
    grilla = np.linspace(0, xmax, puntos)[1:]
    ancho = _redondo(suavizado * xmax)
    seeds_txt = sorted({len(m) for _, m in datos.values()})
    sub = (f"media móvil de {_pasos(ancho)} pasos · línea: media entre seeds"
           + (" · banda: mín–máx entre seeds" if max(seeds_txt) > 1 else "")
           + f" · seeds por serie: {'/'.join(map(str, seeds_txt))}")
    if any(len(mons) > 1 and min(m["pasos"][-1] for m in mons) < 0.97 * max(m["pasos"][-1] for m in mons)
           for _, mons in datos.values()):
        sub += "\nseeds de distinto largo (en curso): donde termina la banda, la línea sigue sólo con las que llegaron"

    cambios_prom = None
    if len(datos) == 1:
        s, mons = next(iter(datos.values()))
        if s.progresion():
            por_seed = [_cambios_de_etapa(s, d) for d in s.seeds]
            por_seed = [c for c in por_seed if c]
            if por_seed:                                          # paso promedio de cada etapa entre las seeds
                etapas = sorted({st for c in por_seed for _, st in c})
                cambios_prom = [(float(np.mean([x for c in por_seed for x, st2 in c if st2 == st])), st)
                                for st in etapas]
                if len(por_seed) > 1:
                    sub += " · etapas: paso promedio entre seeds"

    paths = []
    for clave, (titulo, ylabel, escala) in METRICAS_MONITOR.items():
        fig, axes = _figura()
        ax = axes[0, 0]
        handles, cortadas = [], []
        for s, mons in datos.values():
            cs = [media_movil(m["pasos"], m[clave], grilla, ancho) for m in mons]
            if all(np.all(np.isnan(c)) for c in cs):
                continue
            mean, lo, hi, n = _agregar(cs)
            k = escala or 1
            if len(mons) > 1:
                band = n >= 2
                ax.fill_between(grilla, lo * k, hi * k, where=band, color=s.color, alpha=0.18, lw=0, zorder=2)
            (ln,) = ax.plot(grilla, mean * k, color=s.color, label=s.etiqueta, zorder=3)
            handles.append(ln)
            ult = np.flatnonzero(~np.isnan(mean))
            if len(ult) and grilla[ult[-1]] < 0.97 * xmax:    # termina antes que las demás: corrida en curso o cortada
                ax.plot(grilla[ult[-1]], mean[ult[-1]] * k, "o", ms=8, color=s.color, mec=FONDO, mew=1.5, zorder=4)
                cortadas.append(s.etiqueta)
        if not handles:
            plt.close(fig)
            _aviso(f"curva de {clave}: ninguna corrida registra esa columna en monitor/")
            continue
        if cambios_prom:
            s = next(iter(datos.values()))[0]
            _sombrear_etapas(ax, cambios_prom, xmax, _nombres_etapas(s.config))
        if escala:
            ax.set_ylim(-2, 102)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel(ylabel)
        _eje_pasos(ax, xmax)
        nota = sub + (f"\n● = la serie termina ahí (corrida en curso o cortada): {', '.join(cortadas)}"
                      if cortadas else "")
        _encabezado(fig, titulo, nota)
        _leyenda(axes[0, 0], handles)
        paths.append(_guardar(fig, out, f"curva_{clave}.png"))
    return paths


# ---------------------------------------------------------------------------------------------------------------------
# 1b. Evaluación periódica (eval_history.csv)
# ---------------------------------------------------------------------------------------------------------------------
def _curva_eval(rows: list[dict], col: str, grilla: np.ndarray) -> np.ndarray:
    x, y = columna(rows, "timesteps"), columna(rows, col)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    out = np.full(len(grilla), np.nan)
    if len(x) == 0:
        return out
    orden = np.argsort(x)
    x, y = x[orden], y[orden]
    dentro = (grilla >= x[0]) & (grilla <= x[-1])
    out[dentro] = np.interp(grilla[dentro], x, y)
    return out


def evaluacion(series: list[Serie], out: Path) -> list[Path]:
    """``eval_exito.png`` y ``eval_progreso.png``: la evaluación determinista en las 10 fases que hace
    BestModelCallback cada ``--eval-freq`` pasos. Si hay niveles de validación, van en un segundo panel."""
    hist = {s.etiqueta: (s, [r for r in (leer_csv(d / "eval_history.csv") for d in s.seeds) if r]) for s in series}
    hist = {k: v for k, v in hist.items() if v[1]}
    if not hist:
        _aviso("ninguna corrida tiene eval_history.csv (se entrenaron sin --eval-freq o son anteriores): no hay "
               "curva de evaluación")
        return []
    xmax = max(np.nanmax(columna(r, "timesteps")) for _, rs in hist.values() for r in rs)
    hay_val = any("val_success" in r[0] for _, rs in hist.values() for r in rs)
    no_reservados = all(s.config.get("val_held_out") is False for s, _ in hist.values())
    val_txt = "niveles de validación" + (" (no reservados: también se entrena en ellos)" if no_reservados else "")
    paneles = [("", "niveles de entrenamiento")] + ([("val_", val_txt)] if hay_val else [])
    nseeds = sorted({len(rs) for _, rs in hist.values()})
    nota = ("modelo determinista en las 10 fases de arranque · un punto por evaluación · línea: media entre seeds"
            + (" · banda: mín–máx entre seeds" if max(nseeds) > 1 else "")
            + f" · seeds por serie: {'/'.join(map(str, nseeds))}")
    paths = []
    for clave, col, titulo, ylabel in (
            ("exito", "success", "Fases ganadas en la evaluación periódica", "fases ganadas"),
            ("progreso", "progress", "Progreso en la evaluación periódica", "progreso medio")):
        fig, axes = _figura(1, len(paneles), size=(12 if len(paneles) == 1 else 15, 6.2), sharey=True)
        handles = []
        for j, (pref, nombre) in enumerate(paneles):
            ax = axes[0, j]
            for s, rs in hist.values():
                rs = [r for r in rs if pref + col in r[0]]
                if not rs:
                    continue
                # los pasos de evaluación de esta serie (todas sus seeds); una seed en curso aporta menos puntos
                pasos = np.unique(np.concatenate([columna(r, "timesteps") for r in rs]))
                pasos = pasos[~np.isnan(pasos)]
                mean, lo, hi, n = _agregar([_curva_eval(r, pref + col, pasos) for r in rs])
                if len(rs) > 1:
                    ax.fill_between(pasos, lo * 100, hi * 100, where=n >= 2, color=s.color, alpha=0.16, lw=0,
                                    zorder=2)
                (ln,) = ax.plot(pasos, mean * 100, color=s.color, marker="o", ms=6, mec=FONDO, mew=1.2, zorder=3,
                                label=s.etiqueta)
                if j == 0:
                    handles.append(ln)
            ax.set_ylim(-2, 102)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            _eje_pasos(ax, xmax * 1.02)
        axes[0, 0].set_ylabel(ylabel)
        _encabezado(fig, titulo, nota)
        filas = _leyenda(axes[0, 0], handles if len(handles) > 1 else [])
        if len(paneles) > 1:                  # nombre de cada panel, por encima de la leyenda
            for ax, (_, nombre) in zip(axes[0], paneles):
                ax.set_title(nombre, loc="left", fontsize=15, color=TINTA_2, pad=8 + 24 * filas)
        paths.append(_guardar(fig, out, f"eval_{clave}.png"))
    return paths


# ---------------------------------------------------------------------------------------------------------------------
# 2. Barras finales (eval_fases*.csv)
# ---------------------------------------------------------------------------------------------------------------------
ARCHIVOS_EVAL = {("final", "train"): "eval_fases.csv", ("mejor", "train"): "eval_fases_best.csv",
                 ("final", "val"): "eval_fases_val.csv", ("mejor", "val"): "eval_fases_best_val.csv"}
MODELOS = {"final": "modelo final", "mejor": "mejor modelo"}


def metricas_eval(rows: list[dict]) -> dict:
    """Las métricas de una evaluación por fases (como ``agents.evaluate.summarize``): éxito, progreso medio y
    saltos / oráculo promediado en las fases ganadas (nan si no ganó ninguna)."""
    suc, prog = columna(rows, "success"), columna(rows, "progress")
    nj, oj = columna(rows, "n_jumps"), columna(rows, "oracle_jumps")
    won = (suc == 1) & (oj > 0)
    return {"exito": float(np.nanmean(suc)) if len(rows) else np.nan,
            "progreso": float(np.nanmean(prog)) if len(rows) else np.nan,
            "saltos": float(np.mean(nj[won] / oj[won])) if won.any() else np.nan,
            "saltos_fases": list(nj[won] / oj[won])}


def _tinte(color: str, f: float = 0.55) -> tuple:
    """El mismo color, mezclado con blanco (para el modelo final; el mejor modelo va con el color pleno)."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * f)


def barras(series: list[Serie], out: Path) -> list[Path]:
    """``barras_entrenamiento.png`` y (si hay) ``barras_validacion.png``: éxito, progreso y saltos/oráculo por serie,
    modelo final (claro) y mejor modelo (pleno). Barra = media entre seeds, bigote = ± desvío, puntos = cada seed."""
    paths = []
    for conjunto, nombre, titulo in (("train", "entrenamiento", "Resultado final en los niveles de entrenamiento"),
                                     ("val", "validacion", "Resultado final en los niveles de validación")):
        tabla = {}                    # (serie, modelo) -> {métrica: [valor por seed]}
        for s in series:
            for modelo in MODELOS:
                ms = [metricas_eval(r) for r in (leer_csv(d / ARCHIVOS_EVAL[modelo, conjunto]) for d in s.seeds) if r]
                if ms:
                    tabla[s.etiqueta, modelo] = {k: [m[k] for m in ms] for k in ms[0]}
        if not tabla:
            if conjunto == "train":
                _aviso("ninguna seed tiene eval_fases.csv (¿corridas sin terminar?): no hay barras")
            continue
        filas = [s for s in series if any((s.etiqueta, m) in tabla for m in MODELOS)]
        modelos = [m for m in MODELOS if any((s.etiqueta, m) in tabla for s in filas)]
        alto = max(3.8, 1.1 + 0.62 * len(filas) * len(modelos) + 0.35 * len(filas))
        fig, axes = _figura(1, 3, size=(15, alto), sharey=True)
        h = 0.8 / len(modelos)
        for j, (met, xlabel, escala, xlim) in enumerate((("exito", "fases ganadas", 100, 100),
                                                        ("progreso", "progreso medio", 100, 100),
                                                        ("saltos", "saltos / oráculo (fases ganadas)", 1, None))):
            ax = axes[0, j]
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            tope = 0.0
            for i, s in enumerate(filas):
                for k, modelo in enumerate(modelos):
                    vals = np.array(tabla.get((s.etiqueta, modelo), {}).get(met, []), dtype=float) * escala
                    vals = vals[~np.isnan(vals)]
                    y = i + (k - (len(modelos) - 1) / 2) * h
                    if len(vals) == 0:
                        if (s.etiqueta, modelo) in tabla:
                            ax.text(0.01, y, "sin fases ganadas" if met == "saltos" else "—",
                                    transform=ax.get_yaxis_transform(), va="center", fontsize=11, color=TINTA_2)
                        continue
                    media, desv = vals.mean(), vals.std()
                    if met == "saltos":    # como resumen.md: promedio de todas las fases ganadas de todas las seeds
                        media = float(np.mean(sum(tabla[s.etiqueta, modelo]["saltos_fases"], [])))
                    claro = modelo == "final" and len(modelos) > 1
                    ax.barh(y, media, height=h * 0.9, color=_tinte(s.color) if claro else s.color,
                            zorder=2)
                    if len(vals) > 1:
                        ax.errorbar(media, y, xerr=desv, color=TINTA, capsize=4, lw=1.4, zorder=4)
                        ax.scatter(vals, np.full(len(vals), y), s=16, color=TINTA, zorder=5, alpha=0.7, lw=0)
                    fin = max(media + desv, vals.max())
                    tope = max(tope, fin)
                    ax.text(fin, y, f"  {media:.0f}%" if escala == 100 else f"  {media:.2f}", va="center",
                            ha="left", fontsize=12, color=TINTA, zorder=6)
            if met == "saltos":
                ax.axvline(1, color=TINTA_2, lw=1.2, ls=(0, (3, 3)), zorder=3)
                ax.text(1, 1.0, " óptimo ", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=11, color=TINTA_2)
                ax.set_xlim(0, max(tope * 1.25, 1.3))
            else:
                ax.set_xlim(0, xlim * 1.18)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_xlabel(xlabel)
            ax.spines["left"].set_visible(j == 0)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].set_yticks(range(len(filas)), [s.etiqueta for s in filas])
        axes[0, 0].set_ylim(len(filas) - 0.5, -0.5)            # la primera corrida arriba
        nseeds = sorted({len(v["exito"]) for v in tabla.values()})
        niveles = _niveles_txt(filas, conjunto)
        sub = (f"{niveles}" + (f"{MODELOS[modelos[0]]} · " if len(modelos) == 1 else "") + "media entre seeds"
               + (" · bigote: ± desvío · puntos: cada seed" if max(nseeds) > 1 else "")
               + f" · seeds: {'/'.join(map(str, nseeds))}"
               + "\nsaltos / oráculo: promedio sobre todas las fases ganadas (como resumen.md); "
                 "1 = tan pocos como el óptimo")
        _encabezado(fig, titulo, sub)
        if len(modelos) > 1:
            _leyenda(axes[0, 0], [Patch(color=_tinte(TINTA_2) if m == "final" else TINTA_2, label=MODELOS[m])
                                  for m in modelos])
        paths.append(_guardar(fig, out, f"barras_{nombre}.png"))
    return paths


def _niveles_txt(series: list[Serie], conjunto: str) -> str:
    """"niveles: a, b · " si todas las series usan los mismos niveles (si no, se ven en barras_niveles.png)."""
    key = "train_levels" if conjunto == "train" else "val_levels"
    sets = {tuple(s.config.get(key) or (["level_1"] if conjunto == "train" else [])) for s in series}
    if len(sets) == 1 and next(iter(sets)):
        lv = next(iter(sets))
        return f"niveles: {', '.join(lv)} · " if len(lv) <= 4 else f"{len(lv)} niveles · "
    return ""


def barras_niveles(series: list[Serie], out: Path) -> list[Path]:
    """``barras_niveles.png``: fases ganadas en cada nivel (entrenamiento y validación) por serie. Usa el mejor
    modelo; la serie que no lo tiene usa el final (y lo dice en la etiqueta). Sólo si alguna serie entrena en
    más de un nivel."""
    datos, niveles = {}, {}            # serie -> {(conjunto, nivel): [éxito por seed]};  (conjunto, nivel) en orden
    for s in series:
        tiene_mejor = any((d / ARCHIVOS_EVAL["mejor", "train"]).is_file() for d in s.seeds)
        modelo = "mejor" if tiene_mejor else "final"
        d_s = {}
        for conjunto in ("train", "val"):
            for d in s.seeds:
                rows = leer_csv(d / ARCHIVOS_EVAL[modelo, conjunto])
                for lv in dict.fromkeys(r.get("level", "level_1") for r in rows):
                    suc = columna([r for r in rows if r.get("level", "level_1") == lv], "success")
                    d_s.setdefault((conjunto, lv), []).append(float(np.nanmean(suc)))
                    niveles.setdefault((conjunto, lv), None)
        if d_s:
            datos[s.etiqueta] = (s, modelo, d_s)
    if max((sum(c == "train" for c, _ in d_s) for _, _, d_s in datos.values()), default=0) < 2:
        return []                    # un solo nivel de entrenamiento: ya está todo en barras_entrenamiento/validacion
    niveles = sorted(niveles, key=lambda k: k[0] != "train")          # primero entrenamiento, después validación
    modelos = {m for _, m, _ in datos.values()}
    n = len(datos)
    h = 0.8 / n
    fig, axes = _figura(size=(12, max(4.0, 1.6 + 0.42 * n * len(niveles))))
    ax = axes[0, 0]
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    handles = []
    for k, (s, modelo, d_s) in enumerate(datos.values()):
        for i, key in enumerate(niveles):
            vals = np.array(d_s.get(key, []), dtype=float) * 100
            if len(vals) == 0:
                continue
            y = i + (k - (n - 1) / 2) * h
            ax.barh(y, vals.mean(), height=h * 0.9, color=s.color, zorder=2)
            if len(vals) > 1:
                ax.errorbar(vals.mean(), y, xerr=vals.std(), color=TINTA, capsize=3, lw=1.2, zorder=4)
                ax.scatter(vals, np.full(len(vals), y), s=12, color=TINTA, zorder=5, alpha=0.7, lw=0)
            if n <= 3:
                ax.text(max(vals.mean() + vals.std(), vals.max()), y, f"  {vals.mean():.0f}%", va="center",
                        fontsize=11, color=TINTA)
        lab = s.etiqueta + (" (modelo final)" if modelo == "final" and len(modelos) > 1 else "")
        handles.append(Patch(color=s.color, label=lab))
    etiquetas = [lv + ("  (valid.)" if c == "val" else "") for c, lv in niveles]
    ax.set_yticks(range(len(niveles)), etiquetas)
    ax.set_ylim(len(niveles) - 0.5, -0.5)
    corte = sum(c == "train" for c, _ in niveles)
    if 0 < corte < len(niveles):
        ax.axhline(corte - 0.5, color=TINTA_2, lw=1)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("fases ganadas (de 10)")
    ax.tick_params(axis="y", length=0)
    modelo_txt = "mejor modelo" if modelos == {"mejor"} else "modelo final" if modelos == {"final"} else \
        "mejor modelo (o final si no hay)"
    nseeds = sorted({len(v) for _, _, d_s in datos.values() for v in d_s.values()})
    _encabezado(fig, "Fases ganadas en cada nivel", f"{modelo_txt} · media entre seeds"
                + (" · bigote: ± desvío · puntos: cada seed" if max(nseeds) > 1 else ""))
    if n > 1:
        _leyenda(axes[0, 0], handles)
    return [_guardar(fig, out, "barras_niveles.png")]


# ---------------------------------------------------------------------------------------------------------------------
# 3. Progresión de niveles (progresion.csv)
# ---------------------------------------------------------------------------------------------------------------------
def progresion(serie: Serie, out: Path, slug: str) -> list[Path]:
    """``progresion_<slug>.png``: una fila por seed. Izquierda: éxito reciente de cada nivel (la ventana que usa
    LevelProgression para desbloquear) con el umbral; derecha: probabilidad de que salga cada nivel. Fondo: etapas."""
    seeds = [(d, leer_csv(d / "progresion.csv")) for d in serie.seeds]
    seeds = [(d, r) for d, r in seeds if r]
    if not seeds:
        return []
    cfg = serie.progresion() or {}
    niveles = [c[2:] for c in seeds[0][1][0] if c.startswith("w_")]
    colores = {lv: COLORES[i % len(COLORES)] for i, lv in enumerate(niveles)}
    nombres = _nombres_etapas(serie.config)
    xmax = 0.0
    for d, rows in seeds:
        mon = leer_monitor(d)
        xmax = max(xmax, np.nanmax(columna(rows, "timesteps")), mon["pasos"][-1] if mon else 0)
    fig, axes = _figura(len(seeds), 2, size=(15, 1.4 + 3.4 * len(seeds)), sharex=True,
                        gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (d, rows) in enumerate(seeds):
        x = columna(rows, "timesteps")
        cambios = _cambios_de_etapa(serie, d)
        pesos = np.vstack([columna(rows, f"w_{lv}") for lv in niveles])
        total = pesos.sum(axis=0)
        prob = np.divide(pesos, total, out=np.zeros_like(pesos), where=total > 0)
        ax, ax2 = axes[i, 0], axes[i, 1]
        for a in (ax, ax2):
            _sombrear_etapas(a, cambios, xmax, None, rotulo=False)
        for j, lv in enumerate(niveles):
            suc = columna(rows, f"success_{lv}") * 100
            suc[pesos[j] == 0] = np.nan               # nivel bloqueado: todavía no se juega, no es "0% de éxito"
            # se extiende hasta el final del entrenamiento con el último valor (el CSV registra cada N episodios)
            ax.step(np.append(x, xmax), np.append(suc, suc[-1]), where="post", color=colores[lv], label=lv, lw=2)
        if cfg.get("desbloquear") is not None:
            ax.axhline(cfg["desbloquear"] * 100, color=TINTA_2, lw=1.2, ls=(0, (3, 3)))
            ax.text(0.01 * xmax, cfg["desbloquear"] * 100, "umbral", ha="left", va="bottom", fontsize=11,
                    color=TINTA_2)
        ax2.stackplot(np.append(x, xmax), np.hstack([prob, prob[:, -1:]]), step="post",
                      colors=[colores[lv] for lv in niveles], edgecolor=FONDO, lw=0.8, zorder=2)
        for st_i, (x0, st) in enumerate(cambios):
            x1 = cambios[st_i + 1][0] if st_i + 1 < len(cambios) else xmax
            ax.text((x0 + x1) / 2, 1.0, f"etapa {st + 1}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=11, color=TINTA_2)
        ax.set_ylim(-2, 102)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel(f"seed {_seed_num(d)}\néxito reciente")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax2.set_ylabel("prob. de salir")
        ax2.grid(visible=False)
        for a in (ax, ax2):
            a.set_xlim(0, xmax)
            a.xaxis.set_major_formatter(FuncFormatter(_pasos))
    for a in axes[-1]:
        a.set_xlabel("pasos de entrenamiento")
    win = cfg.get("ventana")
    sub = (f"izq.: éxito en los últimos {win} episodios de cada nivel (con exploración); bloqueado = sin línea · "
           f"der.: con qué probabilidad sale cada nivel")
    if nombres:
        sub += "\n" + " → ".join(f"etapa {k + 1}: {n}" for k, n in enumerate(nombres))
    _encabezado(fig, f"Progresión de niveles · {serie.etiqueta}", sub)
    _leyenda(axes[0, 1], [Line2D([], [], color=colores[lv], lw=3, label=lv) for lv in niveles], ncol=2)
    return [_guardar(fig, out, f"progresion_{slug}.png")]


# ---------------------------------------------------------------------------------------------------------------------
# 4. Penalización por salto (jump_penalty.csv, sólo con --ramp)
# ---------------------------------------------------------------------------------------------------------------------
def penalizacion(series: list[Serie], out: Path) -> list[Path]:
    """``penalizacion_salto.png``: el valor de ``c`` en el tiempo (escalones), una línea por seed, color por serie."""
    datos = []
    for s in series:
        for d in s.seeds:
            rows = leer_csv(d / "jump_penalty.csv")
            if rows:
                mon = leer_monitor(d)
                datos.append((s, d, rows, mon["pasos"][-1] if mon else np.nanmax(columna(rows, "timesteps"))))
    if not datos:
        return []
    xmax = max(x for *_, x in datos)
    fig, axes = _figura()
    ax = axes[0, 0]
    handles, vistas = [], set()
    for s, d, rows, fin in datos:
        x, c = columna(rows, "timesteps"), columna(rows, "jump_penalty")
        (ln,) = ax.step(np.append(x, fin), np.append(c, c[-1]), where="post", color=s.color, lw=2, alpha=0.85,
                        label=s.etiqueta)
        if s.etiqueta not in vistas:
            vistas.add(s.etiqueta)
            handles.append(ln)
    maximos = {s.config.get("jump_penalty") for s, *_ in datos}
    cmax = maximos.pop() if len(maximos) == 1 else None          # una sola línea de "máximo" si todas comparten c_max
    if cmax:
        ax.axhline(cmax, color=TINTA_2, lw=1.2, ls=(0, (3, 3)))
        ax.text(xmax, cmax, "máximo  ", ha="right", va="bottom", fontsize=11, color=TINTA_2)
    ax.set_ylabel("penalización por salto (c)")
    ax.set_ylim(bottom=0)
    _eje_pasos(ax, xmax)
    _encabezado(fig, "Penalización por salto con --ramp",
                "c sube cuando el agente gana seguido y baja cuando pierde · una línea por seed")
    _leyenda(axes[0, 0], handles if len(handles) > 1 else [])
    return [_guardar(fig, out, "penalizacion_salto.png")]


# ---------------------------------------------------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------------------------------------------------
def _slug(texto: str) -> str:
    return re.sub(r"[^\w.+-]+", "_", texto).strip("_")


def graficar(corridas: list[str | Path], out: str | Path | None = None, etiquetas: list[str] | None = None,
             tipos=TIPOS, suavizado: float = 0.04) -> list[Path]:
    """Hace todas las figuras pedidas en ``tipos`` (ver ``TIPOS``) y devuelve las rutas de los PNG escritos."""
    tipos = list(tipos)
    malos = set(tipos) - set(TIPOS)
    if malos:
        raise ValueError(f"tipos desconocidos: {sorted(malos)} (opciones: {', '.join(TIPOS)})")
    series = buscar_series(corridas, etiquetas)
    if not series:
        raise FileNotFoundError("no encontré ninguna corrida con <algo>/seed*/ adentro")
    out = Path(out) if out else Path("figuras") / nombre_salida(corridas)
    paths = []
    with plt.rc_context(ESTILO):
        if "curvas" in tipos:
            paths += curvas(series, out, suavizado)
        if "eval" in tipos:
            paths += evaluacion(series, out)
        if "barras" in tipos:
            paths += barras(series, out)
        if "niveles" in tipos:
            paths += barras_niveles(series, out)
        if "progresion" in tipos:
            con_prog = [s for s in series if any((d / "progresion.csv").is_file() for d in s.seeds)]
            for s in con_prog:
                paths += progresion(s, out, _slug(f"{s.corrida.name}_{s.algo}"))
        if "penalizacion" in tipos:
            paths += penalizacion(series, out)
    return paths

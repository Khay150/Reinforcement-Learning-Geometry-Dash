"""Dificultad de un nivel medida con la física (sin entrenar ningún agente).

Mide lo que acordamos como dificultad: **qué tan preciso tiene que ser cada salto** y **cuántos saltos hay que
encadenar**. Como la física es determinista y x depende sólo del tick, se enumeran *todos* los estados alcanzables
desde el arranque (un DAG por capas de tick, como en el oráculo) y, de atrás para adelante, se calcula para cada uno:

* ``J(s)``: mínima cantidad de saltos para ganar desde ``s`` (infinito si ya no se puede ganar);
* ``P(s)``: probabilidad de ganar desde ``s`` apretando al azar (50% saltar / 50% no, en cada tick).

Con eso se recorre un camino óptimo salto por salto (despegando en el centro de cada ventana) y se mide:

* **ventana** de un salto: cuántos ticks de despegue distintos permiten seguir ganando *sin gastar saltos de más*.
  Ventana chica = salto preciso. Un tick = ``vx`` = 0.2 tiles de avance.
* **trampas**: ticks de despegue desde los que el cubo aterriza vivo pero el nivel ya no se puede ganar. Es el caso
  "el primer salto determina los siguientes": el error no se ve al aterrizar sino más adelante.
* **encadenado**: el salto ``j+1`` depende del ``j`` si cambiar cuándo se despega en ``j`` (dentro de su ventana)
  cambia la ventana del ``j+1``. ``max_chain`` = la secuencia más larga de saltos dependientes entre sí.
* **P(ganar al azar)**: resumen global de qué tan difícil es dar con la solución explorando (lo que le cuesta a un
  agente de RL al principio del entrenamiento). Se reporta como ``log10``: -3 = uno de cada mil episodios.

Uso::

    from gd_env import get_level, level_difficulty
    d = level_difficulty(get_level("level_1"))      # en la fase por defecto (x0 = level.start_x)
    print(d.summary())                              # dict con las métricas resumidas
    for j in d.jumps: print(j)                      # detalle salto por salto

Desde la línea de comandos: ``python scripts/difficulty.py`` (ver ese archivo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import physics as ph
from .physics import Level, Physics, State

INF = math.inf


class _StateGraph:
    """Todos los estados alcanzables desde ``start`` (deduplicados como en el oráculo) con sus transiciones, y para
    cada uno ``J`` (saltos mínimos para ganar) y ``P`` (probabilidad de ganar apretando al azar)."""

    def __init__(self, level: Level, p: Physics, start: State):
        self.states: dict[tuple, State] = {}
        # key -> {acción: (key siguiente o None si el episodio terminó, status)}. En el aire sólo existe la acción 0
        # (saltar no hace nada), así que no se duplica el estado.
        self.next: dict[tuple, dict[int, tuple[tuple | None, str]]] = {}
        max_ticks = int(np.ceil((level.goal_x - start.x) / p.vx)) + 2     # x avanza siempre: el nivel termina antes
        layers, layer = [], {start.key(): start}
        for _ in range(max_ticks):
            layers.append(layer)
            nxt: dict[tuple, State] = {}
            for k, s in layer.items():
                self.states[k] = s
                out = {}
                for a in ((0, 1) if s.on_ground else (0,)):
                    ns, status = ph.step(level, p, s, a)
                    if status == ph.OK:
                        nk = ns.key()
                        nxt.setdefault(nk, ns)
                        out[a] = (nk, status)
                    else:
                        out[a] = (None, status)         # murió o ganó: terminal
                self.next[k] = out
            if not nxt:
                break
            layer = nxt
        else:
            raise RuntimeError("el episodio no termina: ¿la meta está fuera del nivel?")

        # De atrás para adelante (la última capa ya no tiene sucesores vivos).
        self.J: dict[tuple, float] = {}
        self.P: dict[tuple, float] = {}
        for layer in reversed(layers):
            for k in layer:
                js, ps = [], []
                for a in self.next[k]:
                    j, pr = self._value(k, a)
                    js.append(j + a)                    # a == 1 sólo existe apoyado: es un salto efectivo
                    ps.append(pr)
                self.J[k] = min(js)
                self.P[k] = sum(ps) / len(ps)

    def _value(self, k: tuple, a: int) -> tuple[float, float]:
        """(J, P) del estado al que se llega desde ``k`` con la acción ``a`` (terminal: ganar = (0, 1), morir = (inf, 0))."""
        nk, status = self.next[k][a]
        if nk is None:
            return (0, 1.0) if status == ph.WIN else (INF, 0.0)
        return self.J[nk], self.P[nk]

    def j_after(self, k: tuple, a: int) -> float:
        return self._value(k, a)[0]

    def after_jump(self, k: tuple) -> tuple | None:
        """Estado siguiente a saltar desde ``k`` (None si el salto termina el episodio en ese mismo tick)."""
        return self.next[k][1][0]

    def landing(self, k: tuple | None) -> tuple[tuple | None, str]:
        """Sigue la caída libre desde ``k`` hasta que el cubo vuelve a estar apoyado o termina el episodio.
        Devuelve (key del estado apoyado o None, status final)."""
        status = ph.OK
        while k is not None and not self.states[k].on_ground:
            k, status = self.next[k][0]
        return k, status


@dataclass
class JumpInfo:
    """Un salto del camino óptimo analizado. Los ticks son absolutos (desde el arranque)."""
    tick: int                        # tick de despegue elegido (centro de la ventana)
    x: float                         # x de despegue elegida
    first_tick: int                  # primer tick de la ventana
    last_tick: int                   # último tick de la ventana
    x_from: float                    # x de despegue del primer tick de la ventana
    x_to: float                      # x de despegue del último tick de la ventana
    traps: int                       # ticks de despegue que aterrizan vivos pero dejan el nivel imposible
    linked_next: bool = False        # cambiar cuándo se despega acá (dentro de la ventana) cambia la ventana del próximo
    next_windows: tuple[int, int] = (0, 0)   # (mín, máx) de la ventana del próximo salto según cuándo se despegue acá

    @property
    def window(self) -> int:
        """Ancho de la ventana en ticks: 1 = hay un único tick posible para despegar."""
        return self.last_tick - self.first_tick + 1


@dataclass
class Difficulty:
    """Métricas de dificultad de un nivel desde una fase de arranque ``x0``."""
    level: str
    x0: float
    n_jumps: int                         # saltos mínimos para ganar (lo mismo que el oráculo)
    p_random_win: float                  # probabilidad de ganar apretando al azar (50/50 en cada tick)
    jumps: list[JumpInfo] = field(default_factory=list)

    @property
    def windows(self) -> list[int]:
        return [j.window for j in self.jumps]

    @property
    def chains(self) -> list[int]:
        """Largo de cada cadena de saltos dependientes (en saltos). Un salto independiente es una cadena de 1."""
        out, run = [], 0
        for j in self.jumps:
            run += 1
            if not j.linked_next:
                out.append(run)
                run = 0
        return out

    @property
    def max_chain(self) -> int:
        return max(self.chains, default=0)

    @property
    def log10_p_random(self) -> float:
        return math.log10(self.p_random_win) if self.p_random_win > 0 else -INF

    def summary(self) -> dict:
        """Métricas resumidas (una fila de la tabla de ``scripts/difficulty.py``)."""
        w = self.windows
        return {
            "level": self.level, "x0": round(self.x0, 3), "n_jumps": self.n_jumps,
            "min_window": min(w, default=None),                         # el salto más preciso
            "mean_window": round(float(np.mean(w)), 2) if w else None,
            "jumps_with_traps": sum(j.traps > 0 for j in self.jumps),
            "trap_ticks": sum(j.traps for j in self.jumps),
            "max_chain": self.max_chain,
            "linked_jumps": sum(c for c in self.chains if c > 1),       # saltos que forman parte de alguna cadena
            "log10_p_random": round(self.log10_p_random, 2),
        }


def _scan_window(g: _StateGraph, k: tuple) -> list[tuple]:
    """Desde ``k`` (con J(k) >= 1) avanza sin saltar mientras esperar siga siendo óptimo y devuelve la primera racha
    de estados apoyados desde los que saltar es óptimo (la ventana del próximo salto)."""
    jc, window, cur = g.J[k], [], k
    while True:
        if g.states[cur].on_ground and 1 + g.j_after(cur, 1) == jc:
            window.append(cur)
        elif window:
            break                                   # terminó la primera racha
        nk, _ = g.next[cur][0]
        if nk is None or g.J[nk] != jc:             # esperar ya no es óptimo (o mata): hay que haber saltado
            break
        cur = nk
    return window


def _count_traps(g: _StateGraph, k: tuple, last_tick: int) -> int:
    """Ticks de despegue desde los que el cubo aterriza vivo pero ya no puede ganar. Recorre la espera (no saltar)
    desde ``k`` hasta que esperar mata o, pasada la ventana (``last_tick``), el cubo deja de estar apoyado (se cayó
    de un borde: lo que sigue ya es terreno del próximo salto)."""
    traps, cur = 0, k
    while cur is not None:
        s = g.states[cur]
        if s.on_ground:
            land, _ = g.landing(g.after_jump(cur))
            if land is not None and g.J[land] == INF:
                traps += 1
        elif s.t > last_tick:
            break
        cur, _ = g.next[cur][0]
    return traps


def level_difficulty(level: Level, x0: float | None = None, p: Physics | None = None) -> Difficulty | None:
    """Analiza ``level`` arrancando en ``x0`` (por defecto ``level.start_x``). None si no tiene solución desde ahí.

    Las ventanas se miden sobre un camino óptimo que despega en el centro de cada ventana: la de cada salto está
    condicionada a ese camino (si el salto anterior fuera en otro tick, podría cambiar: eso es lo que mide
    ``linked_next``).
    """
    p = p or Physics()
    start = ph.initial_state(level, p, x0)
    g = _StateGraph(level, p, start)
    k0 = start.key()
    if g.J[k0] == INF:
        return None
    d = Difficulty(level.name, start.x, int(g.J[k0]), g.P[k0])

    cur = k0
    while g.J[cur] > 0:
        window = _scan_window(g, cur)
        lo, hi = g.states[window[0]], g.states[window[-1]]
        mid = window[len(window) // 2]
        info = JumpInfo(tick=g.states[mid].t, x=g.states[mid].x, first_tick=lo.t, last_tick=hi.t,
                        x_from=lo.x, x_to=hi.x, traps=_count_traps(g, cur, hi.t))
        if g.J[cur] > 1:
            # Ventana del próximo salto según en qué tick de esta ventana se despegue.
            nxt = {tuple(_scan_window(g, g.after_jump(w))) for w in window}
            sizes = [len(n) for n in nxt]
            info.linked_next = len(nxt) > 1
            info.next_windows = (min(sizes), max(sizes))
        d.jumps.append(info)
        cur = g.after_jump(mid)
        if cur is None:                              # el salto ganó en el mismo tick (meta justo adelante)
            break
    return d


def difficulty_over_phases(level: Level, phases, p: Physics | None = None) -> dict:
    """Resume ``level_difficulty`` en varias fases de arranque tomando el **peor caso** de cada métrica (ventanas
    más chicas, más trampas, cadenas más largas, menor probabilidad al azar). Las métricas cambian un poco con la fase
    porque el despegue sólo puede ocurrir en ticks enteros."""
    ds = [level_difficulty(level, float(x0), p) for x0 in phases]
    rows = [d.summary() for d in ds if d is not None]
    out = {"level": level.name, "solvable_phases": f"{len(rows)}/{len(ds)}"}
    worst = {"n_jumps": max, "min_window": min, "mean_window": min, "jumps_with_traps": max, "trap_ticks": max,
             "max_chain": max, "linked_jumps": max, "log10_p_random": min}
    for key, pick in worst.items():
        vals = [r[key] for r in rows if r[key] is not None]
        out[key] = pick(vals) if vals else None
    return out

"""Oráculo: planificador exacto para un nivel fijo y determinista.

Como la dinámica es determinista y x depende sólo del tick t, el espacio de
estados es un DAG por capas (una capa por tick). Se recorre capa a capa con
programación dinámica, guardando para cada estado alcanzable la mínima cantidad
de saltos y su predecesor. El resultado es un plan de acciones que gana el
nivel con la **menor cantidad de saltos** (óptimo bajo ese criterio).

Sirve para: (1) comprobar que el nivel tiene solución, (2) techo de desempeño,
(3) demostraciones expertas (clonado de comportamiento / DAgger) y (4) ubicar
dónde "deben" estar los saltos (para modelar la recompensa de timing).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import physics as ph
from .physics import Level, Physics, State


@dataclass
class Plan:
    """Resultado del planificador: la secuencia de acciones ganadora y dónde/cuándo salta."""
    actions: list[int]          # acción por tick desde el estado inicial
    jump_ticks: list[int]       # ticks (absolutos) donde se salta efectivamente
    jump_x: list[float]         # posición x del cubo al despegar
    n_jumps: int
    start_t: int


def solve(level: Level, p: Physics, start: State | None = None, max_ticks: int = 5000) -> Plan | None:
    """Devuelve el plan de mínimos saltos desde ``start`` (o None si no hay solución)."""
    start = start or ph.initial_state(level, p)
    # ``layer``: estados vivos del tick actual -> (estado, saltos usados para llegar). Se avanza un tick por iteración.
    layer: dict[tuple, tuple[State, int]] = {start.key(): (start, 0)}    # key -> (estado, saltos)
    back: list[dict[tuple, tuple[tuple, int]]] = []                      # por capa: key -> (key_prev, acción)
    winners: dict[tuple, tuple[State, int]] = {}

    for _ in range(max_ticks):
        # ``nxt``: estados del tick siguiente. ``parent``: de qué estado y con qué acción se llegó a cada uno.
        nxt: dict[tuple, tuple[State, int]] = {}
        parent: dict[tuple, tuple[tuple, int]] = {}
        for k, (s, cost) in layer.items():
            for a in ((0, 1) if s.on_ground else (0,)):     # en el aire el salto no hace nada
                ns, status = ph.step(level, p, s, a)
                if status in ph.DEAD:
                    continue                                    # camino que muere: se descarta
                nk, ncost = ns.key(), cost + (1 if a == 1 else 0)
                # Si dos caminos llegan al mismo estado, se queda el que usó menos saltos (programación dinámica).
                tgt = winners if status == ph.WIN else nxt
                if nk not in tgt or ncost < tgt[nk][1]:
                    tgt[nk] = (ns, ncost)
                    parent[nk] = (k, a)
        back.append(parent)
        if winners:                      # alguien llegó a la meta: se elige el ganador con menos saltos
            end_key = min(winners, key=lambda kk: winners[kk][1])
            return _reconstruct(back, end_key, start, winners[end_key][1], p)
        if not nxt:
            return None                  # no queda ningún estado vivo: el nivel no tiene solución desde ``start``
        layer = nxt
    return None


def _reconstruct(back, end_key, start: State, n_jumps: int, p: Physics) -> Plan:
    """Recorre los ``parent`` de atrás para adelante desde el estado ganador y arma la lista de acciones."""
    actions, key = [], end_key
    for parent in reversed(back):
        key, a = parent[key][0], parent[key][1]
        actions.append(a)
    actions.reverse()
    # Un 1 sólo cuenta como salto si el cubo estaba apoyado: se recomputa re-simulando.
    return Plan(actions=actions, jump_ticks=[], jump_x=[], n_jumps=n_jumps, start_t=start.t)


def annotate_jumps(level: Level, p: Physics, start: State, plan: Plan) -> Plan:
    """Re-simula el plan y completa ``jump_ticks`` / ``jump_x`` con los saltos que realmente despegan."""
    s, ticks, xs = start, [], []
    for a in plan.actions:
        if a == 1 and s.on_ground:
            ticks.append(s.t)
            xs.append(s.x)
        s, _ = ph.step(level, p, s, a)
    plan.jump_ticks, plan.jump_x = ticks, xs
    return plan


class Oracle:
    """Resuelve (y cachea) el plan desde cualquier estado alcanzable.

    Cada plan resuelto deja anotada la acción óptima de todos los estados de su
    camino, así que seguirlo tick a tick no vuelve a planificar.
    """

    def __init__(self, level: Level, p: Physics):
        self.level, self.p = level, p
        self._cache: dict[tuple, Plan | None] = {}     # plan ya resuelto desde cada estado inicial
        self._act: dict[tuple, int] = {}               # acción óptima de cada estado visitado por algún plan

    @staticmethod
    def _k(s: State) -> tuple:
        return (s.x0, *s.key())          # x0 forma parte de la clave: define la fase

    def plan(self, start: State | None = None) -> Plan | None:
        """Plan óptimo desde ``start`` (se calcula una vez y se cachea)."""
        start = start or ph.initial_state(self.level, self.p)
        k = self._k(start)
        if k not in self._cache:
            pl = solve(self.level, self.p, start)
            if pl is not None:
                annotate_jumps(self.level, self.p, start, pl)
                s = start
                for a in pl.actions:               # acción óptima de cada estado del camino
                    self._act.setdefault(self._k(s), a)
                    s, _ = ph.step(self.level, self.p, s, a)
            self._cache[k] = pl
        return self._cache[k]

    def action(self, state: State) -> int:
        """Mejor acción desde ``state`` (0 si ya no hay forma de ganar)."""
        k = self._k(state)
        if k not in self._act:
            pl = self.plan(state)
            if pl is None:
                return 0
        return self._act.get(k, 0)

    def is_winnable(self, state: State) -> bool:
        """¿Todavía se puede ganar desde este estado? (sirve para detectar si el agente ya "arruinó" el episodio)."""
        return self.plan(state) is not None


class OraclePolicy:
    """Política con la interfaz de SB3 (``predict``) que consulta el estado del entorno.

    Funciona aunque el estado no sea el inicial (re-planifica y cachea).
    """

    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._oracles: dict[int, Oracle] = {}          # un oráculo por nivel (el entorno puede cambiar de nivel)

    def predict(self, obs=None, state=None, episode_start=None, deterministic=True):
        # No usa ``obs``: mira el estado interno del entorno. Los argumentos extra existen sólo para imitar a SB3.
        return self.oracle.action(self.env.state), None

    @property
    def oracle(self) -> Oracle:
        """Oráculo del nivel que el entorno está jugando ahora (se crea la primera vez que se lo necesita)."""
        lv = self.env.level
        if id(lv) not in self._oracles:
            self._oracles[id(lv)] = Oracle(lv, self.env.p)
        return self._oracles[id(lv)]

"""Entorno Gymnasium de un Geometry Dash minimalista (modo cubo) sobre un nivel fijo."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import physics as ph
from .levels import level_1
from .physics import DEAD, WIN, Level, Physics, State
from .rewards import RewardFn, Transition, adapt_reward_fn, default_reward


class GeometryDashEnv(gym.Env):
    """Cubo que avanza solo; la acción es saltar (1) o no (0).

    Observaciones (``obs_mode``)
    ----------------------------
    * ``"grid"``     Box uint8 (3, H, W): canales [bloques, pinchos, jugador],
                     ventana ego-céntrica que arranca en la columna del cubo
                     (fila 0 de la imagen = techo, última = piso).
    * ``"features"`` Box float32 (4 + 2*lookahead,): [y/H, vy/jump_v, en_suelo,
                     parte fraccionaria de x] + por columna adelante
                     [altura del bloque más alto / H, hay pincho].
    * ``"dict"``     {"grid": ..., "vec": [y/H, vy/jump_v, en_suelo, frac(x)]}
                     (la grilla sola no trae la velocidad vertical).

    Recompensa por defecto: progreso (suma 1 en todo el nivel) + win_bonus al
    terminar - death_penalty al morir + survive_bonus por tick. Se puede
    reemplazar entera con ``reward_fn(tr)``, donde ``tr`` es un ``Transition`` (ver ``gd_env/rewards.py``);
    también se acepta la firma vieja ``reward_fn(prev, new, status, level)``.

    ``jump_penalty``: se resta en cada salto *efectivo* (acción 1 con el cubo
    apoyado; apretar 1 en el aire no cuesta nada). Se aplica también si hay
    ``reward_fn``, sumado a lo que esta devuelva.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        level: Level | None = None,
        physics: Physics | None = None,
        obs_mode: str = "grid",          # "grid" | "features" | "dict" (ver el docstring de la clase)
        window: int = 16,                # ancho (columnas) de la ventana de la grilla
        lookahead: int = 12,             # columnas hacia adelante que mira el modo "features"
        progress_scale: float = 1.0,     # cuánto vale recorrer todo el nivel (recompensa por avanzar)
        win_bonus: float = 1.0,          # bonus al llegar a la meta
        death_penalty: float = 10.0,     # castigo al morir (alto frente a jump_penalty: morir cuesta más que saltar)
        survive_bonus: float = 0.0,      # bonus por tick vivo (0 = no se usa)
        jump_penalty: float = 0.0,       # castigo por cada salto efectivo (0 = no se usa)
        reward_fn: RewardFn | None = None,   # reemplaza la recompensa por defecto (ver gd_env/rewards.py)
        max_steps: int | None = None,    # límite de ticks por episodio (por defecto 1.5x lo que dura el nivel)
        render_mode: str | None = None,
    ):
        assert obs_mode in ("grid", "features", "dict")
        self.level = level or level_1()
        self.p = physics or Physics()
        self.obs_mode, self.window, self.lookahead = obs_mode, window, lookahead
        self.progress_scale, self.win_bonus = progress_scale, win_bonus
        self.death_penalty, self.survive_bonus = death_penalty, survive_bonus
        self.jump_penalty = jump_penalty
        self.reward_fn = reward_fn                       # property: acepta fn(tr) o fn(prev, new, status, level)
        self.render_mode = render_mode

        # Ticks que dura el nivel a velocidad constante; el episodio se trunca a 1.5x para no quedar colgado.
        ticks_needed = int(np.ceil((self.level.goal_x - self.level.start_x) / self.p.vx))
        self.max_steps = max_steps or int(ticks_needed * 1.5)

        # Espacios de Gymnasium. Acción: 0 = nada, 1 = saltar. Observación: depende de ``obs_mode``.
        H = self.level.height
        self.action_space = spaces.Discrete(2)
        grid_space = spaces.Box(0, 1, (3, H, window), dtype=np.uint8)
        vec_space = spaces.Box(-2.0, 2.0, (4,), dtype=np.float32)
        feat_space = spaces.Box(-2.0, 2.0, (4 + 2 * lookahead,), dtype=np.float32)
        self.observation_space = {
            "grid": grid_space, "features": feat_space,
            "dict": spaces.Dict({"grid": grid_space, "vec": vec_space}),
        }[obs_mode]

        self.state: State = ph.initial_state(self.level, self.p)
        self.status = ph.OK                               # status del último tick (ok / crash / spike / fell / win)
        self._done = False
        self._trail: list[tuple[float, float]] = []
        self._jumps: list[tuple[float, float]] = []      # (x, y) de cada despegue

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)               # el entorno es determinista; seed sólo por API
        x0 = (options or {}).get("x0")         # permite variaciones de la fase de arranque
        self.state = ph.initial_state(self.level, self.p, x0)
        self.status = ph.OK
        self._done = False
        self._trail = [(self.state.x, self.state.y)]      # recorrido y saltos: sólo para graficar (render.py)
        self._jumps = []
        return self._obs(), self._info()

    def step(self, action):
        if self._done:      # estado terminal absorbente: el cubo muerto/ganador no se mueve más
            return self._obs(), 0.0, self.status in DEAD or self.status == WIN, False, self._info()
        prev = self.state
        jumped = int(action) == 1 and prev.on_ground       # salto efectivo: pedir salto estando apoyado
        if jumped:
            self._jumps.append((prev.x, prev.y))
        self.state, self.status = ph.step(self.level, self.p, prev, int(action))
        self._trail.append((self.state.x, self.state.y))

        tr = Transition(prev, self.state, self.status, self.level, jumped)
        reward = self._reward(tr)
        if jumped:                                          # la penalización por salto se suma a cualquier recompensa
            reward -= self.jump_penalty
        terminated = self.status in DEAD or self.status == WIN     # murió o ganó
        self._done = terminated
        truncated = (not terminated) and self.state.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ internals
    @property
    def reward_fn(self) -> RewardFn | None:
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, fn):
        self._reward_fn = adapt_reward_fn(fn)

    def _reward(self, tr: Transition) -> float:
        """Recompensa del tick: la de ``reward_fn`` si hay, si no la de por defecto con los parámetros del entorno."""
        if self._reward_fn is not None:
            return float(self._reward_fn(tr))
        return default_reward(tr, progress_scale=self.progress_scale, win_bonus=self.win_bonus,
                              death_penalty=self.death_penalty, survive_bonus=self.survive_bonus)

    def _info(self) -> dict:
        """Datos extra que devuelve Gymnasium en cada reset/step (no son la observación del agente)."""
        s = self.state
        span = self.level.goal_x - self.level.start_x
        return {"x": s.x, "y": s.y, "t": s.t, "status": self.status,
                "progress": float(np.clip((s.x - self.level.start_x) / span, 0, 1)),
                "success": self.status == WIN}

    def _vec(self) -> np.ndarray:
        """Vector de 4 números: altura y velocidad vertical normalizadas, si está apoyado, y la parte fraccionaria de x."""
        s, H = self.state, self.level.height
        return np.array([s.y / H, s.vy / self.p.jump_v, float(s.on_ground), s.x % 1.0], dtype=np.float32)

    def _grid(self) -> np.ndarray:
        """"Imagen" de (3, H, W) con 0/1: canal 0 = bloques, 1 = pinchos, 2 = jugador. La ventana arranca en la columna
        del cubo y muestra las ``window`` columnas siguientes (lo que viene por delante)."""
        s, lv = self.state, self.level
        c0 = int(np.floor(s.x))                           # primera columna de la ventana
        g = np.zeros((3, lv.height, self.window), dtype=np.uint8)
        for j in range(self.window):
            c = c0 + j
            if 0 <= c < lv.length:
                col = lv.grid[:, c]
                g[0, :, j] = col == ph.BLOCK
                g[1, :, j] = col == ph.SPIKE
        g = g[:, ::-1, :]                                       # piso abajo
        # Posición del jugador en la grilla (fila y columna del tile que ocupa su centro).
        pr = int(np.clip(np.floor(s.y + 0.5), 0, lv.height - 1))
        pc = int(np.floor(s.x + 0.5)) - c0
        if 0 <= pc < self.window:
            g[2, lv.height - 1 - pr, pc] = 1
        return np.ascontiguousarray(g)

    def _features(self) -> np.ndarray:
        """Vector plano: el de ``_vec`` + por cada una de las ``lookahead`` columnas siguientes
        [altura del bloque más alto / H, hay pincho]."""
        lv, c0 = self.level, int(np.floor(self.state.x))
        look = np.zeros(2 * self.lookahead, dtype=np.float32)
        for j in range(self.lookahead):
            look[2 * j] = lv.column_top(c0 + j) / lv.height
            look[2 * j + 1] = float(lv.has_spike(c0 + j))
        return np.concatenate([self._vec(), look])

    def _obs(self):
        """Observación según ``obs_mode`` (tipo y forma: ver ``observation_space``)."""
        if self.obs_mode == "grid":
            return self._grid()
        if self.obs_mode == "features":
            return self._features()
        return {"grid": self._grid(), "vec": self._vec()}

    # ------------------------------------------------------------- render
    def render(self):
        """``"rgb_array"``: frame como imagen (H, W, 3). ``"ansi"``: el nivel en texto (P = cubo, # = bloque, ^ = pincho)."""
        lv, s = self.level, self.state
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        lines = []
        pc, pr = int(np.floor(s.x + 0.5)), int(np.floor(s.y + 0.5))
        for r in range(lv.height - 1, -1, -1):
            row = []
            for c in range(lv.length):
                if (r, c) == (pr, pc):
                    row.append("P")
                else:
                    row.append(".#^"[int(lv.grid[r, c])])
            lines.append("".join(row))
        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        from .render import render_frame
        return render_frame(self.level, self.state, self.status, self._trail, self._jumps)

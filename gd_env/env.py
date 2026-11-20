"""Entorno Gymnasium de un Geometry Dash minimalista (modo cubo) sobre un nivel fijo."""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import physics as ph
from .levels import level_1
from .physics import DEAD, WIN, Level, Physics, State

RewardFn = Callable[[State, State, str, Level], float]


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
    reemplazar entera con ``reward_fn(prev, new, status, level)``.

    ``jump_penalty``: se resta en cada salto *efectivo* (acción 1 con el cubo
    apoyado; apretar 1 en el aire no cuesta nada). Se aplica también si hay
    ``reward_fn``, sumado a lo que esta devuelva.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        level: Level | None = None,
        physics: Physics | None = None,
        obs_mode: str = "grid",
        window: int = 16,
        lookahead: int = 12,
        progress_scale: float = 1.0,
        win_bonus: float = 1.0,
        death_penalty: float = 10.0,
        survive_bonus: float = 0.0,
        jump_penalty: float = 0.0,
        reward_fn: RewardFn | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ):
        assert obs_mode in ("grid", "features", "dict")
        self.level = level or level_1()
        self.p = physics or Physics()
        self.obs_mode, self.window, self.lookahead = obs_mode, window, lookahead
        self.progress_scale, self.win_bonus = progress_scale, win_bonus
        self.death_penalty, self.survive_bonus = death_penalty, survive_bonus
        self.jump_penalty = jump_penalty
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        ticks_needed = int(
            np.ceil((self.level.goal_x - self.level.start_x) / self.p.vx))
        self.max_steps = max_steps or int(ticks_needed * 1.5)

        H = self.level.height
        self.action_space = spaces.Discrete(2)
        grid_space = spaces.Box(0, 1, (3, H, window), dtype=np.uint8)
        vec_space = spaces.Box(-2.0, 2.0, (4,), dtype=np.float32)
        feat_space = spaces.Box(-2.0, 2.0,
                                (4 + 2 * lookahead,), dtype=np.float32)
        self.observation_space = {
            "grid": grid_space, "features": feat_space,
            "dict": spaces.Dict({"grid": grid_space, "vec": vec_space}),
        }[obs_mode]

        self.state: State = ph.initial_state(self.level, self.p)
        self.status = ph.OK
        self._done = False
        self._trail: list[tuple[float, float]] = []
        # (x, y) de cada despegue
        self._jumps: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # el entorno es determinista; seed sólo por API
        super().reset(seed=seed)
        # permite variaciones de la fase de arranque
        x0 = (options or {}).get("x0")
        self.state = ph.initial_state(self.level, self.p, x0)
        self.status = ph.OK
        self._done = False
        self._trail = [(self.state.x, self.state.y)]
        self._jumps = []
        return self._obs(), self._info()

    def step(self, action):
        if self._done:      # estado terminal absorbente: el cubo muerto/ganador no se mueve más
            return self._obs(), 0.0, self.status in DEAD or self.status == WIN, False, self._info()
        prev = self.state
        jumped = int(action) == 1 and prev.on_ground
        if jumped:
            self._jumps.append((prev.x, prev.y))
        self.state, self.status = ph.step(
            self.level, self.p, prev, int(action))
        self._trail.append((self.state.x, self.state.y))

        reward = self._reward(prev, self.state, self.status)
        if jumped:
            reward -= self.jump_penalty
        terminated = self.status in DEAD or self.status == WIN
        self._done = terminated
        truncated = (not terminated) and self.state.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ internals
    def _reward(self, prev: State, new: State, status: str) -> float:
        if self.reward_fn is not None:
            return self.reward_fn(prev, new, status, self.level)
        span = self.level.goal_x - self.level.start_x
        r = self.progress_scale * (new.x - prev.x) / span + self.survive_bonus
        if status in DEAD:
            r -= self.death_penalty
        elif status == WIN:
            r += self.win_bonus
        return r

    def _info(self) -> dict:
        s = self.state
        span = self.level.goal_x - self.level.start_x
        return {"x": s.x, "y": s.y, "t": s.t, "status": self.status,
                "progress": float(np.clip((s.x - self.level.start_x) / span, 0, 1)),
                "success": self.status == WIN}

    def _vec(self) -> np.ndarray:
        s, H = self.state, self.level.height
        return np.array([s.y / H, s.vy / self.p.jump_v, float(s.on_ground), s.x % 1.0], dtype=np.float32)

    def _grid(self) -> np.ndarray:
        s, lv = self.state, self.level
        c0 = int(np.floor(s.x))
        g = np.zeros((3, lv.height, self.window), dtype=np.uint8)
        for j in range(self.window):
            c = c0 + j
            if 0 <= c < lv.length:
                col = lv.grid[:, c]
                g[0, :, j] = col == ph.BLOCK
                g[1, :, j] = col == ph.SPIKE
        g = g[:, ::-1, :]                                       # piso abajo
        pr = int(np.clip(np.floor(s.y + 0.5), 0, lv.height - 1))
        pc = int(np.floor(s.x + 0.5)) - c0
        if 0 <= pc < self.window:
            g[2, lv.height - 1 - pr, pc] = 1
        return np.ascontiguousarray(g)

    def _features(self) -> np.ndarray:
        lv, c0 = self.level, int(np.floor(self.state.x))
        look = np.zeros(2 * self.lookahead, dtype=np.float32)
        for j in range(self.lookahead):
            look[2 * j] = lv.column_top(c0 + j) / lv.height
            look[2 * j + 1] = float(lv.has_spike(c0 + j))
        return np.concatenate([self._vec(), look])

    def _obs(self):
        if self.obs_mode == "grid":
            return self._grid()
        if self.obs_mode == "features":
            return self._features()
        return {"grid": self._grid(), "vec": self._vec()}

    # ------------------------------------------------------------- render
    def render(self):
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

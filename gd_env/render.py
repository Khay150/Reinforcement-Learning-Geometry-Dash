"""Visualización: frames, GIF del episodio y mapa del recorrido sobre el nivel.

Funciona con cualquier política: el oráculo, una función ``f(obs) -> acción`` o un
modelo de Stable-Baselines3 (cualquier objeto con ``predict``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import physics as ph
from .physics import DEAD, Level, State

# Colores (RGB). El cielo es un degradé vertical de SKY_TOP a SKY_BOT.
SKY_TOP, SKY_BOT = (70, 90, 200), (120, 160, 240)
BLOCK_FILL, BLOCK_EDGE = (25, 25, 45), (90, 90, 140)
SPIKE_COL = (230, 60, 60)
PLAYER_FILL, PLAYER_EDGE = (250, 210, 40), (60, 40, 0)


def _sky(w: int, h: int) -> Image.Image:
    """Fondo con degradé vertical de tamaño w x h."""
    t = np.linspace(0, 1, h)[:, None, None]
    top, bot = np.array(SKY_TOP), np.array(SKY_BOT)
    arr = (top * (1 - t) + bot * t).astype(np.uint8).repeat(w, axis=1)
    return Image.fromarray(arr)


def _draw_tiles(d: ImageDraw.ImageDraw, level: Level, c_from: int, c_to: int, cell: int, x_off: float, H: int):
    """Dibuja columnas c_from..c_to; x_off = desplazamiento (en tiles) de la cámara."""
    for c in range(max(c_from, 0), min(c_to, level.length)):
        px = (c - x_off) * cell
        for r in range(level.height):
            t = int(level.grid[r, c])
            if not t:
                continue
            y0 = (H - 1 - r) * cell
            if t == ph.BLOCK:
                d.rectangle([px, y0, px + cell - 1, y0 + cell - 1], fill=BLOCK_FILL, outline=BLOCK_EDGE)
            elif t == ph.SPIKE_DOWN:
                d.polygon([(px + 2, y0 + 1), (px + cell - 3, y0 + 1), (px + cell / 2, y0 + cell - 4)], fill=SPIKE_COL)
            else:
                d.polygon([(px + 2, y0 + cell - 1), (px + cell / 2, y0 + 3), (px + cell - 3, y0 + cell - 1)], fill=SPIKE_COL)


def render_frame(level: Level, state: State, status: str = ph.OK, trail=None, jumps=None,
                 cols: int = 24, cell: int = 24, cam_lead: float = 6.0) -> np.ndarray:
    """Un frame con cámara que sigue al cubo (el cubo queda ``cam_lead`` tiles desde la izquierda)."""
    H = level.height
    W, Hpx = cols * cell, H * cell              # tamaño en píxeles del área de juego (la barra de HUD va abajo, +28 px)
    img = _sky(W, Hpx + 28)
    d = ImageDraw.Draw(img)
    cam = state.x - cam_lead                    # x (en tiles) del borde izquierdo de la cámara
    _draw_tiles(d, level, int(np.floor(cam)), int(np.floor(cam)) + cols + 2, cell, cam, H)

    for (tx, ty) in (trail or [])[-120:]:                       # estela
        px, py = (tx - cam + 0.5) * cell, (H - ty - 0.5) * cell
        d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 255, 255))
    for (jx, jy) in (jumps or []):                              # marcas de salto
        px, py = (jx - cam + 0.5) * cell, (H - jy) * cell
        if 0 <= px <= W:
            d.polygon([(px, py + 6), (px - 5, py + 14), (px + 5, py + 14)], fill=(120, 255, 160))

    px, py = (state.x - cam) * cell, (H - state.y - 1) * cell     # el eje y de la imagen apunta hacia abajo
    col = (255, 90, 90) if status in DEAD else PLAYER_FILL
    d.rectangle([px, py, px + cell - 1, py + cell - 1], fill=col, outline=PLAYER_EDGE, width=2)
    d.rectangle([px + cell * .55, py + cell * .25, px + cell * .8, py + cell * .45], fill=PLAYER_EDGE)  # ojo

    span = level.goal_x - level.start_x                         # barra de progreso + HUD
    prog = float(np.clip((state.x - level.start_x) / span, 0, 1))
    d.rectangle([0, Hpx, W, Hpx + 28], fill=(15, 15, 25))
    d.rectangle([8, Hpx + 9, 8 + (W - 140) * prog, Hpx + 19], fill=(90, 220, 130))
    d.rectangle([8, Hpx + 9, W - 132, Hpx + 19], outline=(160, 160, 190))
    label = {ph.WIN: "¡NIVEL COMPLETO!", ph.CRASH: "choque", ph.SPIKE_HIT: "pincho", ph.FELL: "caída"}.get(status, "")
    d.text((W - 124, Hpx + 8), f"{prog:4.0%} t={state.t} {label}", fill=(235, 235, 245))
    return np.asarray(img.convert("RGB"))


def trajectory_image(level: Level, trail, jumps=(), status: str = ph.OK, cell: int = 12) -> np.ndarray:
    """Nivel completo con el recorrido del agente, los saltos (verde) y dónde terminó."""
    H, W = level.height, level.length
    img = _sky(W * cell, H * cell)
    d = ImageDraw.Draw(img)
    _draw_tiles(d, level, 0, W, cell, 0.0, H)
    pts = [((x + 0.5) * cell, (H - y - 0.5) * cell) for x, y in trail]
    if len(pts) > 1:
        d.line(pts, fill=(255, 255, 255), width=2)
    for jx, jy in jumps:
        px, py = (jx + 0.5) * cell, (H - jy) * cell
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(60, 230, 120), outline=(0, 60, 20))
    if status in DEAD and pts:
        x, y = pts[-1]
        d.line([x - 6, y - 6, x + 6, y + 6], fill=(255, 40, 40), width=3)
        d.line([x - 6, y + 6, x + 6, y - 6], fill=(255, 40, 40), width=3)
    return np.asarray(img.convert("RGB"))


def _act(policy, obs, env):
    """Acción (int) de cualquier tipo de política: objeto con ``predict`` (SB3, oráculo) o función ``f(obs)``."""
    if hasattr(policy, "predict"):
        a = policy.predict(obs, deterministic=True)
        a = a[0] if isinstance(a, tuple) else a
    else:
        a = policy(obs)
    return int(np.asarray(a).reshape(-1)[0])


def record_episode(env, policy, gif_path: str | None = "episode.gif", trajectory_path: str | None = None,
                   frame_skip: int = 2, fps: int = 30, seed: int | None = 0, options: dict | None = None,
                   max_frames: int = 1500) -> dict:
    """Corre un episodio con ``policy`` y guarda un GIF (y opcionalmente el mapa del recorrido).

    ``frame_skip``: se guarda 1 de cada N ticks (2 → GIF más liviano y fluido).
    Devuelve ``info`` final más ``return`` y la lista de ``frames``.
    """
    u = env.unwrapped                           # entorno sin wrappers: se necesita el estado interno para dibujar
    obs, info = env.reset(seed=seed, options=options)
    frames = [render_frame(u.level, u.state, u.status, u._trail, u._jumps)]
    done, ret = False, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(_act(policy, obs, u))
        ret += r
        done = term or trunc
        if u.state.t % frame_skip == 0 or done:
            frames.append(render_frame(u.level, u.state, u.status, u._trail, u._jumps))
    frames += [frames[-1]] * fps                                 # pausa de 1 s al final
    frames = frames[-max_frames:] if len(frames) > max_frames else frames

    if gif_path:
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    if trajectory_path:
        Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(trajectory_image(u.level, u._trail, u._jumps, u.status)).save(trajectory_path)
    return {**info, "return": ret, "frames": frames}

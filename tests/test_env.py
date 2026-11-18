import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import gd_env
from gd_env import GeometryDashEnv, Oracle, OraclePolicy, level_1, parse_level, to_ascii
from gd_env import physics as ph


def run(env, policy, **reset_kw):
    obs, info = env.reset(**reset_kw)
    done, ret = False, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        ret += r
        done = term or trunc
    return info, ret


@pytest.mark.parametrize("mode", ["grid", "features", "dict"])
def test_check_env(mode):
    check_env(GeometryDashEnv(obs_mode=mode), skip_render_check=True)


def test_registered():
    import gymnasium as gym
    env = gym.make("GeometryDash-v0")
    env.reset()
    env.step(0)


def test_jump_physics():
    lv, p = level_1(), ph.Physics()
    s = ph.initial_state(lv, p)
    s, _ = ph.step(lv, p, s, 1)
    assert not s.on_ground and s.y > 1.0
    peak, n = s.y, 1
    while not s.on_ground:
        s, st = ph.step(lv, p, s, 0)
        peak, n = max(peak, s.y), n + 1
    assert 2.4 < peak < 2.7          # apex ~1.5 tiles sobre el piso
    assert 14 <= n <= 18             # ~16 ticks de vuelo
    assert s.y == 1.0


def test_deterministic():
    a, b = GeometryDashEnv(obs_mode="features"), GeometryDashEnv(obs_mode="features")
    oa, _ = a.reset(seed=0); ob, _ = b.reset(seed=123)
    rng = np.random.default_rng(0)
    for act in rng.integers(0, 2, 200):
        oa, ra, ta, tra, _ = a.step(act); ob, rb, tb, trb, _ = b.step(act)
        assert np.array_equal(oa, ob) and ra == rb and ta == tb
        if ta or tra:
            break


def test_trivial_policies_fail():
    env = GeometryDashEnv(obs_mode="features")
    info, _ = run(env, lambda o, e: 0)
    assert info["status"] == ph.SPIKE_HIT and not info["success"]
    info, _ = run(env, lambda o, e: 1)
    assert not info["success"]


def test_random_policy_does_not_win():
    env = GeometryDashEnv(obs_mode="features")
    rng = np.random.default_rng(0)
    wins = sum(run(env, lambda o, e: int(rng.random() < 0.1))[0]["success"] for _ in range(300))
    assert wins == 0


def test_oracle_wins_with_min_jumps():
    env = GeometryDashEnv(obs_mode="grid")
    pol = OraclePolicy(env)
    info, ret = run(env, lambda o, e: pol.predict(o)[0])
    assert info["success"] and info["progress"] == 1.0
    plan = pol.oracle.plan(ph.initial_state(env.level, env.p))
    assert plan.n_jumps == len(plan.jump_ticks) == 13


def test_oracle_is_minimal_vs_bruteforce_on_tiny_level():
    lv = parse_level("""
........
........
..^.....
########
""", start_x=0.0, goal_margin=1)
    p = ph.Physics()
    plan = Oracle(lv, p).plan()
    assert plan is not None and plan.n_jumps == 1


def test_oracle_from_other_phases():
    """Variaciones de fase de arranque (x0): informa cuáles tienen solución."""
    env = GeometryDashEnv(obs_mode="features")
    pol = OraclePolicy(env)
    for x0 in np.linspace(2.0, 2.95, 20):
        pl = pol.oracle.plan(ph.initial_state(env.level, env.p, x0))
        assert pl is not None, f"x0={x0:.2f} sin solución"
        info, _ = run(env, lambda o, e: pol.predict()[0], options={"x0": float(x0)})
        assert info["success"]


def test_obs_grid_contents():
    env = GeometryDashEnv(obs_mode="grid", window=16)
    obs, _ = env.reset()
    assert obs.shape == (3, 7, 16) and obs.dtype == np.uint8
    assert obs[2].sum() == 1                       # un único marcador de jugador
    assert obs[0, -1].all()                        # piso en la última fila
    assert obs[1].sum() == 1                       # primer pincho (col 14) dentro de la ventana


def test_render():
    env = GeometryDashEnv(render_mode="ansi"); env.reset()
    txt = env.render(); assert "P" in txt and "^" in txt
    env = GeometryDashEnv(render_mode="rgb_array"); env.reset()
    assert env.render().ndim == 3


def test_ascii_roundtrip():
    lv = level_1()
    lv2 = parse_level(to_ascii(lv))
    assert np.array_equal(lv.grid, lv2.grid)


def test_record_episode(tmp_path):
    from gd_env import record_episode
    env = GeometryDashEnv(obs_mode="features")
    pol = OraclePolicy(env)
    out = record_episode(env, pol, gif_path=str(tmp_path / "ep.gif"),
                         trajectory_path=str(tmp_path / "tray.png"), frame_skip=4, fps=20)
    assert out["success"] and len(out["frames"]) > 100
    assert (tmp_path / "ep.gif").stat().st_size > 1000 and (tmp_path / "tray.png").exists()
    assert len(env.unwrapped._jumps) == 13


def test_dead_cube_stops():
    """Al morir el cubo queda congelado; seguir llamando a step no lo mueve."""
    env = GeometryDashEnv(obs_mode="features")
    env.reset()
    done = False
    while not done:                                   # nunca salta -> muere en el primer pincho
        _, _, done, _, info = env.step(0)
    assert info["status"] == ph.SPIKE_HIT
    x, y, t = env.state.x, env.state.y, env.state.t
    for _ in range(10):
        _, r, term, _, _ = env.step(0)
        assert term and r == 0.0
    assert (env.state.x, env.state.y, env.state.t) == (x, y, t)


def test_side_crash_stops_at_block_face():
    """Choque de costado: el cubo queda pegado a la cara del bloque, sin solaparse."""
    lv = parse_level("""
..........
..........
.....#....
##########""", start_x=0.0, goal_margin=1)
    env = GeometryDashEnv(level=lv, obs_mode="features")
    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(0)
    assert info["status"] == ph.CRASH
    assert env.state.x + 1.0 <= 5.0 + 1e-6            # borde derecho del cubo <= cara izquierda del bloque (x=5)

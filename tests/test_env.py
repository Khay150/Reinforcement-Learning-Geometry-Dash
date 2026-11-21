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


def test_jump_penalty_only_on_effective_jumps():
    """Sólo cuesta el salto efectivo (apoyado); mantener 1 en el aire no suma penalización."""
    c = 0.05
    for policy in (lambda o, e: 1, lambda o, e: int(e.state.t % 3 == 0)):
        base, pen = GeometryDashEnv(obs_mode="features"), GeometryDashEnv(obs_mode="features", jump_penalty=c)
        _, r0 = run(base, policy)
        _, r1 = run(pen, policy)
        assert len(pen._jumps) > 0
        assert r1 == pytest.approx(r0 - c * len(pen._jumps))


def test_oracle_return_with_jump_penalty():
    env = GeometryDashEnv(obs_mode="features", jump_penalty=0.02)
    pol = OraclePolicy(env)
    info, ret = run(env, lambda o, e: pol.predict(o)[0])
    assert info["success"] and ret == pytest.approx(2.0 - 0.02 * 13)


def test_death_penalty_default():
    assert GeometryDashEnv().death_penalty == 10.0


def ppo_on_vec_env(random_phase=False):
    """PPO sin entrenar sobre 2 entornos (para probar callbacks)."""
    from agents import build_model, make_train_env
    venv = make_train_env("features", n_envs=2, seed=0, random_phase=random_phase)
    return venv, build_model("ppo", venv, seed=0)


def test_performance_ramp_logic():
    """Sube c cuando gana >= up en la ventana, baja con <= down, no se mueve en el medio; topes 0 y c_max."""
    from agents.callbacks import PerformanceJumpPenaltyRamp
    r = PerformanceJumpPenaltyRamp(c_max=0.02, step=0.01, window=4, up=0.75, down=0.25)
    assert not any(r.observe(False) for _ in range(4)) and r.c == 0.0        # pierde siempre: no baja de 0
    # ventana móvil [F,F,F,F] -> ... -> [F,T,T,T]: al tercer éxito llega a 0.75 >= up
    assert [r.observe(True) for _ in range(3)] == [False, False, True] and r.c == 0.01
    assert len(r.results) == 0                                                  # ventana vacía tras el cambio
    for s in (True, False, True, False):                                        # 50%: zona neutra
        assert not r.observe(s)
    assert r.c == 0.01
    for _ in range(8):
        r.observe(True)
    assert r.c == 0.02                                                          # tope c_max
    for _ in range(4):
        r.observe(True)
    assert r.c == 0.02
    for _ in range(4):
        r.observe(False)
    assert r.c == pytest.approx(0.01)                                           # se afloja si muere mucho


def test_performance_ramp_reads_monitor_episodes():
    """Integrado con SB3: lee el éxito de los episodios desde Monitor y aplica c al entorno de adentro."""
    from agents.callbacks import PerformanceJumpPenaltyRamp
    venv, model = ppo_on_vec_env(random_phase=True)          # random_phase: hay un wrapper más entre Monitor y el entorno
    ramp = PerformanceJumpPenaltyRamp(c_max=0.05, step=0.01, window=2, up=1.0, down=0.0)
    model.learn(total_timesteps=2048, callback=ramp)         # sin entrenar, muere siempre -> éxito 0 <= down
    assert ramp.last_rate == 0.0 and ramp.c == 0.0
    assert [e.unwrapped.jump_penalty for e in venv.envs] == [0.0, 0.0]


def test_performance_ramp_reaches_inner_env():
    """Cuando c cambia, el valor llega al entorno de adentro (debajo de Monitor/RandomPhase), no al wrapper."""
    from agents.callbacks import PerformanceJumpPenaltyRamp
    venv, model = ppo_on_vec_env(random_phase=True)
    ramp = PerformanceJumpPenaltyRamp(c_max=0.05)
    ramp.init_callback(model)
    ramp.c = 0.03
    ramp._apply()
    assert [e.unwrapped.jump_penalty for e in venv.envs] == [pytest.approx(0.03)] * 2


def test_best_model_callback(tmp_path):
    import json
    from agents.callbacks import BestModelCallback
    _, model = ppo_on_vec_env()
    cb = BestModelCallback(eval_freq=512, save_dir=tmp_path, phases=[2.0, 2.5])
    model.learn(total_timesteps=1024, callback=cb)
    assert (tmp_path / "best_model.zip").exists()
    info = json.loads((tmp_path / "best_model.json").read_text())
    assert {"timesteps", "success", "progress", "mean_jumps_won"} <= info.keys()
    hist = (tmp_path / "eval_history.csv").read_text().strip().splitlines()
    assert len(hist) >= 3                                    # encabezado + >= 2 evaluaciones (incluye la final)


def test_score_prefers_success_then_fewer_jumps():
    from agents.callbacks import _score
    from agents.evaluate import metrics
    row = lambda s, j: {"success": s, "progress": 1.0 if s else 0.5, "n_jumps": j}
    assert _score(metrics([row(1, 27)])) > _score(metrics([row(0, 13)]))        # ganar pesa más que saltar poco
    assert _score(metrics([row(1, 13)])) > _score(metrics([row(1, 27)]))        # a igual éxito, menos saltos


@pytest.mark.parametrize("algo", ["ppo", "a2c", "dqn", "qrdqn"])
def test_continue_training_from_saved_model(tmp_path, algo):
    import torch
    from agents import ON_POLICY, build_model, make_train_env
    from agents.common import load_for_training
    if algo == "qrdqn":
        pytest.importorskip("sb3_contrib")
    on_policy = algo in ON_POLICY
    model = build_model(algo, make_train_env("features", n_envs=2 if on_policy else 1, seed=0, random_phase=False), seed=0)
    model.save(tmp_path / "m")
    params = {k: v.clone() for k, v in model.policy.state_dict().items()}
    # PPO/A2C: otra cantidad de entornos que el original
    env2 = make_train_env("features", n_envs=3 if on_policy else 1, seed=1, random_phase=False)
    cont = load_for_training(algo, str(tmp_path / "m.zip"), env2, seed=1)
    assert all(torch.equal(params[k], v) for k, v in cont.policy.state_dict().items())   # mismos pesos
    if not on_policy:
        assert cont.exploration_schedule(1.0) == pytest.approx(cont.exploration_final_eps)  # no vuelve a ε=1
    cont.learn(total_timesteps=600)
    assert cont.num_timesteps >= 600

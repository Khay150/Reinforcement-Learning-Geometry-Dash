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


def test_reward_params_default_and_custom():
    """win_bonus / death_penalty / progress_scale / survive_bonus se pasan como parámetros del entorno."""
    pol = lambda o, e: 0                       # nunca salta: choca
    base, custom = GeometryDashEnv(obs_mode="features"), GeometryDashEnv(obs_mode="features", death_penalty=3.0)
    _, r0 = run(base, pol)
    _, r1 = run(custom, pol)
    assert r1 == pytest.approx(r0 + 7.0)       # 10 - 3 menos de castigo
    win = GeometryDashEnv(obs_mode="features", win_bonus=5.0)
    opol = OraclePolicy(win)
    _, ret = run(win, lambda o, e: opol.predict(o)[0])
    assert ret == pytest.approx(1.0 + 5.0)     # progreso (1) + bonus de meta


def test_survive_bonus_every_interval():
    """survive_bonus se da cada survive_interval ticks vivo; el tick en que muere no lo cobra."""
    pol = lambda o, e: 0                       # nunca salta: choca
    def total(**kw):
        env = GeometryDashEnv(obs_mode="features", progress_scale=0.0, death_penalty=0.0, **kw)
        env.reset()
        ticks, tot = 0, 0.0
        while True:
            _, r, term, trunc, _ = env.step(0)
            ticks += 1; tot += r
            if term or trunc:
                return ticks, tot
    n, base = total()
    assert base == pytest.approx(0.0)
    n1, every = total(survive_bonus=0.5)                       # cada tick vivo (el último, el de la muerte, no cuenta)
    assert every == pytest.approx(0.5 * (n1 - 1))
    n2, spaced = total(survive_bonus=0.5, survive_interval=5)
    assert spaced == pytest.approx(0.5 * ((n2 - 1) // 5))


def test_survive_rate_follows_jump_penalty():
    """survive_rate: bonus por tick = rate * jump_penalty vigente (cambia si la rampa cambia jump_penalty)."""
    def total(c):
        env = GeometryDashEnv(obs_mode="features", progress_scale=0.0, death_penalty=0.0,
                              jump_penalty=0.0, survive_rate=0.5, survive_interval=5)
        env.reset(); env.jump_penalty = c           # la rampa cambia jump_penalty durante el entrenamiento
        tot, n = 0.0, 0
        while True:
            _, r, term, trunc, _ = env.step(0)      # nunca salta: no paga saltos
            tot += r; n += 1
            if term or trunc:
                return tot, n
    t0, _ = total(0.0)
    t1, n1 = total(0.1)
    t2, n2 = total(0.2)
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(0.5 * 0.1 * 5 * ((n1 - 1) // 5))
    assert t2 == pytest.approx(2 * t1) and n1 == n2
    with pytest.raises(ValueError):
        GeometryDashEnv(survive_rate=0.1, survive_bonus=0.1)


def test_reward_fn_new_and_legacy_signature():
    """reward_fn(tr) y la firma vieja (prev, new, status, level) dan lo mismo; jump_penalty se suma encima."""
    from gd_env import Transition, default_reward
    new_style = lambda tr: default_reward(tr, death_penalty=2.0) + (0.5 if tr.jumped else 0.0)
    legacy = lambda prev, new, status, level: (
        (new.x - prev.x) / (level.goal_x - level.start_x) - (2.0 if status in ph.DEAD else 0.0)
        + (1.0 if status == ph.WIN else 0.0) + (0.5 if (new.vy > 0 and prev.on_ground) else 0.0))
    pol = lambda o, e: int(e.state.t % 3 == 0)
    _, r_new = run(GeometryDashEnv(obs_mode="features", reward_fn=new_style), pol)
    _, r_old = run(GeometryDashEnv(obs_mode="features", reward_fn=legacy), pol)
    assert r_new == pytest.approx(r_old)
    pen = GeometryDashEnv(obs_mode="features", reward_fn=new_style, jump_penalty=0.1)
    _, r_pen = run(pen, pol)
    assert r_pen == pytest.approx(r_new - 0.1 * len(pen._jumps))
    with pytest.raises(TypeError):
        GeometryDashEnv(reward_fn=lambda a, b: 0.0)


def test_load_reward_fn_and_reward_kwargs(tmp_path):
    from agents.runs import reward_kwargs
    from gd_env import load_reward_fn, reward_spec
    f = tmp_path / "mi_reward.py"
    f.write_text("def mi_fn(tr):\n    return 1.0\n")
    fn = load_reward_fn(f"{f}:mi_fn")
    assert fn(None) == 1.0 and reward_spec(fn).endswith(":mi_fn")
    kw = reward_kwargs({"jump_penalty": 0.05, "win_bonus": 2.0, "death_penalty": None, "reward_fn": f"{f}:mi_fn"})
    assert kw["jump_penalty"] == 0.05 and kw["win_bonus"] == 2.0 and "death_penalty" not in kw
    assert GeometryDashEnv(**kw).reward_fn is not None
    with pytest.raises(ValueError):
        load_reward_fn("sin_dos_puntos")
    assert reward_kwargs({"jump_penalty": 0.1, "reward_fn": "no_existe_xyz:f"}) == {"jump_penalty": 0.1}


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


# ------------------------------------------------------------------ varios niveles
def mini_levels():
    """Dos niveles chicos (misma altura que level_1) para probar el modo multinivel."""
    from gd_env import build_level
    return (build_level(length=40, height=7, spikes=[14], name="mini_a"),
            build_level(length=50, height=7, spikes=[20], pits=[(30, 2)], name="mini_b"))


@pytest.fixture
def registered_minis(monkeypatch):
    """Registra los niveles chicos en gd_env.LEVELS sólo durante el test."""
    a, b = mini_levels()
    monkeypatch.setitem(gd_env.LEVELS, "mini_a", lambda: mini_levels()[0])
    monkeypatch.setitem(gd_env.LEVELS, "mini_b", lambda: mini_levels()[1])
    return a, b


def test_all_registered_levels_solvable_from_eval_phases():
    """Cada nivel de gd_env.LEVELS se gana con el oráculo desde todas las fases de evaluación."""
    from agents.common import EVAL_PHASES
    from agents.evaluate import oracle_jumps
    for name in gd_env.LEVELS:
        lv = gd_env.get_level(name)
        assert lv.height == level_1().height and lv.start_x == 2.0, name
        for x0 in EVAL_PHASES:
            assert oracle_jumps(lv, x0) >= 0, f"{name} sin solución desde x0={x0}"    # -1 = sin solución (0 saltos vale: franco_sin_saltar)


def test_env_samples_levels_by_weight():
    a, b = mini_levels()
    env = GeometryDashEnv(level=[a, b], level_weights=[1.0, 3.0], obs_mode="features")
    env.reset(seed=0)
    names = [env.reset()[1]["level"] for _ in range(400)]
    assert 0.65 < names.count("mini_b") / 400 < 0.85                 # peso 3 de 4
    env.level_weights = [1.0, 0.0]                                    # peso 0: no se juega (se puede cambiar en caliente)
    assert {env.reset()[1]["level"] for _ in range(50)} == {"mini_a"}
    # forzar un nivel por nombre o índice; max_steps sigue al largo de cada nivel
    _, info = env.reset(options={"level": "mini_b"})
    assert info["level"] == "mini_b" and env.max_steps == int(np.ceil((b.goal_x - b.start_x) / env.p.vx) * 1.5)
    assert env.reset(options={"level": 0})[1]["level"] == "mini_a"
    with pytest.raises(ValueError):
        env.reset(options={"level": "no_existe"})
    with pytest.raises(ValueError):                                   # alturas distintas: la observación cambiaría
        GeometryDashEnv(level=[a, gd_env.build_level(length=40, height=5, name="bajo")])


def test_oracle_policy_follows_level_changes():
    a, b = mini_levels()
    env = GeometryDashEnv(level=[a, b], obs_mode="features")
    pol = OraclePolicy(env)
    for name in ("mini_a", "mini_b", "mini_a"):
        info, _ = run(env, lambda o, e: pol.predict()[0], options={"level": name})
        assert info["success"] and info["level"] == name


def test_resolve_levels(registered_minis, tmp_path):
    import json
    from gd_env import resolve_levels
    assert resolve_levels(None) == {"train": {"level_1": 1.0}, "val": ["level_1"], "held_out": False,
                                    "progression": None}
    names = resolve_levels(["level_1", "mini_a"], val_seed=0)            # nombres sueltos: validación null
    assert names["train"] == {"level_1": 1.0, "mini_a": 1.0} and not names["held_out"]
    assert names["val"][0] in names["train"]

    def spec(niveles, validacion):
        f = tmp_path / f"set{len(list(tmp_path.iterdir()))}.json"
        f.write_text(json.dumps({"_nota": "se ignora", "niveles": niveles, "validacion": validacion}))
        return str(f)

    fixed = resolve_levels([spec({"level_1": 1, "mini_a": 2, "mini_b": 1}, "mini_b")])
    assert fixed == {"train": {"level_1": 1.0, "mini_a": 2.0}, "val": ["mini_b"], "held_out": True,
                     "progression": None}
    lst = resolve_levels([spec(["level_1", "mini_a"], ["mini_b"])])
    assert lst["train"] == {"level_1": 1.0, "mini_a": 1.0} and lst["val"] == ["mini_b"]
    drawn = [resolve_levels([spec(["level_1", "mini_a", "mini_b"], 1)], val_seed=s) for s in (0, 0, 1, 2, 3)]
    assert drawn[0] == drawn[1]                                          # reproducible con la misma semilla
    for d in drawn:
        assert len(d["val"]) == 1 and d["held_out"] and d["val"][0] not in d["train"] and len(d["train"]) == 2
    with pytest.raises(ValueError):
        resolve_levels([spec(["level_1", "mini_a"], 2)])                 # no quedaría ninguno para entrenar
    with pytest.raises(ValueError):
        resolve_levels(["no_existe"])


def test_evaluate_and_history_with_validation(registered_minis, tmp_path):
    """Evaluación por (nivel, fase), niveles desde config.json y columnas val_* en el historial del mejor modelo."""
    from agents.callbacks import BestModelCallback
    from agents.evaluate import describe, evaluate_policy_phases
    from agents.runs import level_kwargs
    cfg = {"train_levels": {"mini_a": 1.0, "level_1": 2.0}, "val_levels": ["mini_b"]}
    train_kw, val_kw = level_kwargs(cfg), level_kwargs(cfg, "val")
    assert [lv.name for lv in train_kw["level"]] == ["mini_a", "level_1"] and train_kw["level_weights"] == [1.0, 2.0]
    assert level_kwargs({}) == {} and level_kwargs({}, "val") is None          # config vieja: como antes

    class NeverJump:                                            # política que nunca salta: muere en el 1er obstáculo
        def predict(self, obs, deterministic=True):
            return 0, None

    rows = evaluate_policy_phases(NeverJump(), "features", phases=[2.0, 2.5], **train_kw)
    assert [(r["level"], r["x0"]) for r in rows] == [("mini_a", 2.0), ("mini_a", 2.5), ("level_1", 2.0), ("level_1", 2.5)]
    assert not any(r["success"] for r in rows)
    assert [r["oracle_jumps"] for r in rows if r["level"] == "level_1"] == [13, 13]    # oráculo de cada nivel
    assert 0 < rows[0]["oracle_jumps"] < 13
    assert "mini_a 0/2" in describe(rows) and "level_1 0/2" in describe(rows)

    venv, model = ppo_on_vec_env()
    cb = BestModelCallback(eval_freq=512, save_dir=tmp_path, phases=[2.0], val_env_kwargs=val_kw, **train_kw)
    model.learn(total_timesteps=512, callback=cb)
    header = (tmp_path / "eval_history.csv").read_text().splitlines()[0].split(",")
    assert {"success", "val_success", "val_progress", "val_mean_jumps_won", "is_best"} <= set(header)
    assert {"success_mini_a", "success_level_1", "val_success_mini_b"} <= set(header)     # éxito por nivel


# ------------------------------------------------------------------ progresión de niveles
def test_progression_in_level_file(registered_minis, tmp_path):
    import json
    from gd_env import resolve_levels

    def spec(**data):
        f = tmp_path / f"p{len(list(tmp_path.iterdir()))}.json"
        f.write_text(json.dumps(data))
        return str(f)

    out = resolve_levels([spec(niveles=["level_1", "mini_a", "mini_b"], validacion="mini_b",
                               progresion={"etapas": [["level_1"], ["mini_a", "mini_b"]], "ventana": 10})])
    # mini_b es de validación: sale de su etapa; los valores que faltan toman el valor por defecto
    assert out["progression"] == {"etapas": [["level_1"], ["mini_a"]], "desbloquear": 0.8, "ventana": 10, "piso": 0.2}
    with pytest.raises(ValueError):            # un nivel de entrenamiento sin etapa
        resolve_levels([spec(niveles=["level_1", "mini_a"], progresion={"etapas": [["level_1"]]})])
    with pytest.raises(ValueError):            # un nivel en dos etapas
        resolve_levels([spec(niveles=["level_1", "mini_a"], progresion={"etapas": [["level_1", "mini_a"], ["mini_a"]]})])
    with pytest.raises(ValueError):            # clave mal escrita
        resolve_levels([spec(niveles=["level_1"], progresion={"etapas": [["level_1"]], "piso_": 0.1})])


def test_level_progression_logic():
    """Desbloqueo por etapas, piso, peso máximo para niveles sin datos y recuperación de un nivel olvidado."""
    from agents.callbacks import LevelProgression
    p = LevelProgression(["a", "b", "c"], [1.0, 1.0, 2.0], [["a"], ["b", "c"]], unlock=0.75, window=4, floor=0.2)
    assert p.weights() == [1.0, 0.0, 0.0]                         # sólo la primera etapa; sin datos = peso máximo
    assert not any(p.observe("a", True) for _ in range(3))        # ventana incompleta: no decide
    assert p.weights() == [pytest.approx(0.2), 0.0, 0.0]          # domina "a": baja al piso, nunca a 0
    assert p.observe("a", False) and p.stage == 1                 # 3/4 = 0.75 >= unlock: se desbloquea la etapa 2
    assert p.weights() == [pytest.approx(0.25), 1.0, 2.0]         # nuevos sin datos: peso máximo (x peso base)
    for _ in range(4):
        p.observe("b", True)
        p.observe("c", True)
    assert p.stage == 1                                           # última etapa: no hay más que desbloquear
    for _ in range(4):
        p.observe("a", False)                                     # olvida "a"...
    assert p.weights()[0] == pytest.approx(1.0)                   # ...y su peso vuelve al máximo
    assert p.weights()[1:] == [pytest.approx(0.2), pytest.approx(0.4)]


def test_level_progression_reaches_inner_env(tmp_path):
    """Integrado con SB3: los pesos llegan al entorno de adentro y la columna 'level' queda en monitor/."""
    from agents import build_model, make_train_env
    from agents.callbacks import LevelProgression
    a, b = mini_levels()
    prog = LevelProgression(["mini_a", "mini_b"], [1.0, 1.0], [["mini_a"], ["mini_b"]], window=2,
                            log_path=tmp_path / "progresion.csv")
    venv = make_train_env("features", n_envs=2, seed=0, random_phase=False, monitor_dir=str(tmp_path / "mon"),
                          level=[a, b], level_weights=prog.weights())
    model = build_model("ppo", venv, seed=0)
    model.learn(total_timesteps=512, callback=prog)
    assert [e.unwrapped.level_weights for e in venv.envs] == [prog.weights()] * 2
    assert prog.weights()[1] == 0.0                               # sin entrenar no gana: mini_b sigue bloqueado
    assert "level" in (tmp_path / "mon" / "0.monitor.csv").read_text().splitlines()[1]
    assert {"mini_b"} != {l.split(",")[-1] for l in (tmp_path / "mon" / "0.monitor.csv").read_text().splitlines()[2:]}
    assert (tmp_path / "progresion.csv").read_text().startswith("timesteps,event,stage,w_mini_a,w_mini_b")


# ------------------------------------------------------------------ términos de la recompensa (--rewards)
from gd_env import build_level

SANTI_LEVELS = ["santi_facil", "santi_intermedio_1", "santi_intermedio_2", "santi_intermedio_dificil", "santi_dificil"]


@pytest.mark.parametrize("name", SANTI_LEVELS)
def test_santi_levels_registered_and_solvable(name):
    lv = gd_env.get_level(name)
    assert lv.height == level_1().height and lv.start_x == 2.0
    assert gd_env.solve(lv, ph.Physics()) is not None


def test_rewards_default_is_unchanged():
    """Sin rewards (o con los términos por defecto) la recompensa es exactamente la de siempre."""
    pol = lambda o, e: int(e.state.t % 7 == 0)
    _, r0 = run(GeometryDashEnv(obs_mode="features", jump_penalty=0.05), pol)
    _, r1 = run(GeometryDashEnv(obs_mode="features", jump_penalty=0.05, rewards=gd_env.DEFAULT_TERMS), pol)
    assert r1 == pytest.approx(r0)


def test_rewards_terms_can_be_switched_off():
    pol = lambda o, e: 0                       # nunca salta: choca
    _, r_all = run(GeometryDashEnv(obs_mode="features"), pol)
    _, r_nodeath = run(GeometryDashEnv(obs_mode="features", rewards=["progress", "win"]), pol)
    assert r_nodeath == pytest.approx(r_all + 10.0)             # sin el término death no hay castigo
    _, r_noprog = run(GeometryDashEnv(obs_mode="features", rewards=["death"]), pol)
    assert r_noprog == pytest.approx(-10.0)                     # sólo el castigo por morir
    win = GeometryDashEnv(obs_mode="features", rewards=["progress", "death"])    # sin win: ganar no suma el bonus
    opol = OraclePolicy(win)
    _, ret = run(win, lambda o, e: opol.predict(o)[0])
    assert ret == pytest.approx(1.0)
    # jump_penalty apagado: el valor no se cobra (aunque venga con jump_penalty > 0)
    jp = GeometryDashEnv(obs_mode="features", jump_penalty=0.1, rewards=["progress", "win", "death"])
    opol = OraclePolicy(jp)
    _, ret = run(jp, lambda o, e: opol.predict(o)[0])
    assert ret == pytest.approx(2.0)


def test_rewards_validation_errors():
    with pytest.raises(ValueError, match="excluyentes"):
        GeometryDashEnv(rewards=["progress", "jump_penalty", "salto_preciso"])
    with pytest.raises(ValueError, match="excluyentes"):
        GeometryDashEnv(rewards=["progress", "salto_preciso"], jump_penalty=0.05)
    with pytest.raises(ValueError, match="desconocido"):
        GeometryDashEnv(rewards=["progress", "inventado"])
    with pytest.raises(ValueError, match="vacío"):
        GeometryDashEnv(rewards=[])
    with pytest.raises(ValueError, match="alternativas"):
        GeometryDashEnv(rewards=["progress"], reward_fn=lambda tr: 0.0)


def test_salto_preciso_term_values():
    """Piso liso -> -0.06; un solo pincho -> gratis; pinchos seguidos (^^^) -> +0.03 por cada uno; sin salto -> 0."""
    from gd_env import Transition
    from gd_env.physics import State, OK
    flat = build_level(length=40, height=7, name="plano")
    spikes3 = build_level(length=40, height=7, spikes=[12, 13, 14], name="tres")
    spike1 = build_level(length=40, height=7, spikes=[12], name="uno")

    def tr(lv, x, jumped=True):
        s = ph.initial_state(lv, ph.Physics(), x)
        return Transition(s, s, OK, lv, jumped)

    assert gd_env.salto_preciso_term(tr(flat, 5.0)) == pytest.approx(-0.06)
    assert gd_env.salto_preciso_term(tr(spike1, 10.0)) == 0.0
    assert gd_env.salto_preciso_term(tr(spikes3, 11.0)) == pytest.approx(0.09)
    assert gd_env.salto_preciso_term(tr(spikes3, 11.0, jumped=False)) == 0.0


def test_salto_preciso_in_env():
    pol = lambda o, e: int(e.state.t % 5 == 0)
    lv = build_level(length=40, height=7, spikes=[20], name="un_pincho")
    base = GeometryDashEnv(level=lv, obs_mode="features", rewards=["progress", "win", "death"])
    sp = GeometryDashEnv(level=lv, obs_mode="features", rewards=["progress", "win", "death", "salto_preciso"])
    _, r_base = run(base, pol)
    _, r_sp = run(sp, pol)
    assert r_sp != pytest.approx(r_base)        # el término suma algo (saltos en piso liso)


def test_rewards_survive_through_config():
    pytest.importorskip("stable_baselines3")
    from agents.runs import reward_kwargs
    kw = reward_kwargs({"jump_penalty": 0.0, "rewards": ["progress", "win", "death", "salto_preciso"]})
    assert kw["rewards"] == ["progress", "win", "death", "salto_preciso"]
    assert GeometryDashEnv(**kw).rewards == ("progress", "win", "death", "salto_preciso")
    assert "rewards" not in reward_kwargs({"jump_penalty": 0.0})


def test_train_validates_rewards_before_training(tmp_path):
    pytest.importorskip("stable_baselines3")
    from agents import train
    kw = dict(out=str(tmp_path), timesteps=10, seeds=[0])
    with pytest.raises(ValueError, match="excluyentes"):
        train("ppo", rewards=["progress", "jump_penalty", "salto_preciso"], **kw)
    with pytest.raises(ValueError, match="excluyentes"):
        train("ppo", rewards=["progress", "salto_preciso"], jump_penalty=0.05, **kw)
    with pytest.raises(ValueError, match="death_penalty"):          # parámetro de un término apagado
        train("ppo", rewards=["progress", "win"], death_penalty=5.0, **kw)
    with pytest.raises(ValueError, match="jump_penalty"):
        train("ppo", rewards=["progress", "win", "death"], jump_penalty=0.05, ramp=True, **kw)
    with pytest.raises(ValueError, match="alternativas"):
        train("ppo", rewards=["progress"], reward_fn="ejemplos/reward_santi.py:salto_preciso", **kw)
    assert not list(tmp_path.iterdir())          # falló antes de crear la carpeta de la corrida


@pytest.mark.parametrize("name", SANTI_LEVELS)
def test_santi_levels_solvable_from_eval_phases(name):
    lv = gd_env.get_level(name)
    for x0 in np.linspace(2.0, 2.95, 10):       # las mismas fases que EVAL_PHASES (agents.common)
        start = ph.initial_state(lv, ph.Physics(), float(x0))
        assert gd_env.solve(lv, ph.Physics(), start) is not None, f"{name} sin solución desde x0={x0:.2f}"


def test_franco_sin_saltar_wins_only_without_jumping():
    """Techo de pinchos colgantes: no saltar gana; saltar (aunque sea una vez) mata."""
    lv = gd_env.get_level("franco_sin_saltar")
    assert gd_env.solve(lv, ph.Physics()).n_jumps == 0
    for x0 in np.linspace(2.0, 2.95, 10):
        info, _ = run(GeometryDashEnv(level=lv, obs_mode="features"), lambda o, e: 0, options={"x0": float(x0)})
        assert info["success"], f"no saltar debería ganar desde x0={x0:.2f}"
    for t_jump in (0, 5, 17, 40, 100):          # un solo salto, en distintos momentos (también en la fase de arranque)
        info, _ = run(GeometryDashEnv(level=lv, obs_mode="features"), lambda o, e, t=t_jump: int(e.state.t == t))
        assert not info["success"] and info["status"] == ph.SPIKE_HIT, f"saltar en t={t_jump} debería matar"

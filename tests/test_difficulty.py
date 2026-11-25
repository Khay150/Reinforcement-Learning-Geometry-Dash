"""Tests de gd_env/difficulty.py y scripts/difficulty.py: las métricas se comprueban por separado con el oráculo y
simulando la física (no con el mismo código que las calcula)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import gd_env
from gd_env import Oracle, Physics, build_level, difficulty_over_phases, level_1, level_difficulty, parse_level, solve
from gd_env import physics as ph

P = Physics()


def play(level, takeoffs: list[int]):
    """Salta exactamente en los ticks ``takeoffs`` (y en ningún otro) hasta el último, y después juega el oráculo.
    Devuelve (ganó, saltos efectivos, si el cubo estaba apoyado en cada tick pedido)."""
    s, oracle = ph.initial_state(level, P), Oracle(level, P)
    jumps, grounded, last = 0, [], max(takeoffs)
    while True:
        if s.t <= last:
            a = int(s.t in takeoffs)
            if a:
                grounded.append(s.on_ground)
        else:
            a = oracle.action(s)
        jumps += int(a == 1 and s.on_ground)
        s, status = ph.step(level, P, s, a)
        if status != ph.OK:
            return status == ph.WIN, jumps, all(grounded)


def test_n_jumps_matches_oracle():
    for name in gd_env.LEVELS:
        lv = gd_env.get_level(name)
        for x0 in (2.0, 2.5, 2.95):
            d = level_difficulty(lv, x0)
            assert d is not None and d.n_jumps == solve(lv, P, ph.initial_state(lv, P, x0)).n_jumps
            assert len(d.jumps) == d.n_jumps


def test_windows_checked_with_oracle():
    """Despegar en los bordes de cada ventana gana con los saltos mínimos; un tick antes o después (si el cubo está
    apoyado) ya no: o pierde o necesita saltos de más."""
    lv = level_1()
    d = level_difficulty(lv)
    centers = [j.tick for j in d.jumps]
    for i, j in enumerate(d.jumps):
        for t in (j.first_tick, j.tick, j.last_tick):
            won, n, grounded = play(lv, centers[:i] + [t])
            assert grounded and won and n == d.n_jumps, (i, t)
        for t in (j.first_tick - 1, j.last_tick + 1):
            won, n, grounded = play(lv, centers[:i] + [t])
            if grounded:                                    # en el aire saltar no hace nada: no es un despegue
                assert not (won and n == d.n_jumps), (i, t)


def test_level_1_known_values():
    """Los dos pinchos dobles (x≈23 y x≈65) dan 3 ticks de margen; el salto desde la plataforma baja, 2."""
    d = level_difficulty(level_1())
    by_x = {round(j.x): j.window for j in d.jumps}
    assert by_x[23] == 3 and by_x[65] == 3 and by_x[78] == 2
    assert d.summary()["min_window"] == 2


def test_single_spike_is_easy():
    d = level_difficulty(build_level(length=30, height=7, spikes=[12], name="un_pincho"))
    assert d.n_jumps == 1 and d.max_chain == 1 and d.jumps[0].traps == 0 and not d.jumps[0].linked_next
    assert d.jumps[0].window >= 5
    assert 0 < d.p_random_win < 1


def test_chain_of_close_spikes_vs_far_spikes():
    """Pinchos a 4 columnas: cuándo se salta uno cambia la ventana del siguiente (cadena de 3). Lejos: independientes."""
    close = level_difficulty(build_level(length=40, height=7, spikes=[12, 16, 20], name="cerca"))
    far = level_difficulty(build_level(length=70, height=7, spikes=[12, 30, 48], name="lejos"))
    assert close.n_jumps == far.n_jumps == 3
    assert close.max_chain == 3 and close.summary()["linked_jumps"] == 3
    assert far.max_chain == 1 and far.summary()["linked_jumps"] == 0


def test_traps_checked_with_oracle():
    """Un escalón: despegar demasiado temprano deja al cubo aterrizando pegado a la pared, vivo pero sin salida.
    Se cuentan esas trampas a mano (simulando cada despegue y preguntándole al oráculo) y se comparan."""
    lv = build_level(length=40, height=7, blocks=[(20, 1), (21, 1), (22, 1)], name="escalon")
    d = level_difficulty(lv)
    assert d.n_jumps == 1
    oracle, traps = Oracle(lv, P), 0
    s = ph.initial_state(lv, P)
    while True:                                          # cada tick apoyado de la espera (hasta chocar el escalón)
        if s.on_ground:
            f, status = ph.step(lv, P, s, 1)
            while status == ph.OK and not f.on_ground:   # vuelo hasta aterrizar
                f, status = ph.step(lv, P, f, 0)
            traps += int(status == ph.OK and not oracle.is_winnable(f))
        s, status = ph.step(lv, P, s, 0)
        if status != ph.OK:
            break
    assert traps > 0 and d.jumps[0].traps == traps


def test_random_win_probability():
    flat = level_difficulty(build_level(length=30, height=7, name="plano"))
    assert flat.n_jumps == 0 and flat.jumps == [] and flat.p_random_win == pytest.approx(1.0)
    assert flat.summary()["min_window"] is None and flat.max_chain == 0
    # Monte Carlo con una política 50/50: tiene que coincidir con el valor exacto.
    lv = build_level(length=24, height=7, spikes=[10], name="mc")
    exact = level_difficulty(lv).p_random_win
    rng, wins, n = np.random.default_rng(0), 0, 2000
    for _ in range(n):
        s, status = ph.initial_state(lv, P), ph.OK
        while status == ph.OK:
            s, status = ph.step(lv, P, s, int(rng.integers(2)))
        wins += status == ph.WIN
    assert abs(wins / n - exact) < 4 * np.sqrt(exact * (1 - exact) / n)


def test_unsolvable_level():
    wall = build_level(length=30, height=7, blocks=[(10, 4)], name="pared")
    assert level_difficulty(wall) is None
    row = difficulty_over_phases(wall, [2.0, 2.5])
    assert row["solvable_phases"] == "0/2" and row["min_window"] is None


def test_over_phases_takes_worst_case():
    lv = level_1()
    phases = [2.0, 2.5]
    rows = [level_difficulty(lv, x0).summary() for x0 in phases]
    agg = difficulty_over_phases(lv, phases)
    assert agg["solvable_phases"] == "2/2"
    assert agg["min_window"] == min(r["min_window"] for r in rows)
    assert agg["log10_p_random"] == min(r["log10_p_random"] for r in rows)


# ------------------------------------------------------------------ scripts/difficulty.py
def _script():
    spec = importlib.util.spec_from_file_location("difficulty_script", ROOT / "scripts" / "difficulty.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_phases_match_eval_phases():
    pytest.importorskip("stable_baselines3")
    from agents.common import EVAL_PHASES
    assert _script().eval_phases(level_1(), 10) == pytest.approx(EVAL_PHASES)


def test_script_loads_names_txt_and_json(tmp_path):
    script = _script()
    txt = tmp_path / "mio.txt"
    txt.write_text("\n".join(["." * 30] * 5 + ["..........^" + "." * 19, "#" * 30]))
    js = tmp_path / "set.json"
    js.write_text(json.dumps({"niveles": {"level_1": 1.0}, "validacion": None}))
    levels = script.load_levels(["level_1", str(txt), str(js)])
    assert [lv.name for lv in levels] == ["level_1", "mio", "level_1"]
    rows = [difficulty_over_phases(lv, script.eval_phases(lv, 2)) for lv in levels[:2]]
    table = script.summary_table(rows)
    assert "| level_1 | 2/2 | 13 |" in table and "| mio | 2/2 | 1 |" in table
    assert "saltos mínimos: 13" in script.detail_table(levels[0])
    with pytest.raises(ValueError):
        script.load_levels(["no_existe"])

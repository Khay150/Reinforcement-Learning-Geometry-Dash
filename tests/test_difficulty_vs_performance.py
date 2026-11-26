"""Tests de scripts/difficulty_vs_performance.py con corridas sintéticas (carpetas armadas a mano, sin entrenar)."""
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("dvp", ROOT / "scripts" / "difficulty_vs_performance.py")
dvp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dvp)


def write_monitor(path: Path, episodes: list[tuple[int, bool]], level: str = "x", partial: bool = False):
    """``monitor/<i>.monitor.csv`` como lo deja el Monitor de SB3 (con info_keywords success, progress, level)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['#{"t_start": 0, "env_id": "None"}', "r,l,t,success,progress,level"]
    lines += [f"{1.0 if w else -1.0},{l},{i * 0.1:.2f},{w},{1.0 if w else 0.4},{level}"
              for i, (l, w) in enumerate(episodes)]
    text = "\n".join(lines) + "\n"
    if partial:
        text += "-0.5,1"                    # línea a medio escribir (la corrida sigue entrenando)
    path.write_text(text)


def make_run(root: Path, name: str, level: str, seeds: dict, timesteps: int = 1000, levels_cfg=None) -> Path:
    """Corrida sintética. ``seeds``: seed -> dict con ``envs`` (lista, por entorno, de episodios (largo, ganó)),
    y opcionales ``success``/``progress`` (eval final), ``best`` (éxito del mejor), ``history`` (lista de
    (pasos, éxito)), ``complete`` (hay model.zip) y ``no_success_col``."""
    run = root / name
    run.mkdir(parents=True)
    cfg = {"algos": ["ppo"], "seeds": list(seeds), "timesteps": timesteps, "obs": "features", "n_envs": 8}
    if levels_cfg is None:
        cfg.update(levels=[level], train_levels={level: 1.0}, val_levels=[level])
    else:
        cfg.update(levels_cfg)
    (run / "config.json").write_text(json.dumps(cfg))
    for seed, spec in seeds.items():
        d = run / "ppo" / f"seed{seed}"
        for i, eps in enumerate(spec["envs"]):
            write_monitor(d / "monitor" / f"{i}.monitor.csv", eps, level, partial=spec.get("partial", False))
            if spec.get("no_success_col"):
                f = d / "monitor" / f"{i}.monitor.csv"
                f.write_text("\n".join(",".join(c.split(",")[:3]) if not c.startswith("#") else c
                                       for c in f.read_text().splitlines()) + "\n")
        if spec.get("complete", True):
            (d / "model.zip").write_bytes(b"")
            pd.DataFrame({"level": level, "x0": [2.0, 2.5], "success": [spec.get("success", 0)] * 2,
                          "progress": [spec.get("progress", 0.5)] * 2}).to_csv(d / "eval_fases.csv", index=False)
            if "best" in spec:
                pd.DataFrame({"level": level, "x0": [2.0, 2.5], "success": [spec["best"]] * 2,
                              "progress": [1.0] * 2}).to_csv(d / "eval_fases_best.csv", index=False)
        if "history" in spec:
            pd.DataFrame([{"timesteps": t, "success": s, "progress": 0.5, "is_best": 0} for t, s in spec["history"]]
                         ).to_csv(d / "eval_history.csv", index=False)
    return run


# ------------------------------------------------------------------------------------------- pasos hasta 1ª victoria
def test_first_win_counts_parallel_envs_in_lockstep(tmp_path):
    # 2 entornos: el 0 gana al terminar su 2º episodio (lleva 10 + 20 = 30 pasos) -> VecEnv en 60;
    # el 1 gana en su 3º (5 + 5 + 7 = 17 pasos) -> VecEnv en 34. La primera victoria es la del entorno 1.
    run = make_run(tmp_path, "r", "franco_facil",
                   {0: {"envs": [[(10, False), (20, True)], [(5, False), (5, False), (7, True)]]}})
    perf = dvp.seed_performance(run / "ppo" / "seed0", "franco_facil", 1000)
    assert perf["pasos_1a_victoria"] == 34 and not perf["censurada_1a_victoria"]
    assert perf["fuente_1a_victoria"] == "monitor"
    assert perf["pasos_observados"] == 2 * 30


def test_never_wins_is_censored_at_total_timesteps(tmp_path):
    run = make_run(tmp_path, "r", "franco_facil", {0: {"envs": [[(10, False)] * 5, [(10, False)] * 5]}},
                   timesteps=5000)
    perf = dvp.seed_performance(run / "ppo" / "seed0", "franco_facil", 5000)
    assert perf["pasos_1a_victoria"] == 5000 and perf["censurada_1a_victoria"]
    assert perf["pasos_aprende"] == 5000 and perf["censurada_aprende"]
    assert perf["pasos_eval_gana"] == 5000 and perf["censurada_eval_gana"]


def test_learned_at_uses_rolling_success_over_all_envs():
    ep = pd.DataFrame({"pasos": [10, 20, 30, 40, 50], "gano": [False, True, False, True, True], "env": 0})
    assert dvp.learned_at(ep, window=2, threshold=1.0) == 50          # dos victorias seguidas recién en 40-50
    assert dvp.learned_at(ep, window=2, threshold=0.5) == 20
    assert dvp.learned_at(ep, window=10, threshold=0.5) is None         # menos episodios que la ventana
    assert dvp.first_train_win(ep) == 20


def test_eval_history_and_best_and_final(tmp_path):
    run = make_run(tmp_path, "r", "franco_facil",
                   {0: {"envs": [[(10, True)]], "success": 1, "progress": 1.0, "best": 0.5,
                        "history": [(100, 0.0), (200, 0.3), (300, 1.0)]}})
    perf = dvp.seed_performance(run / "ppo" / "seed0", "franco_facil", 300)
    assert perf["pasos_eval_gana"] == 200 and not perf["censurada_eval_gana"]
    assert perf["exito_final"] == 100 and perf["progreso_final"] == 100 and perf["exito_mejor"] == 50


def test_monitor_without_success_falls_back_to_eval_history(tmp_path):
    run = make_run(tmp_path, "r", "franco_facil",
                   {0: {"envs": [[(10, True)]], "no_success_col": True, "history": [(100, 0.0), (200, 0.1)]}})
    perf = dvp.seed_performance(run / "ppo" / "seed0", "franco_facil", 300)
    assert perf["fuente_1a_victoria"] == "eval_history" and perf["pasos_1a_victoria"] == 200


def test_partial_monitor_line_and_incomplete_seeds(tmp_path):
    make_run(tmp_path, "r", "franco_facil", {
        0: {"envs": [[(10, False), (10, True)]]},
        1: {"envs": [[(10, False), (10, False)]], "complete": False, "partial": True},
    })
    runs = dvp.expand_runs([str(tmp_path / "r*")])
    seeds = dvp.collect(runs, n_phases=1)
    assert list(seeds["seed"]) == ["0"]                       # la seed 1 sigue entrenando: se saltea
    seeds = dvp.collect(runs, n_phases=1, include_incomplete=True)
    s1 = seeds[seeds["seed"] == "1"].iloc[0]
    assert s1["censurada_1a_victoria"] and s1["pasos_1a_victoria"] == 20     # censurada en lo observado (1 env x 20)
    assert math.isnan(s1["exito_final"])


# ------------------------------------------------------------------------------------------------------- corridas
def test_level_from_config_and_skips(tmp_path):
    assert dvp.train_level_names({"train_levels": {"nacho_medio": 1.0}}) == ["nacho_medio"]
    assert dvp.train_level_names({"timesteps": 10}) == ["level_1"]      # corrida vieja sin niveles
    make_run(tmp_path, "multi", "franco_facil", {0: {"envs": [[(5, True)]]}},
             levels_cfg={"train_levels": {"franco_facil": 1.0, "franco_medio": 1.0}})
    make_run(tmp_path, "uno", "franco_facil", {0: {"envs": [[(5, True)]]}})
    seeds = dvp.collect(dvp.expand_runs([str(tmp_path)]), n_phases=1)     # carpeta padre: toma las corridas de adentro
    assert set(seeds["corrida"]) == {"uno"}                                 # la de varios niveles se saltea


# ------------------------------------------------------------------------------------------------------- Spearman
def test_spearman_matches_pandas_and_exact_p():
    rho, p, n = dvp.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == pytest.approx(1.0) and n == 4
    assert p == pytest.approx(2 / 24)                       # sólo la identidad y la inversa llegan a |ρ| = 1
    assert dvp.spearman([1, 2, 3, 4], [4, 3, 2, 1])[0] == pytest.approx(-1.0)
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=12), rng.normal(size=12)
    y[:3] = y[0]                                            # con empates
    assert dvp.spearman(x, y)[0] == pytest.approx(pd.DataFrame({"x": x, "y": y}).corr("spearman").loc["x", "y"])
    assert math.isnan(dvp.spearman([1, 2], [3, 4])[0])
    assert math.isnan(dvp.spearman([1, 2, 3], [5, 5, 5])[0])
    rho, _, n = dvp.spearman([1, 2, np.nan, 4], [1, 2, 3, 4])
    assert n == 3 and rho == pytest.approx(1.0)


def test_spearman_p_exact_brute_force():
    x, y = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
    rho, p, _ = dvp.spearman(x, y)
    rx = np.array(x, float)
    null = [np.corrcoef(rx, perm)[0, 1] for perm in itertools.permutations(y)]
    assert p == pytest.approx(np.mean(np.abs(null) >= abs(rho) - 1e-12))


# ---------------------------------------------------------------------------------------------------- de punta a punta
def test_main_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    # Más difícil -> gana más tarde y con menos éxito.
    make_run(runs, "e7_nivel_franco_facil", "franco_facil",
             {0: {"envs": [[(10, True)] * 60], "success": 1, "progress": 1.0, "best": 1},
              1: {"envs": [[(20, False), (10, True)] + [(10, True)] * 60], "success": 1, "progress": 1.0}})
    make_run(runs, "e7_nivel_level_1", "level_1",
             {0: {"envs": [[(50, False)] * 5 + [(50, True)] * 60], "success": 0.5, "progress": 0.8},
              1: {"envs": [[(50, False)] * 8 + [(50, True)]], "success": 0.2, "progress": 0.6}})
    make_run(runs, "e7_nivel_nacho_muy_dificil", "nacho_muy_dificil",
             {0: {"envs": [[(60, False)] * 10], "success": 0, "progress": 0.3},
              1: {"envs": [[(60, False)] * 12], "success": 0, "progress": 0.2}}, timesteps=1000)
    out = tmp_path / "fig"
    res = dvp.main([str(runs / "e7_nivel_*"), "--out", str(out), "--fases", "1"])
    agg = pd.read_csv(out / "tabla.csv")
    assert len(agg) == 3 and set(agg["nivel"]) == {"franco_facil", "level_1", "nacho_muy_dificil"}
    for f in ("tabla.md", "semillas.csv", "correlaciones.csv", "panel.png",
              "log10_p_random_vs_pasos_1a_victoria.png", "min_window_vs_exito_final.png"):
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    corr = res["corr"].set_index(["dificultad", "desempeno"])
    assert corr.loc[("log10_p_random", "pasos_1a_victoria"), "rho"] == pytest.approx(-1.0)
    assert corr.loc[("log10_p_random", "exito_final"), "rho"] == pytest.approx(1.0)
    md = (out / "tabla.md").read_text()
    assert "nunca" in md and "nacho_muy_dificil" in md
    with pytest.raises(SystemExit):
        dvp.main([str(runs / "e7_nivel_*"), "--out", str(out), "--pares", "foo:bar"])

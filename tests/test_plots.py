"""Tests de agents/plots.py y scripts/plot.py con corridas de mentira (carpetas armadas a mano en tmp_path): una
corrida completa (2 seeds, varios niveles, progresión, rampa) y una "en curso" (1 seed, monitor cortado, sin
evaluaciones). Son rápidos: no entrenan nada."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from agents.evaluate import metrics
from agents.plots import graficar, leer_csv, leer_monitor, media_movil, metricas_eval, nombre_salida

PNG = b"\x89PNG"


def _csv(path: Path, header: list[str], rows: list[list], comment: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ([comment] if comment else []) + [",".join(header)] + [",".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def _monitor(seed_dir: Path, n_envs: int, n_eps: int, t_start: float = 1000.0, levels=("level_1",), cortar=False):
    """``n_envs`` archivos de monitor; los episodios mejoran con el tiempo (para que la curva suba)."""
    rng = np.random.default_rng(0)
    for e in range(n_envs):
        rows = []
        for k in range(n_eps):
            win = rng.random() < k / n_eps
            rows.append([round(1.0 if win else -1.0 + k / n_eps, 3), 100, round(0.5 * k + 0.01 * e, 3), win,
                         1.0 if win else round(k / n_eps, 3), levels[k % len(levels)]])
        path = seed_dir / "monitor" / f"{e}.monitor.csv"
        _csv(path, ["r", "l", "t", "success", "progress", "level"], rows,
             comment=f'#{{"t_start": {t_start + 0.001 * e}, "env_id": "None"}}')
        if cortar and e == 0:                 # última línea a medio escribir (corrida en curso)
            with open(path, "a") as f:
                f.write("-0.5,40")


def _eval(path: Path, levels: list[str], wins: int):
    rows = []
    for lv in levels:
        for i, x0 in enumerate(np.linspace(2.0, 2.95, 10)):
            ok = i < wins
            rows.append([lv, round(x0, 3), int(ok), 1.0 if ok else 0.5, 1.0, 300, "win" if ok else "crash",
                         14 if ok else 5, 7, "" if ok else 40.0])
    _csv(path, ["level", "x0", "success", "progress", "return", "ticks", "status", "n_jumps", "oracle_jumps",
                "death_x"], rows)


def _corrida_completa(root: Path) -> Path:
    run = root / "multi_prog_ramp"
    levels = ["facil", "dificil"]
    cfg = {"algos": ["ppo"], "seeds": [0, 1], "jump_penalty": 0.05, "ramp": True,
           "train_levels": {lv: 1.0 for lv in levels}, "val_levels": ["level_1"],
           "progression": {"etapas": [["facil"], ["dificil"]], "desbloquear": 0.8, "ventana": 50, "piso": 0.2}}
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(cfg))
    for seed in (0, 1):
        d = run / "ppo" / f"seed{seed}"
        _monitor(d, n_envs=4, n_eps=60, levels=levels)
        _eval(d / "eval_fases.csv", levels, wins=5 + seed)
        _eval(d / "eval_fases_best.csv", levels, wins=8)
        _eval(d / "eval_fases_val.csv", ["level_1"], wins=2 * seed)
        _eval(d / "eval_fases_best_val.csv", ["level_1"], wins=3)
        _csv(d / "eval_history.csv", ["timesteps", "success", "progress", "val_success", "val_progress", "is_best"],
             [[t, t / 30000, 0.5 + t / 60000, t / 60000, 0.3, 1] for t in (5000, 10000, 15000, 20000)])
        _csv(d / "progresion.csv", ["timesteps", "event", "stage", "w_facil", "w_dificil", "success_facil",
                                    "success_dificil"],
             [[0, "inicio", 0, 1.0, 0.0, 0.0, 0.0], [4000, "pesos", 0, 0.6, 0.0, 0.4, 0.0],
              [8000 + 1000 * seed, "etapa 2/2", 1, 0.2, 1.0, 0.82, 0.0], [16000, "pesos", 1, 0.2, 0.7, 0.9, 0.3]])
        _csv(d / "jump_penalty.csv", ["timesteps", "jump_penalty", "success_rate"],
             [[0, 0.0, ""], [9000, 0.01, 0.85], [15000, 0.02, 0.9], [19000, 0.01, 0.35]])
    return run


def _corrida_en_curso(root: Path) -> Path:
    """DQN, 1 entorno, 1 seed, sin config.json ni evaluaciones; además una seed que todavía no arrancó."""
    run = root / "dqn_en_curso"
    _monitor(run / "dqn" / "seed0", n_envs=1, n_eps=50, cortar=True)
    (run / "dqn" / "seed1").mkdir(parents=True)
    return run


def test_leer_monitor_junta_entornos_por_tiempo(tmp_path):
    d = tmp_path / "seed0"
    _csv(d / "monitor" / "0.monitor.csv", ["r", "l", "t", "success", "progress"],
         [[1, 10, 1.0, True, 1.0], [2, 20, 3.0, False, 0.5]], comment='#{"t_start": 100.0}')
    _csv(d / "monitor" / "1.monitor.csv", ["r", "l", "t", "success", "progress"],
         [[3, 5, 0.5, False, 0.2]], comment='#{"t_start": 101.0}')       # termina en 101.5: entre los dos de arriba
    with open(d / "monitor" / "1.monitor.csv", "a") as f:
        f.write("4,7")                                                    # fila cortada: se ignora
    m = leer_monitor(d)
    assert m["retorno"].tolist() == [1, 3, 2]
    assert m["pasos"].tolist() == [10, 15, 35]
    assert m["exito"].tolist() == [1, 0, 0]
    assert leer_monitor(tmp_path / "no_existe") is None


def test_media_movil_por_pasos():
    pasos = np.array([10.0, 20.0, 30.0, 40.0])
    vals = np.array([0.0, 1.0, 1.0, 0.0])
    out = media_movil(pasos, vals, np.array([5.0, 20.0, 30.0, 40.0, 50.0]), ancho=15)
    assert np.isnan(out[0]) and np.isnan(out[-1])        # antes del primer episodio y después del último: sin datos
    assert out[1:4].tolist() == [0.5, 1.0, 0.5]


def test_metricas_eval_coincide_con_evaluate(tmp_path):
    _eval(tmp_path / "e.csv", ["a", "b"], wins=6)
    rows = leer_csv(tmp_path / "e.csv")
    m = metricas_eval(rows)
    ref = metrics([{k: float(v) if k in ("success", "progress", "n_jumps") else v for k, v in r.items()}
                   for r in rows])
    assert m["exito"] == pytest.approx(ref["success"])
    assert m["progreso"] == pytest.approx(ref["progress"])
    assert m["saltos"] == pytest.approx(2.0)                # 14 saltos contra 7 del oráculo en las ganadas


def test_graficar_todo(tmp_path):
    runs = [_corrida_completa(tmp_path), _corrida_en_curso(tmp_path)]
    out = tmp_path / "figs"
    paths = graficar(runs, out, etiquetas=["completa", "en curso"])
    nombres = {p.name for p in paths}
    assert {"curva_retorno.png", "curva_exito.png", "curva_progreso.png", "eval_exito.png", "eval_progreso.png",
            "barras_entrenamiento.png", "barras_validacion.png", "barras_niveles.png", "penalizacion_salto.png",
            "progresion_multi_prog_ramp_ppo.png"} <= nombres
    for p in paths:
        assert p.parent == out and p.read_bytes()[:4] == PNG


def test_una_seed_y_tipos(tmp_path):
    run = _corrida_en_curso(tmp_path)
    paths = graficar([run], tmp_path / "f", tipos=["curvas", "barras", "eval", "progresion"])
    assert {p.name for p in paths} == {"curva_retorno.png", "curva_exito.png", "curva_progreso.png"}
    with pytest.raises(ValueError):
        graficar([run], tmp_path / "f", tipos=["torta"])
    with pytest.raises(ValueError):
        graficar([run], tmp_path / "f", etiquetas=["a", "b"])
    with pytest.raises(FileNotFoundError):
        graficar([tmp_path / "vacia"], tmp_path / "f")


def test_nombre_salida():
    assert nombre_salida(["runs/e1"]) == "e1"
    assert nombre_salida(["runs/e1", "runs/e2", "runs/e3"]) == "comparacion_e1_+2"


def test_cli(tmp_path):
    run = _corrida_completa(tmp_path)
    out = tmp_path / "cli"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "plot.py"), str(run), "--tipo", "barras",
                        "--out", str(out)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["barras_entrenamiento.png", "barras_validacion.png"]

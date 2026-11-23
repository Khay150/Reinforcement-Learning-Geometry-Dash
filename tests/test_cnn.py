"""Tests de las observaciones ``grid`` / ``dict`` con el extractor CNN propio (agents/cnn.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from agents import ON_POLICY, build_model, evaluate_policy_phases, make_train_env
from agents.cnn import GridCNN, GridVecCNN
from agents.common import load_model, policy_for
from gd_env import GeometryDashEnv


def test_policy_for():
    assert policy_for("features") == "MlpPolicy"
    assert policy_for("grid") == "CnnPolicy"
    assert policy_for("dict") == "MultiInputPolicy"
    with pytest.raises(ValueError):
        policy_for("pixeles")


def test_grid_extractor_shape():
    """La grilla 3x7x16 (más chica que lo que acepta NatureCNN) pasa por la CNN y sale un vector de features_dim."""
    space = GeometryDashEnv(obs_mode="grid").observation_space
    ext = GridCNN(space, features_dim=64)
    out = ext(torch.as_tensor(np.stack([space.sample()] * 5)).float())
    assert out.shape == (5, 64) and ext.features_dim == 64


def test_dict_extractor_shape():
    """El extractor de ``dict`` concatena las features de la grilla (128) y del vector (32)."""
    space = GeometryDashEnv(obs_mode="dict").observation_space
    ext = GridVecCNN(space)
    batch = {k: torch.as_tensor(np.stack([v.sample()] * 3)).float() for k, v in space.spaces.items()}
    out = ext(batch)
    assert out.shape == (3, 128 + 32) and ext.features_dim == 160


@pytest.mark.parametrize("algo", ["ppo", "a2c", "dqn", "qrdqn"])
@pytest.mark.parametrize("obs_mode", ["grid", "dict"])
def test_train_save_load_and_evaluate(tmp_path, algo, obs_mode):
    """Cada algoritmo entrena un poco con CNN, se guarda, se recarga (la clase del extractor se serializa) y evalúa."""
    if algo == "qrdqn":
        pytest.importorskip("sb3_contrib")
    on_policy = algo in ON_POLICY
    env = make_train_env(obs_mode, n_envs=2 if on_policy else 1, seed=0, random_phase=False)
    model = build_model(algo, env, seed=0, obs_mode=obs_mode)
    extractor = {"grid": GridCNN, "dict": GridVecCNN}[obs_mode]
    # El extractor vive en lugares distintos según el algoritmo (policy.features_extractor, q_net, quantile_net...):
    # se busca entre todos los módulos de la política.
    assert any(isinstance(m, extractor) for m in model.policy.modules())
    model.learn(total_timesteps=600 if not on_policy else 512)

    model.save(tmp_path / "m")
    loaded = load_model(algo, str(tmp_path / "m.zip"))
    obs, _ = GeometryDashEnv(obs_mode=obs_mode).reset()
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)

    rows = evaluate_policy_phases(loaded, obs_mode, phases=[2.0, 2.5])      # corre episodios completos con la CNN
    assert len(rows) == 2 and all(r["progress"] >= 0 for r in rows)

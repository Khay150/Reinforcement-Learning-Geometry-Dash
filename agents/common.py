"""Construcción de entornos de entrenamiento y modelos de Stable-Baselines3 (PPO / A2C / DQN / QR-DQN)."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from gd_env import GeometryDashEnv

ALGOS = ("ppo", "a2c", "dqn", "qrdqn")
ON_POLICY = ("ppo", "a2c")      # usan varios entornos en paralelo; DQN/QR-DQN usan 1 + replay buffer

# Fases de arranque (x0) con solución comprobada en tests: [2.0, 2.95].
TRAIN_PHASE_RANGE = (2.0, 2.99)
EVAL_PHASES = [float(x) for x in np.linspace(2.0, 2.95, 10)]


class RandomPhase(gym.Wrapper):
    """En cada reset sortea la fase de arranque x0 (entrenar con variación de fase)."""

    def __init__(self, env, low: float = TRAIN_PHASE_RANGE[0], high: float = TRAIN_PHASE_RANGE[1]):
        super().__init__(env)
        self.low, self.high = low, high

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("x0", float(self.env.unwrapped.np_random.uniform(self.low, self.high)))
        return self.env.reset(seed=seed, options=options)


def make_train_env(obs_mode: str, n_envs: int, seed: int, random_phase: bool, monitor_dir: str | None = None,
                   reward_fn=None, jump_penalty: float = 0.0):
    """VecEnv de entrenamiento. ``random_phase=False`` entrena siempre con x0 = 2.0 (nivel fijo)."""
    return make_vec_env(
        GeometryDashEnv, n_envs=n_envs, seed=seed, monitor_dir=monitor_dir,
        env_kwargs={"obs_mode": obs_mode, "reward_fn": reward_fn, "jump_penalty": jump_penalty},
        wrapper_class=RandomPhase if random_phase else None,
        monitor_kwargs={"info_keywords": ("success", "progress")},
    )


def policy_for(obs_mode: str) -> str:
    # features -> MLP. grid/dict necesitan un extractor propio (la grilla 7x16 es más chica que NatureCNN).
    if obs_mode == "features":
        return "MlpPolicy"
    raise NotImplementedError(
        f"obs_mode='{obs_mode}' requiere un extractor CNN propio (policy_kwargs): todavía no implementado.")


def _qrdqn_cls():
    """QR-DQN vive en sb3-contrib; se importa recién al usarlo para no exigirlo si sólo se corre PPO/A2C/DQN."""
    try:
        from sb3_contrib import QRDQN
    except ImportError as e:
        raise ImportError("QR-DQN requiere sb3-contrib: pip install sb3-contrib") from e
    return QRDQN


def build_model(algo: str, env, seed: int, obs_mode: str = "features", tensorboard_log: str | None = None):
    """PPO, A2C, DQN o QR-DQN con hiperparámetros iniciales razonables (punto de partida, no tuneados)."""
    algo = algo.lower()
    common = dict(seed=seed, device="cpu", verbose=0, tensorboard_log=tensorboard_log)
    policy = policy_for(obs_mode)
    if algo == "ppo":
        return PPO(policy, env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                   gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                   policy_kwargs=dict(net_arch=[64, 64]), **common)
    if algo == "a2c":
        # Actor-crítico simple: rollout corto (16 pasos x 8 entornos = 128 muestras), 1 solo paso de gradiente por
        # rollout y sin clipping. Es el control de "qué aporta PPO sobre un actor-crítico simple".
        return A2C(policy, env, learning_rate=7e-4, n_steps=16, gamma=0.99, gae_lambda=0.95,
                   ent_coef=0.01, vf_coef=0.5, policy_kwargs=dict(net_arch=[64, 64]), **common)
    dqn_kwargs = dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=5_000,
                      batch_size=64, gamma=0.99, train_freq=4, gradient_steps=1,
                      target_update_interval=500, exploration_fraction=0.3, exploration_final_eps=0.05)
    if algo == "dqn":
        return DQN(policy, env, policy_kwargs=dict(net_arch=[64, 64]), **dqn_kwargs, **common)
    if algo == "qrdqn":
        # Mismos hiperparámetros que DQN; sólo cambia la cabeza (n_quantiles cuantiles por acción en vez de 1 valor).
        return _qrdqn_cls()(policy, env, policy_kwargs=dict(net_arch=[64, 64], n_quantiles=50),
                            **dqn_kwargs, **common)
    raise ValueError(f"algoritmo desconocido: {algo} (usar {', '.join(ALGOS)})")


def load_model(algo: str, path: str):
    algo = algo.lower()
    cls = _qrdqn_cls() if algo == "qrdqn" else {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo]
    return cls.load(path, device="cpu")

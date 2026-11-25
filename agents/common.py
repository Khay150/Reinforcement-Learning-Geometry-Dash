"""Construcción de entornos de entrenamiento y modelos de Stable-Baselines3 (PPO / A2C / DQN / QR-DQN)."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from gd_env import GeometryDashEnv

from .cnn import GridCNN, GridVecCNN

ALGOS = ("ppo", "a2c", "dqn", "qrdqn")         # algoritmos disponibles (valores de --algos)
ON_POLICY = ("ppo", "a2c")      # usan varios entornos en paralelo; DQN/QR-DQN usan 1 + replay buffer

# Fases de arranque (x0) con solución comprobada en tests: [2.0, 2.95].
# TRAIN_PHASE_RANGE: de dónde sortea x0 el modo --train-phase random. EVAL_PHASES: las 10 fases fijas de evaluación.
TRAIN_PHASE_RANGE = (2.0, 2.99)
EVAL_PHASES = [float(x) for x in np.linspace(2.0, 2.95, 10)]


class RandomPhase(gym.Wrapper):
    """En cada reset sortea la fase de arranque x0 (entrenar con variación de fase)."""

    def __init__(self, env, low: float = TRAIN_PHASE_RANGE[0], high: float = TRAIN_PHASE_RANGE[1]):
        super().__init__(env)
        self.low, self.high = low, high

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        # setdefault: si quien llama ya pidió un x0 concreto (p. ej. al evaluar), se respeta; si no, se sortea uno.
        options.setdefault("x0", float(self.env.unwrapped.np_random.uniform(self.low, self.high)))
        return self.env.reset(seed=seed, options=options)


def make_train_env(obs_mode: str, n_envs: int, seed: int, random_phase: bool, monitor_dir: str | None = None,
                   **env_kwargs):
    """VecEnv de entrenamiento. ``random_phase=False`` entrena siempre con x0 = 2.0 (nivel fijo).
    ``env_kwargs`` va directo a ``GeometryDashEnv`` (p. ej. ``jump_penalty``, ``death_penalty``, ``reward_fn``)."""
    # make_vec_env crea ``n_envs`` copias del entorno (cada una envuelta en un Monitor que registra los episodios).
    # ``info_keywords`` agrega "success", "progress" y "level" a las columnas de los CSV de monitor/.
    return make_vec_env(
        GeometryDashEnv, n_envs=n_envs, seed=seed, monitor_dir=monitor_dir,
        env_kwargs={"obs_mode": obs_mode, **env_kwargs},
        wrapper_class=RandomPhase if random_phase else None,
        monitor_kwargs={"info_keywords": ("success", "progress", "level")},
    )


def policy_for(obs_mode: str) -> str:
    """Nombre de la política de SB3 según el tipo de observación.

    * ``features`` -> ``MlpPolicy``        (vector plano -> red densa)
    * ``grid``     -> ``CnnPolicy``        (imagen 3x7x16 -> CNN; el extractor propio se pone en ``policy_kwargs_for``)
    * ``dict``     -> ``MultiInputPolicy`` (grilla + vector -> un extractor por clave, ver ``agents/cnn.py``)
    """
    try:
        return {"features": "MlpPolicy", "grid": "CnnPolicy", "dict": "MultiInputPolicy"}[obs_mode]
    except KeyError:
        raise ValueError(f"obs_mode desconocido: {obs_mode!r} (usar features, grid o dict)") from None


def policy_kwargs_for(obs_mode: str, extra: dict | None = None) -> dict:
    """``policy_kwargs`` de SB3 para cada tipo de observación.

    ``net_arch=[64, 64]`` son las capas ocultas que van *después* del extractor. Con ``grid``/``dict`` se agrega
    el extractor CNN propio: la ``CnnPolicy`` por defecto (NatureCNN) exige imágenes de al menos 36x36 y la grilla
    es de 3x7x16. ``extra`` se mezcla al final (p. ej. ``features_extractor_kwargs=dict(features_dim=64)``).
    """
    kwargs = dict(net_arch=[64, 64])
    if obs_mode == "grid":
        kwargs["features_extractor_class"] = GridCNN
    elif obs_mode == "dict":
        kwargs["features_extractor_class"] = GridVecCNN
    return {**kwargs, **(extra or {})}


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
    # Argumentos comunes a todos los algoritmos. device="cpu": la red es chica y en CPU suele ser igual de rápido.
    common = dict(seed=seed, device="cpu", verbose=0, tensorboard_log=tensorboard_log)
    policy = policy_for(obs_mode)
    pk = policy_kwargs_for(obs_mode)       # extractor CNN (grid/dict) + capas ocultas
    if algo == "ppo":
        return PPO(policy, env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                   gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                   policy_kwargs=pk, **common)
    if algo == "a2c":
        # Actor-crítico simple: rollout corto (16 pasos x 8 entornos = 128 muestras), 1 solo paso de gradiente por
        # rollout y sin clipping. Es el control de "qué aporta PPO sobre un actor-crítico simple".
        return A2C(policy, env, learning_rate=7e-4, n_steps=16, gamma=0.99, gae_lambda=0.95,
                   ent_coef=0.01, vf_coef=0.5, policy_kwargs=pk, **common)
    # DQN y QR-DQN comparten hiperparámetros: replay buffer de 100k transiciones, 5000 pasos de relleno antes de aprender,
    # ε-greedy de 1.0 a 0.05 durante el primer 30% del entrenamiento, y red objetivo actualizada cada 500 pasos.
    dqn_kwargs = dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=5_000,
                      batch_size=64, gamma=0.99, train_freq=4, gradient_steps=1,
                      target_update_interval=500, exploration_fraction=0.3, exploration_final_eps=0.05)
    if algo == "dqn":
        return DQN(policy, env, policy_kwargs=pk, **dqn_kwargs, **common)
    if algo == "qrdqn":
        # Mismos hiperparámetros que DQN; sólo cambia la cabeza (n_quantiles cuantiles por acción en vez de 1 valor).
        return _qrdqn_cls()(policy, env, policy_kwargs={**pk, "n_quantiles": 50},
                            **dqn_kwargs, **common)
    raise ValueError(f"algoritmo desconocido: {algo} (usar {', '.join(ALGOS)})")


def load_model(algo: str, path: str, env=None):
    """Carga un modelo guardado (``env`` sólo hace falta para seguir entrenándolo)."""
    algo = algo.lower()
    # Si el modelo se entrenó con grid/dict, al cargarlo SB3 necesita importar agents.cnn (guarda la clase del extractor).
    cls = _qrdqn_cls() if algo == "qrdqn" else {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo]
    return cls.load(path, env=env, device="cpu")


def load_for_training(algo: str, path: str, env, seed: int):
    """Carga un modelo para **seguir entrenándolo** sobre ``env`` (p. ej. para hacer fine-tuning con otra recompensa).

    * Se conservan los pesos y los hiperparámetros guardados en el ``.zip`` (no los de ``build_model``).
    * ``env`` puede tener otra cantidad de entornos en paralelo que el original.
    * DQN / QR-DQN: el replay buffer no se guarda en el ``.zip``, así que arranca vacío (vuelven a correr los
      ``learning_starts`` pasos sin entrenar). La exploración ε arranca directamente en su valor final (0.05):
      si volviera a 1.0 jugaría al azar durante el 30% del entrenamiento y desaprovecharía el modelo.
    """
    model = load_model(algo, path, env=env)
    model.set_random_seed(seed)
    if algo.lower() not in ON_POLICY:
        eps = model.exploration_final_eps
        model.exploration_schedule = lambda _progress: eps     # ε constante en vez de volver a 1.0
    return model

"""Extractores de features con CNN para las observaciones ``grid`` y ``dict`` (Stable-Baselines3).

Por qué hace falta un extractor propio
--------------------------------------
La ``CnnPolicy`` de SB3 usa por defecto ``NatureCNN`` (3 convoluciones de 8x8/4, 4x4/2 y 3x3/1), pensada para
imágenes de Atari de al menos 36x36 píxeles. La observación ``grid`` de este entorno es de **3 x 7 x 16**
(canales: bloques / pinchos / jugador; alto = altura del nivel; ancho = ``window``), así que con esos strides
la imagen se "acaba" antes de la última convolución y NatureCNN falla. Estas clases reemplazan a NatureCNN:

* ``GridCNN``    : para ``obs_mode="grid"``  (Box). Dos convoluciones 3x3 con ``padding=1`` (no achican la
                   grilla) + una capa lineal que resume todo en ``features_dim`` números.
* ``GridVecCNN`` : para ``obs_mode="dict"``  ({"grid", "vec"}). Procesa la grilla con la misma CNN y el vector
                   ``vec`` = [y/H, vy/jump_v, en_suelo, frac(x)] con una capa lineal, y concatena los dos.
                   El vector aporta la velocidad vertical, que la grilla sola no trae.

Cómo se usan
------------
No se instancian a mano: ``agents.common.build_model`` las pasa a SB3 vía
``policy_kwargs=dict(features_extractor_class=..., features_extractor_kwargs=...)``. La política de SB3 queda
así:  observación -> extractor CNN -> (capas ocultas ``net_arch``) -> acción / valor.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _grid_cnn(grid_space: gym.spaces.Box, channels: tuple[int, int], features_dim: int) -> nn.Sequential:
    """CNN para una grilla (C, H, W): conv3x3 -> ReLU -> conv3x3 -> ReLU -> Flatten -> Linear -> ReLU.

    ``padding=1`` mantiene el tamaño espacial (H x W) en las dos convoluciones: con una grilla tan chica no
    conviene achicarla con strides ni pooling, porque cada celda importa (un pincho corrido una columna es otro salto).
    El tamaño de la capa lineal se calcula pasando una observación de ejemplo por la parte convolucional.
    """
    n_channels = grid_space.shape[0]
    conv = nn.Sequential(
        nn.Conv2d(n_channels, channels[0], kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():                                         # ejemplo con batch de 1 -> tamaño aplanado
        n_flat = conv(torch.as_tensor(grid_space.sample()[None]).float()).shape[1]
    return nn.Sequential(conv, nn.Linear(n_flat, features_dim), nn.ReLU())


class GridCNN(BaseFeaturesExtractor):
    """Extractor para ``obs_mode="grid"``: entrada (B, 3, H, W) con 0/1 -> salida (B, features_dim).

    Parámetros (se pasan por ``features_extractor_kwargs``):
      * ``features_dim``: tamaño del vector que sale del extractor (por defecto 128).
      * ``channels``: filtros de las dos convoluciones (por defecto (32, 64)).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128,
                 channels: tuple[int, int] = (32, 64)):
        super().__init__(observation_space, features_dim)
        self.net = _grid_cnn(observation_space, channels, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 ya convirtió la observación (uint8 0/1) a float; no se normaliza porque el Box es 0..1, no 0..255.
        return self.net(observations)


class GridVecCNN(BaseFeaturesExtractor):
    """Extractor para ``obs_mode="dict"``: {"grid": (B,3,H,W), "vec": (B,4)} -> (B, features_dim + vec_dim).

    Cada parte se procesa por separado y las features se concatenan:
      * ``grid`` -> CNN (igual que ``GridCNN``) -> ``features_dim`` números.
      * ``vec``  -> Linear -> ReLU -> ``vec_dim`` números.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128,
                 channels: tuple[int, int] = (32, 64), vec_dim: int = 32):
        super().__init__(observation_space, features_dim + vec_dim)    # la dimensión de salida es la suma
        self.grid_net = _grid_cnn(observation_space["grid"], channels, features_dim)
        self.vec_net = nn.Sequential(nn.Linear(observation_space["vec"].shape[0], vec_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.grid_net(observations["grid"]), self.vec_net(observations["vec"])], dim=1)

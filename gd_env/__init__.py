from gymnasium.envs.registration import register

from .env import GeometryDashEnv
from .levels import build_level, level_1, parse_level, to_ascii
from .oracle import Oracle, OraclePolicy, solve
from .physics import Level, Physics
from .rewards import Transition, adapt_reward_fn, default_reward, load_reward_fn, reward_spec
from .render import record_episode, render_frame, trajectory_image

register(id="GeometryDash-v0", entry_point="gd_env.env:GeometryDashEnv")

__all__ = ["GeometryDashEnv", "Level", "Physics", "level_1", "build_level", "parse_level",
           "to_ascii", "solve", "Oracle", "OraclePolicy",
           "Transition", "default_reward", "load_reward_fn", "reward_spec",
           "record_episode", "render_frame", "trajectory_image"]

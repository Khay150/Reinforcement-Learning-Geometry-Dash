# API pública del paquete: lo que se importa con ``from agents import ...``.
# El extractor CNN (grid/dict) vive en agents/cnn.py y lo usa build_model automáticamente.
from .common import ALGOS, EVAL_PHASES, ON_POLICY, TRAIN_PHASE_RANGE, build_model, make_train_env
from .evaluate import evaluate_policy_phases, summarize

__all__ = ["ALGOS", "ON_POLICY", "EVAL_PHASES", "TRAIN_PHASE_RANGE", "build_model", "make_train_env",
           "evaluate_policy_phases", "summarize"]

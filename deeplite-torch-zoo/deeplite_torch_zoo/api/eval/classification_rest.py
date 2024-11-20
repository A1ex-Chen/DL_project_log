import torch
from tqdm import tqdm

from deeplite_torch_zoo.utils import switch_train_mode
from deeplite_torch_zoo.api.registries import EVAL_WRAPPER_REGISTRY

__all__ = ['classification_eval']


@EVAL_WRAPPER_REGISTRY.register(task_type='classification')
import torch

from deeplite_torch_zoo.src.object_detection.eval.evaluate import evaluate
from deeplite_torch_zoo.api.registries import EVAL_WRAPPER_REGISTRY


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection')
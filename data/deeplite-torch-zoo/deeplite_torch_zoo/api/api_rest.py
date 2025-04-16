import collections
import fnmatch

import texttable
import torch

import deeplite_torch_zoo.api.datasets  # pylint: disable=unused-import
import deeplite_torch_zoo.api.eval  # pylint: disable=unused-import
import deeplite_torch_zoo.api.models  # pylint: disable=unused-import
from deeplite_torch_zoo.utils import switch_train_mode, deprecated, LOGGER
from deeplite_torch_zoo.utils.profiler import profile_macs, profile_ram
from deeplite_torch_zoo.api.registries import (
    DATASET_WRAPPER_REGISTRY,
    EVAL_WRAPPER_REGISTRY,
    MODEL_WRAPPER_REGISTRY,
)

__all__ = [
    "get_dataloaders",
    "get_model",
    "create_model",
    "get_eval_function",
    "list_models",
    "profile",
    "list_models_by_dataset",
    # deprecated API:
    "get_model_by_name",
    "get_data_splits_by_name",
]














@deprecated


@deprecated


@deprecated
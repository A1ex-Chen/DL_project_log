# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module):
    """
    A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training,
    validation, prediction, exporting, and benchmarking. It handles different types of models, including those
    loaded from local files, Ultralytics HUB, or Triton Server. The class is designed to be flexible and
    extendable for different tasks and model configurations.

    Args:
        model (Union[str, Path], optional): Path or name of the model to load or create. This can be a local file
            path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to 'yolov8n.pt'.
        task (Any, optional): The task type associated with the YOLO model. This can be used to specify the model's
            application domain, such as object detection, segmentation, etc. Defaults to None.
        verbose (bool, optional): If True, enables verbose output during the model's operations. Defaults to False.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (dict): The latest training/validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initializes a new model based on a configuration file.
        _load: Loads a model from a checkpoint file.
        _check_is_pytorch_model: Ensures that the model is a PyTorch model.
        reset_weights: Resets the model's weights to their initial state.
        load: Loads model weights from a specified file.
        save: Saves the current state of the model to a file.
        info: Logs or returns information about the model.
        fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference.
        predict: Performs object detection predictions.
        track: Performs object tracking.
        val: Validates the model on a dataset.
        benchmark: Benchmarks the model on various export formats.
        export: Exports the model to different formats.
        train: Trains the model on a dataset.
        tune: Performs hyperparameter tuning.
        _apply: Applies a function to the model's tensors.
        add_callback: Adds a callback function for an event.
        clear_callback: Clears all callbacks for an event.
        reset_callbacks: Resets all callbacks to their default functions.
        is_triton_model: Checks if a model is a Triton Server model.
        is_hub_model: Checks if a model is an Ultralytics HUB model.
        _reset_ckpt_args: Resets checkpoint arguments when loading a PyTorch model.
        _smart_load: Loads the appropriate module based on the model task.
        task_map: Provides a mapping from model tasks to corresponding classes.

    Raises:
        FileNotFoundError: If the specified model file does not exist or is inaccessible.
        ValueError: If the model file or configuration is invalid or unsupported.
        ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
        TypeError: If the model is not a PyTorch model when required.
        AttributeError: If required attributes or methods are not implemented or available.
        NotImplementedError: If a specific model task or mode is not supported.
    """



    @staticmethod

    @staticmethod


















    @property

    @property

    @property




    @staticmethod

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


    @property
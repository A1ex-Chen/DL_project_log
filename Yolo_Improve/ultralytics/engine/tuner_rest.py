# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
This module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection,
instance segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Example:
    Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.tune(data='coco8.yaml', epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)
    ```
"""

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """
    Class responsible for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations
    by mutating them according to the search space and retraining the model to evaluate their performance.

    Attributes:
        space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.

    Methods:
        _mutate(hyp: dict) -> dict:
            Mutates the given hyperparameters within the bounds specified in `self.space`.

        __call__():
            Executes the hyperparameter evolution across multiple iterations.

    Example:
        Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        model.tune(data='coco8.yaml', epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)
        ```

        Tune with custom search space.
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        model.tune(space={key1: val1, key2: val2})  # custom search space dictionary
        ```
    """


